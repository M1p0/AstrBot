from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from astrbot.core import conversation_mgr


@pytest.fixture
def context_manager(monkeypatch):
    db = AsyncMock()
    db.create_conversation.return_value = SimpleNamespace(conversation_id="new")
    manager = conversation_mgr.ConversationManager(db)
    for method in ("session_get", "session_put", "session_remove"):
        monkeypatch.setattr(conversation_mgr.sp, method, AsyncMock(return_value=None))
    return manager


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["FriendMessage", "GroupMessage"])
@pytest.mark.parametrize(
    "operation", ["new", "switch", "clear", "delete", "delete_all"]
)
async def test_context_reset_callback_runs_after_successful_management_change(
    context_manager, chat_type, operation
):
    manager = context_manager
    umo = f"qq:{chat_type}:456"
    manager.session_conversations[umo] = "current"

    observed = []
    listener = AsyncMock(
        side_effect=lambda affected: observed.append(
            {
                "umo": affected,
                "selected": manager.session_conversations.get(umo),
                "put_count": conversation_mgr.sp.session_put.await_count,
                "remove_count": conversation_mgr.sp.session_remove.await_count,
                "updated_history": manager.db.update_conversation.await_args.kwargs[
                    "content"
                ]
                if manager.db.update_conversation.await_args
                else None,
            }
        )
    )
    manager.register_on_context_reset(listener)
    if operation == "new":
        assert await manager.new_conversation(umo) == "new"
    elif operation == "switch":
        await manager.switch_conversation(umo, "new")
    elif operation == "clear":
        await manager.update_conversation(umo, history=[])
    elif operation == "delete":
        await manager.delete_conversation(umo)
    else:
        await manager.delete_conversations_by_user_id(umo)

    listener.assert_awaited_once_with(umo)
    assert observed[0]["umo"] == umo
    if operation in {"new", "switch"}:
        assert observed[0]["selected"] == "new"
        assert observed[0]["put_count"] == 1
    elif operation == "clear":
        assert observed[0]["updated_history"] == []
    else:
        assert observed[0]["selected"] is None
        assert observed[0]["remove_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["FriendMessage", "GroupMessage"])
@pytest.mark.parametrize(
    "operation",
    [
        "first_new",
        "save",
        "title",
        "persona",
        "delete_inactive",
        "clear_inactive",
        "no_clear",
    ],
)
async def test_ordinary_history_and_unselected_conversations_do_not_reset_context(
    context_manager, chat_type, operation
):
    manager = context_manager
    umo = f"qq:{chat_type}:456"
    manager.session_conversations[umo] = "current"
    listener = AsyncMock()
    manager.register_on_context_reset(listener)
    if operation == "first_new":
        manager.session_conversations.clear()
        await manager.new_conversation(umo)
    elif operation == "save":
        await manager.update_conversation(
            umo, history=[{"role": "user", "content": "Normal turn"}]
        )
    elif operation == "title":
        await manager.update_conversation(umo, title="New title")
    elif operation == "persona":
        await manager.update_conversation(umo, persona_id="persona")
    elif operation == "delete_inactive":
        await manager.delete_conversation(umo, "inactive")
        assert manager.session_conversations[umo] == "current"
    elif operation == "clear_inactive":
        await manager.update_conversation(umo, "inactive", history=[])
        assert manager.session_conversations[umo] == "current"
    else:
        manager.session_conversations.clear()
        await manager.update_conversation(umo, history=[])

    listener.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_reset_listener_registration_and_unloading_are_idempotent(
    context_manager,
):
    manager = context_manager
    listener = AsyncMock()
    manager.register_on_context_reset(listener)
    manager.register_on_context_reset(listener)

    await manager.switch_conversation("qq:GroupMessage:456", "new")
    listener.assert_awaited_once()

    manager.unregister_on_context_reset(listener)
    manager.unregister_on_context_reset(listener)
    await manager.switch_conversation("qq:GroupMessage:456", "other")
    listener.assert_awaited_once()


@pytest.mark.asyncio
async def test_failing_context_listener_does_not_fail_change_or_other_listeners(
    context_manager,
):
    manager = context_manager
    failing = AsyncMock(side_effect=RuntimeError("Sensitive source content"))
    succeeding = AsyncMock()
    manager.register_on_context_reset(failing)
    manager.register_on_context_reset(succeeding)

    with patch.object(conversation_mgr.logger, "warning") as warning:
        await manager.switch_conversation("qq:FriendMessage:456", "new")

    succeeding.assert_awaited_once_with("qq:FriendMessage:456")
    assert manager.session_conversations["qq:FriendMessage:456"] == "new"
    warning.assert_called_once_with(
        "Conversation context reset callback failed: RuntimeError"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["new", "switch", "clear", "delete", "delete_all"]
)
async def test_failed_management_operation_does_not_signal_context_reset(
    context_manager, operation
):
    manager = context_manager
    umo = "qq:GroupMessage:456"
    manager.session_conversations[umo] = "current"
    listener = AsyncMock()
    manager.register_on_context_reset(listener)
    failure = RuntimeError("Management operation failed")
    if operation == "new":
        manager.db.create_conversation.side_effect = failure
    elif operation == "switch":
        conversation_mgr.sp.session_put.side_effect = failure
    elif operation == "clear":
        manager.db.update_conversation.side_effect = failure
    elif operation == "delete":
        manager.db.delete_conversation.side_effect = failure
    else:
        manager.db.delete_conversations_by_user_id.side_effect = failure

    with pytest.raises(RuntimeError, match="Management operation failed"):
        if operation == "new":
            await manager.new_conversation(umo)
        elif operation == "switch":
            await manager.switch_conversation(umo, "new")
        elif operation == "clear":
            await manager.update_conversation(umo, history=[])
        elif operation == "delete":
            await manager.delete_conversation(umo)
        else:
            await manager.delete_conversations_by_user_id(umo)

    listener.assert_not_awaited()
