import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from astrbot.builtin_stars.builtin_commands.commands import (
    conversation as conversation_module,
)
from astrbot.core.agent.message import Message
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.utils.active_event_registry import ActiveEventRegistry
from astrbot.core.utils.session_lock import SessionLockManager


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["FriendMessage", "GroupMessage"])
@pytest.mark.parametrize("reset", [False, True], ids=["stop", "reset"])
async def test_aborted_turn_cannot_restore_history_after_reset(chat_type, reset):
    umo = f"qq:{chat_type}:456"

    class Event:
        def __init__(self):
            self.unified_msg_origin = umo
            self.stopped = False
            self.extras = {}

        def stop_event(self):
            self.stopped = True

        def is_stopped(self):
            return self.stopped

        def get_extra(self, key):
            return self.extras.get(key)

        def set_extra(self, key, value):
            self.extras[key] = value

    event = Event()
    registry = ActiveEventRegistry()
    registry.register(event)
    history = [Message(role="user", content="Before reset")]
    if reset:
        registry.stop_all(umo)
        stored_history = []
    else:
        registry.request_agent_stop_all(umo)
        stored_history = [message.model_dump() for message in history]

    async def update(_umo, _cid, *, history, token_usage):
        stored_history[:] = history

    stage = InternalAgentSubStage()
    stage.conv_manager = SimpleNamespace(
        update_conversation=AsyncMock(side_effect=update)
    )
    request = SimpleNamespace(
        conversation=SimpleNamespace(cid="current-conversation"), tool_calls_result=None
    )
    await stage._save_to_history(
        event,
        request,
        LLMResponse(role="assistant", completion_text="Partial reply"),
        history + [Message(role="assistant", content="Partial reply")],
        runner_stats=None,
        user_aborted=True,
    )

    if reset:
        assert stored_history == []
        stage.conv_manager.update_conversation.assert_not_awaited()
    else:
        assert stored_history[-1] == {"role": "assistant", "content": "Partial reply"}
        stage.conv_manager.update_conversation.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["FriendMessage", "GroupMessage"])
async def test_reset_waits_for_inflight_history_write_before_clearing(
    monkeypatch, chat_type
):
    umo = f"qq:{chat_type}:456"
    locks = SessionLockManager()
    registry = ActiveEventRegistry()
    save_started, release_save, stop_requested = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    extras = {}
    active = Mock(unified_msg_origin=umo)
    active.is_stopped.return_value = False
    active.get_extra.side_effect = extras.get
    active.set_extra.side_effect = extras.__setitem__

    def stop_event():
        active.is_stopped.return_value = True
        stop_requested.set()

    active.stop_event.side_effect = stop_event
    registry.register(active)
    stored_history = []

    async def update(_umo, _cid, *, history, token_usage=None):
        if history:
            save_started.set()
            await release_save.wait()
        stored_history[:] = history

    manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value="current-conversation"),
        update_conversation=AsyncMock(side_effect=update),
    )
    stage = InternalAgentSubStage()
    stage.conv_manager = manager
    context = SimpleNamespace(
        conversation_manager=manager,
        get_config=lambda **_kwargs: {"agent_runner": {"runner_type": "local"}},
    )
    commands = conversation_module.ConversationCommands(context)
    reset_event = Mock(unified_msg_origin=umo)
    monkeypatch.setattr(conversation_module, "active_event_registry", registry)
    monkeypatch.setattr(conversation_module, "session_lock_manager", locks)

    async def finish_inflight_save():
        async with locks.acquire_lock(umo):
            await stage._save_to_history(
                active,
                SimpleNamespace(
                    conversation=SimpleNamespace(cid="current-conversation"),
                    tool_calls_result=None,
                ),
                LLMResponse(role="assistant", completion_text="Old reply"),
                [Message(role="assistant", content="Old reply")],
                runner_stats=None,
            )

    save_task = asyncio.create_task(finish_inflight_save())
    await asyncio.wait_for(save_started.wait(), timeout=1)
    reset_task = asyncio.create_task(commands.reset(reset_event))
    try:
        await asyncio.wait_for(stop_requested.wait(), timeout=1)
        await asyncio.sleep(0)
        assert not reset_task.done()
        assert manager.update_conversation.await_count == 1
    finally:
        release_save.set()
        await asyncio.wait_for(asyncio.gather(save_task, reset_task), timeout=1)

    assert stored_history == []
    assert manager.update_conversation.await_args.kwargs["history"] == []
    reset_event.set_result.assert_called_once()
