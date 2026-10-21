from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiocqhttp import Event

from astrbot.core.message.components import Plain
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.astr_message_event import current_message_event
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform import Platform
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_platform_adapter import (
    AiocqhttpAdapter,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_type", "session_id"),
    [
        (MessageType.FRIEND_MESSAGE, "456"),
        (MessageType.GROUP_MESSAGE, "789"),
        (MessageType.GROUP_MESSAGE, "456_789"),
    ],
    ids=["private", "group", "isolated-group"],
)
async def test_current_session_tool_send_preserves_receipt_and_account_routing(
    message_type, session_id
):
    observer = AsyncMock()
    raw = Event.from_payload(
        {
            "post_type": "message",
            "message_type": "group"
            if message_type == MessageType.GROUP_MESSAGE
            else "private",
            "self_id": 123,
            "user_id": 456,
            "group_id": 789,
            "sender": {"user_id": 456},
            "message_id": 1,
            "message": [{"type": "text", "data": {"text": "Question"}}],
        }
    )
    source = SimpleNamespace(
        _execution_active=True,
        message_obj=SimpleNamespace(raw_message=raw),
        get_platform_name=lambda: "aiocqhttp",
        get_platform_id=lambda: "qq",
        get_message_type=lambda: message_type,
        get_group_id=lambda: "789" if message_type == MessageType.GROUP_MESSAGE else "",
        get_sender_id=lambda: "456",
        get_extra=lambda _key: observer,
    )
    adapter = AiocqhttpAdapter.__new__(AiocqhttpAdapter)
    adapter.metadata = SimpleNamespace(id="qq")
    adapter.bot = AsyncMock()
    sender = (
        adapter.bot.send_group_msg
        if message_type == MessageType.GROUP_MESSAGE
        else adapter.bot.send_private_msg
    )
    sender.return_value = {"message_id": 42}
    token = current_message_event.set(source)
    try:
        with patch.object(Platform, "send_by_session", new=AsyncMock()):
            await adapter.send_by_session(
                MessageSession("qq", message_type, session_id),
                MessageChain([Plain("Tool reply")]),
            )
    finally:
        current_message_event.reset(token)

    expected_target = (
        {"group_id": 789}
        if message_type == MessageType.GROUP_MESSAGE
        else {"user_id": 456}
    )
    sender.assert_awaited_once_with(
        **expected_target,
        self_id=123,
        message=[{"type": "text", "data": {"text": "Tool reply"}}],
    )
    observer.assert_awaited_once_with(
        {
            "message_id": "42",
            "message": [{"type": "text", "data": {"text": "Tool reply"}}],
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatch",
    [
        "unbound",
        "inactive",
        "platform_type",
        "platform_id",
        "target_platform",
        "adapter_platform",
        "destination",
        "chat_type",
    ],
)
@pytest.mark.parametrize(
    "message_type", [MessageType.FRIEND_MESSAGE, MessageType.GROUP_MESSAGE]
)
async def test_unbound_or_cross_target_send_cannot_inherit_current_provenance(
    mismatch, message_type
):
    observer = AsyncMock()
    source = SimpleNamespace(
        _execution_active=mismatch != "inactive",
        message_obj=SimpleNamespace(raw_message=Event.from_payload({"self_id": 123})),
        get_platform_name=lambda: (
            "other" if mismatch == "platform_type" else "aiocqhttp"
        ),
        get_platform_id=lambda: "other" if mismatch == "platform_id" else "qq",
        get_message_type=lambda: (
            MessageType.OTHER_MESSAGE if mismatch == "chat_type" else message_type
        ),
        get_group_id=lambda: "999" if mismatch == "destination" else "789",
        get_sender_id=lambda: "999" if mismatch == "destination" else "456",
        get_extra=lambda _key: observer,
    )
    adapter = AiocqhttpAdapter.__new__(AiocqhttpAdapter)
    adapter.metadata = SimpleNamespace(
        id="other" if mismatch == "adapter_platform" else "qq"
    )
    adapter.bot = AsyncMock()
    adapter.bot.send_group_msg.return_value = {"message_id": 42}
    adapter.bot.send_private_msg.return_value = {"message_id": 42}
    token = current_message_event.set(None if mismatch == "unbound" else source)
    try:
        with patch.object(Platform, "send_by_session", new=AsyncMock()):
            await adapter.send_by_session(
                MessageSession(
                    "other" if mismatch == "target_platform" else "qq",
                    message_type,
                    "789" if message_type == MessageType.GROUP_MESSAGE else "456",
                ),
                MessageChain([Plain("Cross-target reply")]),
            )
    finally:
        current_message_event.reset(token)

    observer.assert_not_awaited()
    sender = (
        adapter.bot.send_group_msg
        if message_type == MessageType.GROUP_MESSAGE
        else adapter.bot.send_private_msg
    )
    assert "self_id" not in sender.await_args.kwargs
