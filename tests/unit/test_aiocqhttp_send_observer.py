from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from astrbot.core.message.components import At, File, Node, Plain
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_group", [False, True])
async def test_send_observer_receives_confirmed_id_and_exact_wire_segments(is_group):
    bot = AsyncMock()
    bot.send_group_msg.return_value = {"message_id": 42}
    bot.send_private_msg.return_value = {"message_id": 42}
    observer = AsyncMock()

    await AiocqhttpMessageEvent.send_message(
        bot,
        MessageChain([At(qq="123"), Plain("A reply")]),
        is_group=is_group,
        session_id="456",
        send_observer=observer,
    )

    sender = bot.send_group_msg if is_group else bot.send_private_msg
    wire = sender.await_args.kwargs["message"]
    observer.assert_awaited_once_with({"message_id": "42", "message": wire})
    assert wire == [
        {"type": "at", "data": {"qq": "123"}},
        {"type": "text", "data": {"text": " "}},
        {"type": "text", "data": {"text": "A reply"}},
    ]
    observer.await_args.args[0]["message"][0]["data"]["qq"] = "999"
    assert wire[0]["data"]["qq"] == "123"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_group", [False, True])
async def test_forward_and_file_sends_report_individual_receipts(is_group, tmp_path):
    bot = AsyncMock()
    bot.call_action.return_value = {"message_id": 10}
    sender = bot.send_group_msg if is_group else bot.send_private_msg
    sender.side_effect = [{"message_id": 11}, {"message_id": 12}]
    file_path = tmp_path / "sent.txt"
    file_path.write_text("file contents", encoding="utf-8")
    observer = AsyncMock()

    with patch(
        "astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event.asyncio.sleep",
        new=AsyncMock(),
    ):
        await AiocqhttpMessageEvent.send_message(
            bot,
            MessageChain(
                [
                    Node([Plain("Forwarded claim")], uin="321", name="Participant"),
                    File(name="sent.txt", file=str(file_path)),
                    Plain("Outer reply"),
                ]
            ),
            is_group=is_group,
            session_id="456",
            send_observer=observer,
        )

    receipts = [call.args[0] for call in observer.await_args_list]
    assert [receipt["message_id"] for receipt in receipts] == ["10", "11", "12"]
    assert receipts[0]["message"] == bot.call_action.await_args.kwargs["messages"]
    assert receipts[0]["message"][0]["data"]["user_id"] == "321"
    assert receipts[1]["message"] == sender.await_args_list[0].kwargs["message"]
    assert receipts[2]["message"] == [{"type": "text", "data": {"text": "Outer reply"}}]


@pytest.mark.asyncio
async def test_receipt_is_reported_before_a_later_segment_fails(tmp_path):
    bot = AsyncMock()
    bot.send_private_msg.side_effect = [
        {"message_id": 10},
        RuntimeError("Later delivery failed"),
    ]
    file_path = tmp_path / "sent.txt"
    file_path.write_text("file contents", encoding="utf-8")
    observer = AsyncMock()

    with pytest.raises(RuntimeError, match="Later delivery failed"):
        await AiocqhttpMessageEvent.send_message(
            bot,
            MessageChain([File(name="sent.txt", file=str(file_path)), Plain("Reply")]),
            session_id="456",
            send_observer=observer,
        )

    assert observer.await_count == 1
    assert observer.await_args.args[0]["message_id"] == "10"


@pytest.mark.asyncio
@pytest.mark.parametrize("forwarded", [False, True])
async def test_observer_failure_does_not_retry_or_fail_delivery(forwarded):
    bot = AsyncMock()
    bot.send_private_msg.return_value = {"message_id": 42}
    bot.call_action.return_value = {"message_id": 42}
    observer = AsyncMock(side_effect=RuntimeError("Sensitive callback details"))
    components = [Node([Plain("Forward")])] if forwarded else [Plain("Reply")]

    with patch(
        "astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event.logger.warning"
    ) as warning:
        await AiocqhttpMessageEvent.send_message(
            bot,
            MessageChain(components),
            session_id="456",
            send_observer=observer,
        )

    assert observer.await_count == 1
    sender = bot.call_action if forwarded else bot.send_private_msg
    assert sender.await_count == 1
    assert warning.call_args.args == ("[aiocqhttp] Send observer failed: RuntimeError",)


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [None, {}, {"message_id": None}])
async def test_unconfirmed_send_does_not_report_a_receipt(receipt):
    bot = AsyncMock()
    bot.send_private_msg.return_value = receipt
    observer = AsyncMock()

    await AiocqhttpMessageEvent.send_message(
        bot,
        MessageChain([Plain("Reply")]),
        session_id="456",
        send_observer=observer,
    )

    observer.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("is_group", [False, True])
async def test_event_local_observer_covers_streaming_sends(fallback, is_group):
    event = AiocqhttpMessageEvent.__new__(AiocqhttpMessageEvent)
    observer = AsyncMock()
    event._extras = {"_aiocqhttp_send_observer": observer}
    event.message_obj = SimpleNamespace(
        group_id="123" if is_group else "",
        sender=SimpleNamespace(user_id="456"),
        raw_message=None,
    )
    event.bot = AsyncMock()
    sender = event.bot.send_group_msg if is_group else event.bot.send_private_msg
    sender.side_effect = [
        {"message_id": 42},
        {"message_id": 43},
    ]

    async def stream():
        yield MessageChain([Plain("First。")])
        yield MessageChain([Plain("Second。")])

    with (
        patch.object(AstrMessageEvent, "send", new=AsyncMock()),
        patch.object(AstrMessageEvent, "send_streaming", new=AsyncMock()),
        patch(
            "astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        await event.send_streaming(stream(), use_fallback=fallback)

    receipts = [call.args[0] for call in observer.await_args_list]
    assert [receipt["message_id"] for receipt in receipts] == (
        ["42", "43"] if fallback else ["42"]
    )
    assert (
        "".join(
            part["data"]["text"] for receipt in receipts for part in receipt["message"]
        )
        == "First。Second。"
    )
