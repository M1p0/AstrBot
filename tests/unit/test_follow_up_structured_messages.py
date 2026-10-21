import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from astrbot.core.agent.runners.tool_loop_agent_runner import FollowUpTicket
from astrbot.core.message.components import At, File, Image, Plain, Record, Reply
from astrbot.core.pipeline.process_stage import follow_up


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["FriendMessage", "GroupMessage"])
@pytest.mark.parametrize(
    "component",
    [
        Image(file="https://example.com/image.png"),
        Record(file="https://example.com/voice.wav"),
        File(name="document.txt", url="https://example.com/document.txt"),
        Reply(id="123", sender_id="999", message_str="Quoted claim"),
        At(qq="999"),
    ],
    ids=["image", "voice", "file", "quote", "mention"],
)
async def test_structured_follow_up_keeps_components_for_normal_request(
    monkeypatch, chat_type, component
):
    umo = f"qq:{chat_type}:456"
    event = SimpleNamespace(
        unified_msg_origin=umo,
        message_obj=SimpleNamespace(message=[Plain("Please inspect this"), component]),
        get_sender_id=lambda: "111",
        get_message_str=lambda: "Please inspect this",
        get_message_outline=lambda: "Please inspect this [structured content]",
        get_extra=lambda *args: None,
    )
    origin = SimpleNamespace(get_sender_id=lambda: "111", get_extra=lambda *args: None)
    runner = SimpleNamespace(
        run_context=SimpleNamespace(context=SimpleNamespace(event=origin)),
        follow_up=Mock(),
    )
    monkeypatch.setattr(follow_up, "_ACTIVE_AGENT_RUNNERS", {umo: runner})
    monkeypatch.setattr(follow_up, "_FOLLOW_UP_ORDER_STATE", {})

    assert follow_up.try_capture_follow_up(event) is None

    runner.follow_up.assert_not_called()
    assert event.message_obj.message == [Plain("Please inspect this"), component]
    assert follow_up._FOLLOW_UP_ORDER_STATE == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["FriendMessage", "GroupMessage"])
async def test_plain_follow_up_from_same_sender_remains_capturable(
    monkeypatch, chat_type
):
    umo = f"qq:{chat_type}:456"
    ticket = FollowUpTicket(seq=0, text="One more detail")
    event = SimpleNamespace(
        unified_msg_origin=umo,
        message_obj=SimpleNamespace(
            message=[Plain("One more detail")], message_id="123"
        ),
        get_sender_id=lambda: "111",
        get_message_str=lambda: "One more detail",
        get_message_outline=lambda: "One more detail",
        get_extra=lambda *args: None,
    )
    origin = SimpleNamespace(
        message_obj=SimpleNamespace(message_id="122"),
        get_sender_id=lambda: "111",
        get_extra=lambda *args: None,
    )
    runner = SimpleNamespace(
        run_context=SimpleNamespace(context=SimpleNamespace(event=origin)),
        follow_up=Mock(return_value=ticket),
    )
    monkeypatch.setattr(follow_up, "_ACTIVE_AGENT_RUNNERS", {umo: runner})
    monkeypatch.setattr(follow_up, "_FOLLOW_UP_ORDER_STATE", {})

    capture = follow_up.try_capture_follow_up(event)

    assert capture is not None
    runner.follow_up.assert_called_once_with(message_text="One more detail")
    capture.monitor_task.cancel()
    await asyncio.gather(capture.monitor_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_group_follow_up_from_other_sender_stays_a_separate_turn(monkeypatch):
    umo = "qq:GroupMessage:456"
    event = SimpleNamespace(
        unified_msg_origin=umo,
        message_obj=SimpleNamespace(message=[Plain("Another participant")]),
        get_sender_id=lambda: "222",
    )
    runner = SimpleNamespace(
        run_context=SimpleNamespace(
            context=SimpleNamespace(event=SimpleNamespace(get_sender_id=lambda: "111"))
        ),
        follow_up=Mock(),
    )
    monkeypatch.setattr(follow_up, "_ACTIVE_AGENT_RUNNERS", {umo: runner})

    assert follow_up.try_capture_follow_up(event) is None
    runner.follow_up.assert_not_called()
