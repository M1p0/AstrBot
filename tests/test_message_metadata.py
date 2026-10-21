from types import SimpleNamespace

import pytest

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import (
    CheckpointData,
    Message,
    TextPart,
    bind_checkpoint_messages,
    dump_messages_with_checkpoints,
)
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.provider.provider import Provider


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_message_metadata_is_internal_to_history(role):
    message = Message(
        role=role,
        content=[
            TextPart(text="original message"),
            TextPart(text="temporary evidence").mark_as_temp(),
        ],
        metadata={"source": {"message_id": "123", "sender_id": "456"}},
    )
    message._checkpoint_after = CheckpointData(id="checkpoint")

    assert "metadata" not in message.model_dump()
    assert "message_id" not in message.model_dump_json()
    saved = dump_messages_with_checkpoints([message])
    assert saved == [
        {
            "role": role,
            "content": [{"type": "text", "text": "original message"}],
            "metadata": {"source": {"message_id": "123", "sender_id": "456"}},
        },
        {"role": "_checkpoint", "content": {"id": "checkpoint"}},
    ]
    restored = bind_checkpoint_messages(saved)
    assert restored[0].metadata == message.metadata
    assert dump_messages_with_checkpoints(restored) == saved

    restored[0].metadata["source"]["message_id"] = "changed"
    restored[0].content[0].text = "changed"
    assert saved[0]["metadata"]["source"]["message_id"] == "123"
    assert saved[0]["content"][0]["text"] == "original message"
    message.metadata["source"]["sender_id"] = "changed"
    assert saved[0]["metadata"]["source"]["sender_id"] == "456"


def test_provider_excludes_metadata_from_models_and_raw_history_without_mutation():
    raw = {
        "role": "user",
        "content": [{"type": "text", "text": "original message"}],
        "metadata": {"source": {"message_id": "123"}},
    }
    assistant = Message(
        role="assistant",
        content="reply",
        metadata={"source": {"message_id": "456"}},
    )

    provider_messages = Provider._ensure_message_to_dicts(object(), [raw, assistant])

    assert provider_messages == [
        {"role": "user", "content": [{"type": "text", "text": "original message"}]},
        {"role": "assistant", "content": "reply"},
    ]
    provider_messages[0]["content"][0]["text"] = "changed"
    assert raw["content"][0]["text"] == "original message"
    assert raw["metadata"]["source"]["message_id"] == "123"
    assert assistant.metadata["source"]["message_id"] == "456"


@pytest.mark.asyncio
@pytest.mark.parametrize("modalities", [None, ["text"], ["image", "audio", "tool_use"]])
async def test_runner_keeps_current_and_prior_metadata_out_of_provider_content(
    modalities,
):
    prior = {
        "role": "assistant",
        "content": "previous reply",
        "metadata": {"source": {"message_id": "100"}},
    }
    request = ProviderRequest(
        prompt="current message",
        contexts=[prior],
        image_urls=["unused-image"] if modalities == ["text"] else [],
        extra_user_metadata={"source": {"message_id": "101"}},
        extra_user_content_parts=[TextPart(text="evidence").mark_as_temp()],
    )
    runner = ToolLoopAgentRunner()
    context = ContextWrapper(context=None)
    await runner.reset(
        provider=SimpleNamespace(provider_config={"modalities": modalities}),
        request=request,
        run_context=context,
        tool_executor=None,
        agent_hooks=BaseAgentRunHooks(),
    )

    saved = dump_messages_with_checkpoints(context.messages)
    assert saved[0]["metadata"] == {"source": {"message_id": "100"}}
    assert saved[1]["metadata"] == {"source": {"message_id": "101"}}
    assert "evidence" not in str(saved)
    if modalities == ["text"]:
        assert {"type": "text", "text": "[Image]"} in saved[1]["content"]
    assembled = await request.assemble_context() if not request.image_urls else None
    if assembled:
        assert "metadata" not in assembled

    provider_contexts = Provider._ensure_message_to_dicts(
        object(), runner._sanitize_contexts_for_provider(context.messages)
    )
    assert "metadata" not in str(provider_contexts)
    assert "evidence" in str(provider_contexts)
    context.messages[0].metadata["source"]["message_id"] = "changed"
    request.extra_user_metadata["source"]["message_id"] = "changed"
    assert prior["metadata"]["source"]["message_id"] == "100"
    assert context.messages[1].metadata["source"]["message_id"] == "101"
