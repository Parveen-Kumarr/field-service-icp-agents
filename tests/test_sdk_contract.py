"""Contract test: the tool loop against the REAL google-genai SDK (Interactions API).

The SDK talks to an in-memory HTTP server that returns responses in the Gemini
Interactions API's JSON format (google_search_call/result steps, model_output
with url_citation annotations, function_call steps, usage). This checks that
the SDK parses them into what the loop reads, and that the follow-up requests
the loop sends are valid API payloads.
"""
import asyncio
import json
import sys
from pathlib import Path

import httpx
from google import genai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from icp_agents.llm import LLM, ClientTool, search_tool, url_tool  # noqa: E402

SUBMIT = ClientTool("submit_x", "submit", {"type": "object", "properties": {"answer": {"type": "string"}},
                                           "required": ["answer"]})


def sdk_client(responses: list[dict], sent: list):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append((request.url.path, body))
        r = responses[len(sent) - 1]
        return httpx.Response(200, json={"model": body["model"], **r})

    return genai.Client(api_key="test-key", http_options={
        "httpx_async_client": httpx.AsyncClient(transport=httpx.MockTransport(handler))})


def test_search_function_call_and_stateful_follow_up():
    responses = [
        {"id": "int_1", "status": "requires_action", "steps": [
            {"type": "thought", "signature": "c2lnLTE="},
            {"type": "google_search_call", "id": "gs_1", "arguments": {"queries": ["Stryker ProCare field service"]}},
            {"type": "google_search_result", "call_id": "gs_1", "result": [{"search_suggestions": "<div/>"}]},
            {"type": "url_context_call", "id": "uc_1", "arguments": {"urls": ["https://www.stryker.com/procare"]}},
            {"type": "url_context_result", "call_id": "uc_1",
             "result": [{"url": "https://www.stryker.com/procare", "status": "success"}]},
            {"type": "model_output", "content": [{"type": "text", "text": "ProCare services hospital equipment.",
                                                   "annotations": [{"type": "url_citation", "url": "https://www.stryker.com/about",
                                                                    "title": "stryker.com", "start_index": 0, "end_index": 7}]}]},
            {"type": "function_call", "id": "fc_1", "name": "ask_researcher", "arguments": {"question": "How many FSEs?"}}],
         "usage": {"total_input_tokens": 500, "total_output_tokens": 40, "total_thought_tokens": 30,
                   "total_tool_use_tokens": 1200, "grounding_tool_count": [{"type": "google_search", "count": 1}]}},
        {"id": "int_2", "status": "requires_action", "steps": [
            {"type": "function_call", "id": "fc_2", "name": "submit_x", "arguments": {"answer": "yes"}}],
         "usage": {"total_input_tokens": 60, "total_output_tokens": 12}},
    ]
    sent: list = []
    asked = []

    async def ask(args):
        asked.append(args["question"])
        return {"answer": "About 1,000 field service engineers."}

    llm = LLM(model="gemini-3.8-flash", client=sdk_client(responses, sent))
    res = asyncio.run(llm.run(
        agent="Analyst", system="sys", prompt="Research Stryker", server_tools=[search_tool(), url_tool()],
        tools=[ClientTool("ask_researcher", "ask", {"type": "object", "properties": {"question": {"type": "string"}}}, ask)],
        submit=SUBMIT))

    assert res.output == {"answer": "yes"} and asked == ["How many FSEs?"]
    assert [s["url"] for s in res.sources] == ["https://www.stryker.com/procare", "https://www.stryker.com/about"]
    assert res.searches == 1 and res.fetches == 1
    u = llm.usage["Analyst"]
    assert (u.calls, u.web_searches, u.url_fetches, u.thought_tokens) == (2, 1, 1, 30)
    assert u.input_tokens == 500 + 1200 + 60  # tool-use (grounding) tokens count as input

    path, first = sent[0]
    assert path.endswith("/interactions")
    assert first["model"] == "gemini-3.8-flash" and first["system_instruction"] == "sys"
    # the SDK wraps the prompt parts into one user_input step
    assert first["input"] == [{"type": "user_input", "content": [{"type": "text", "text": "Research Stryker"}]}]
    assert [t["type"] for t in first["tools"]] == ["google_search", "url_context", "function", "function"]
    assert first["tools"][3]["name"] == "submit_x" and first["tools"][3]["parameters"]["required"] == ["answer"]
    assert "tool_choice" not in first["generation_config"]

    _, second = sent[1]  # stateful follow-up: previous id + only the new function result
    assert second["previous_interaction_id"] == "int_1"
    [fr] = second["input"]
    assert fr["type"] == "function_result" and fr["call_id"] == "fc_1" and fr["name"] == "ask_researcher"
    assert "1,000" in fr["result"]


def test_prose_answer_is_nudged_into_forced_submit():
    responses = [
        {"id": "int_1", "status": "completed", "steps": [
            {"type": "model_output", "content": [{"type": "text", "text": "Here is my answer in prose."}]}]},
        {"id": "int_2", "status": "requires_action", "steps": [
            {"type": "function_call", "id": "fc_9", "name": "submit_x", "arguments": {"answer": "ok"}}]},
    ]
    sent: list = []
    llm = LLM(model="gemini-3.8-flash", client=sdk_client(responses, sent))
    res = asyncio.run(llm.run(agent="Coordinator", system="s", prompt="p", submit=SUBMIT))
    assert res.output == {"answer": "ok"}
    _, second = sent[1]
    assert second["previous_interaction_id"] == "int_1"
    assert second["generation_config"]["tool_choice"] == {"allowed_tools": {"mode": "any", "tools": ["submit_x"]}}


def test_image_parts_and_failed_interaction():
    responses = [{"id": "int_1", "status": "failed", "steps": [], "errors": [{"message": "quota exhausted"}]}]
    sent: list = []
    llm = LLM(model="gemini-3.8-flash", client=sdk_client(responses, sent))
    prompt = [{"type": "text", "text": "Logo 0:"}, {"type": "image", "data": "iVBORw0KGgo=", "mime_type": "image/png"}]
    try:
        asyncio.run(llm.run(agent="Scout", system="s", prompt=prompt, submit=SUBMIT, force_submit=True))
        raise AssertionError("expected LLMError")
    except Exception as exc:
        assert "failed" in str(exc) and "quota exhausted" in str(exc)
    _, first = sent[0]
    assert first["input"][0]["content"][1] == {"type": "image", "data": "iVBORw0KGgo=", "mime_type": "image/png"}
    assert first["generation_config"]["tool_choice"]["allowed_tools"]["tools"] == ["submit_x"]
