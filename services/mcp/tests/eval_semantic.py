"""I-CARE MCP semantic eval script.

Connects to the running MCP server (HTTP, localhost:8002), queries tool schemas,
then uses a local LLM (qwen3.5:4b via Ollama) to answer golden Q&A pairs and
evaluates correctness programmatically.

Pre-conditions:
  - platform-mcp container running:  docker compose up mcp -d  (in forgety/)
  - Ollama running with qwen3.5:4b

Run:
  cd zoo/unlearning
  C:/Users/.../sd-interpretability/.venv/Scripts/python.exe \
      forgety/services/mcp/tests/eval_semantic.py
"""
import json
import sys
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# Local import — run from zoo/unlearning/ as cwd
sys.path.insert(0, ".")
from forgety.services.mcp.tests.golden_qa import GOLDEN_QA

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MCP_BASE_URL = "http://localhost:8002/mcp"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen3.5:4b"
MAX_TOOL_ROUNDS = 6   # safety cap on the tool-calling loop


# ---------------------------------------------------------------------------
# MCP client — thin HTTP wrapper (no session state needed for stateless tools)
# ---------------------------------------------------------------------------

def _mcp_init() -> str:
    """Initialize MCP session and return session ID."""
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "eval_semantic", "version": "1.0"},
        },
    }
    resp = httpx.post(
        MCP_BASE_URL,
        json=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        timeout=15,
    )
    session_id: str = resp.headers["mcp-session-id"]
    return session_id


def _mcp_call(session_id: str, method: str, params: Dict[str, Any]) -> Any:
    """Call an MCP method and return the result."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    resp = httpx.post(
        MCP_BASE_URL,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Mcp-Session-Id": session_id,
        },
        timeout=120,  # some RTs download from HF
    )
    body = resp.text.strip()
    # SSE format: "event: message\ndata: {...}"  or just  "data: {...}"
    for line in body.splitlines():
        if line.startswith("data:"):
            body = line[5:].strip()
            break
    return json.loads(body)


def fetch_mcp_tools(session_id: str) -> List[Dict[str, Any]]:
    """Fetch tool list from MCP server and return as OpenAI tool schemas."""
    result = _mcp_call(session_id, "tools/list", {})
    mcp_tools = result["result"]["tools"]

    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
            },
        })
    return openai_tools


def call_mcp_tool(session_id: str, name: str, arguments: str) -> str:
    """Execute a MCP tool and return the result as a string."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid JSON arguments: {arguments}"})

    result = _mcp_call(session_id, "tools/call", {"name": name, "arguments": args})
    if "error" in result:
        return json.dumps({"error": result["error"]})

    content = result.get("result", {}).get("content", [])
    if not content:
        return json.dumps({"error": "empty response from tool"})

    # Return the first text content block, truncated if very large.
    # 4B models struggle with >8000 chars of tool output; truncate with a note.
    _MAX_TOOL_CHARS = 8000
    for block in content:
        if block.get("type") == "text":
            text = block["text"]
            if len(text) > _MAX_TOOL_CHARS:
                text = (
                    text[:_MAX_TOOL_CHARS]
                    + f"\n\n[... TRUNCATED: original response was {len(block['text'])} chars ...]"
                )
            return text
    return json.dumps({"error": "no text content in tool response"})


# ---------------------------------------------------------------------------
# LLM agent loop
# ---------------------------------------------------------------------------

def run_agent(
    client: OpenAI,
    session_id: str,
    tools: List[Dict[str, Any]],
    question: str,
) -> tuple[str, List[str]]:
    """
    Run the tool-calling loop until the LLM produces a final answer.

    Returns (final_answer, tools_called).
    """
    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are an assistant that helps users understand machine unlearning "
                "benchmark results from the I-CARE benchmark. "
                "Always use the available tools to fetch real data before answering. "
                "Be concise but accurate."
            ),
        },
        {"role": "user", "content": question},
    ]

    tools_called: List[str] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=tools,  # type: ignore[arg-type]
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            # Append assistant message with tool_calls
            messages.append(choice.message)  # type: ignore[arg-type]

            # Execute each tool call
            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                fn_args = tc.function.arguments
                tools_called.append(fn_name)
                tool_result = call_mcp_tool(session_id, fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
        else:
            # Final answer
            final = choice.message.content or ""
            return final.strip(), tools_called

    return "[max tool rounds exceeded]", tools_called


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval() -> None:
    print("=" * 70)
    print("I-CARE MCP Semantic Eval")
    print(f"Model: {OLLAMA_MODEL}  |  MCP: {MCP_BASE_URL}")
    print("=" * 70)

    # Init
    print("\n[init] Connecting to MCP server...")
    session_id = _mcp_init()
    print(f"[init] Session: {session_id[:16]}...")

    print("[init] Fetching tool schemas...")
    tools = fetch_mcp_tools(session_id)
    print(f"[init] {len(tools)} tools: {[t['function']['name'] for t in tools]}")

    # Set generous timeouts — local 4B model with large tool results can take 3-5 min.
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=300.0)

    passed = 0
    results = []

    for qa_id, question, expected_tools, check_fn in GOLDEN_QA:
        print(f"\n{'-' * 70}")
        print(f"[{qa_id}] {question[:80]}{'...' if len(question) > 80 else ''}")
        print(f"  Expected tools: {expected_tools}")

        answer, called = run_agent(client, session_id, tools, question)
        ok, reason = check_fn(answer)

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1

        print(f"  Tools called:   {called}")
        print(f"  Result:         {status} — {reason}")
        print(f"  Answer:\n{textwrap.indent(answer[:500], '    ')}")
        if len(answer) > 500:
            print("    [... truncated]")

        results.append({
            "id": qa_id,
            "passed": ok,
            "reason": reason,
            "tools_called": called,
            "answer": answer,
        })

    print(f"\n{'=' * 70}")
    print(f"RESULT: {passed}/{len(GOLDEN_QA)} passed")
    print("=" * 70)

    # Save results for appending to plan file
    out_path = "forgety/services/mcp/tests/eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {out_path}")

    if passed < len(GOLDEN_QA):
        sys.exit(1)


if __name__ == "__main__":
    run_eval()
