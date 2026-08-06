#!/usr/bin/env python3
"""
An agentic research loop: local Maple + web search.

Maple runs locally and knows nothing after its training cut-off. This script
gives it a `web_search` tool (served by mcp/search_server.py, backed by Grok or
MiniMax) and runs the standard tool-calling loop:

    ask -> model requests a search -> run it -> feed results back -> repeat
         -> model answers with sources

This exercises the same tool-calling path the web UI uses, but from the command
line, so it works without configuring anything in the browser.

Usage
-----
    # 1. start the search server
    python mcp/search_server.py --port 8181 --backend grok

    # 2. start llama-server
    ./scripts/04-run.ps1 -Mode server

    # 3. ask something it cannot know from training data
    python mcp/research.py "What did deepgrove release in 2026 and how fast is it?"
"""

import argparse
import json
import sys
import urllib.request

DEFAULT_LLM = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MCP = "http://127.0.0.1:8181/mcp"

SYSTEM = (
    "You are a research assistant. You have a web_search tool. "
    "Use it whenever the question involves recent events, specific facts, "
    "or anything you are not certain about -- do not guess. "
    "After searching, answer concisely and cite the source URLs you used."
)


def rpc(url: str, method: str, params=None, rpc_id: int = 1, timeout: int = 300):
    body = json.dumps({"jsonrpc": "2.0", "id": rpc_id,
                       "method": method, "params": params or {}}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def mcp_tools(mcp_url: str):
    """Fetch the MCP tool list and convert it to OpenAI tool schema."""
    rpc(mcp_url, "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {},
         "clientInfo": {"name": "research.py", "version": "1.0"}})
    listed = rpc(mcp_url, "tools/list", rpc_id=2)["result"]["tools"]
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": t["inputSchema"]}} for t in listed]


def mcp_call(mcp_url: str, name: str, arguments: dict, timeout: int) -> str:
    res = rpc(mcp_url, "tools/call", {"name": name, "arguments": arguments},
              rpc_id=3, timeout=timeout)
    if "error" in res:
        return f"[tool error: {res['error'].get('message')}]"
    parts = res.get("result", {}).get("content", [])
    return "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")


def chat(llm_url: str, messages, tools, timeout: int):
    body = json.dumps({"messages": messages, "tools": tools,
                       "tool_choice": "auto", "max_tokens": 2048}).encode()
    req = urllib.request.Request(llm_url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["choices"][0]["message"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--llm", default=DEFAULT_LLM)
    ap.add_argument("--mcp", default=DEFAULT_MCP)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    question = " ".join(args.question)

    try:
        tools = mcp_tools(args.mcp)
    except Exception as exc:                                    # noqa: BLE001
        sys.exit(f"cannot reach MCP server at {args.mcp}: {exc}\n"
                 f"start it with: python mcp/search_server.py")

    print(f"tools available: {[t['function']['name'] for t in tools]}\n")

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]

    for round_no in range(1, args.max_rounds + 1):
        msg = chat(args.llm, messages, tools, args.timeout)
        calls = msg.get("tool_calls") or []

        if not calls:
            print(f"--- answer (after {round_no - 1} search round(s)) ---\n")
            print(msg.get("content", "").strip())
            return

        # Echo the assistant turn back verbatim; the tool results must follow it.
        messages.append(msg)

        for call in calls:
            fn = call["function"]
            try:
                fn_args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                fn_args = {}
            print(f"[round {round_no}] {fn['name']}({fn_args}) ...", flush=True)

            result = mcp_call(args.mcp, fn["name"], fn_args, args.timeout)
            print(f"    -> {len(result)} chars returned\n", flush=True)

            messages.append({"role": "tool",
                             "tool_call_id": call.get("id", ""),
                             "content": result})

    print("[hit max rounds without a final answer]")


if __name__ == "__main__":
    main()
