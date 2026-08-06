#!/usr/bin/env python3
"""
Give the llama-server web UI web search without configuring anything in the browser.

Why
---
llama-server stores MCP server config in the browser's IndexedDB, reachable only
through a modal in the web UI. There is no server-side flag for it, so the setup
cannot be scripted, version-controlled, or reproduced on another machine.

This proxy removes that requirement. It sits in front of llama-server and:

  * passes every request through untouched (the UI, /props, /health, ...)
  * except POST /v1/chat/completions, where it injects a `web_search` tool
    definition and runs the tool-calling loop server-side against
    mcp/search_server.py

The browser sees an ordinary chat endpoint. The model gets search.

Layout
------
    browser :8080  ->  this proxy  ->  llama-server :8081
                            |
                            +------->  search_server.py :8181  ->  Grok / MiniMax

Usage
-----
    python mcp/search_server.py --port 8181 --backend grok
    ./scripts/04-run.ps1 -Mode server -Port 8081
    python mcp/search_proxy.py --listen 8080 --upstream 8081 --mcp 8181

Then open http://127.0.0.1:8080 as usual and just ask a question that needs
current information.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Maple does not always emit a structured `tool_calls` field. It sometimes writes
# the Hermes-style <tool_call>{...}</tool_call> block as plain text -- and when it
# does so inside the reasoning channel, llama-server's parser does not lift it out,
# so the request silently comes back with no tool call and no answer. Recovering it
# here is what makes tool use reliable rather than intermittent.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

CHAT_PATH = "/v1/chat/completions"

SEARCH_HINT = (
    "You have a web_search tool. Use it whenever the question involves recent "
    "events, specific facts, or anything you are not certain about -- do not "
    "guess. After searching, answer concisely and cite the source URLs."
)


def mcp_rpc(mcp_url, method, params=None, rpc_id=1, timeout=300):
    body = json.dumps({"jsonrpc": "2.0", "id": rpc_id,
                       "method": method, "params": params or {}}).encode()
    req = urllib.request.Request(mcp_url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def salvage_tool_calls(msg):
    """Extract <tool_call> blocks the model wrote as text into OpenAI tool_calls."""
    blob = f"{msg.get('content') or ''}\n{msg.get('reasoning_content') or ''}"
    found = []
    for i, raw in enumerate(TOOL_CALL_RE.findall(blob)):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = parsed.get("name")
        if not name:
            continue
        found.append({
            "id": f"salvaged_{i}",
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(parsed.get("arguments") or {})},
        })
    return found


def fetch_tools(mcp_url):
    """MCP tool list -> OpenAI tool schema. Returns [] if the server is down."""
    try:
        mcp_rpc(mcp_url, "initialize",
                {"protocolVersion": "2025-06-18", "capabilities": {},
                 "clientInfo": {"name": "search_proxy", "version": "1.0"}}, timeout=15)
        listed = mcp_rpc(mcp_url, "tools/list", rpc_id=2, timeout=15)["result"]["tools"]
        return [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["inputSchema"]}} for t in listed]
    except Exception as exc:                                    # noqa: BLE001
        sys.stderr.write(f"  [warn] MCP unreachable, search disabled: {exc}\n")
        return []


class Proxy(BaseHTTPRequestHandler):
    upstream = "http://127.0.0.1:8081"
    mcp_url = "http://127.0.0.1:8181/mcp"
    max_rounds = 4
    timeout = 600

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # too chatty; we log the interesting events ourselves

    # -- plumbing ----------------------------------------------------------

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _passthrough(self, body=None):
        """Forward the request verbatim and mirror the response back."""
        url = self.upstream + self.path
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "accept-encoding")}
        req = urllib.request.Request(url, data=body, headers=headers,
                                     method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() in ("transfer-encoding", "connection", "content-length"):
                        continue
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as e:
            payload = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:                                # noqa: BLE001
            self._json({"error": {"message": f"upstream unreachable: {exc}"}}, 502)

    def _json(self, obj, status=200):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _upstream_chat(self, payload):
        req = urllib.request.Request(
            self.upstream + CHAT_PATH, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    # -- verbs -------------------------------------------------------------

    def do_GET(self):
        self._passthrough()

    def do_HEAD(self):
        self._passthrough()

    def do_DELETE(self):
        self._passthrough()

    def do_PUT(self):
        self._passthrough(self._read_body())

    def do_PATCH(self):
        self._passthrough(self._read_body())

    def do_OPTIONS(self):
        self._passthrough()

    def do_POST(self):
        body = self._read_body()
        if self.path.split("?")[0] != CHAT_PATH:
            return self._passthrough(body)

        try:
            payload = json.loads(body or "{}")
        except json.JSONDecodeError:
            return self._passthrough(body)

        tools = fetch_tools(self.mcp_url)
        if not tools:
            return self._passthrough(body)          # no search available; behave normally

        wants_stream = bool(payload.get("stream"))

        # Nudge the model to actually reach for the tool.
        messages = list(payload.get("messages") or [])
        if messages and messages[0].get("role") == "system":
            messages[0] = {**messages[0],
                           "content": f"{messages[0].get('content','')}\n\n{SEARCH_HINT}"}
        else:
            messages.insert(0, {"role": "system", "content": SEARCH_HINT})

        # Run the loop unstreamed; we re-emit as SSE at the end if asked.
        work = {**payload, "messages": messages, "stream": False,
                "tools": payload.get("tools") or tools,
                "tool_choice": payload.get("tool_choice", "auto")}

        final = None
        for rnd in range(1, self.max_rounds + 1):
            try:
                data = self._upstream_chat(work)
            except Exception as exc:                            # noqa: BLE001
                return self._json({"error": {"message": f"upstream error: {exc}"}}, 502)

            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []

            if not calls:
                calls = salvage_tool_calls(msg)
                if calls:
                    sys.stderr.write(f"  [salvaged {len(calls)} tool call(s) from text]\n")
                    # Strip the raw block so it is not replayed back as context.
                    msg = {**msg,
                           "content": TOOL_CALL_RE.sub("", msg.get("content") or "").strip(),
                           "tool_calls": calls}
                    msg.pop("reasoning_content", None)

            if not calls:
                final = data
                break

            work["messages"].append(msg)
            for call in calls:
                fn = call.get("function", {})
                try:
                    fn_args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    fn_args = {}
                sys.stderr.write(f"  [round {rnd}] {fn.get('name')}({fn_args})\n")
                try:
                    res = mcp_rpc(self.mcp_url, "tools/call",
                                  {"name": fn.get("name"), "arguments": fn_args},
                                  rpc_id=3, timeout=self.timeout)
                    parts = res.get("result", {}).get("content", [])
                    text = "\n".join(p.get("text", "") for p in parts
                                     if p.get("type") == "text")
                except Exception as exc:                        # noqa: BLE001
                    text = f"[search failed: {exc}]"
                sys.stderr.write(f"           -> {len(text)} chars\n")
                work["messages"].append({"role": "tool",
                                         "tool_call_id": call.get("id", ""),
                                         "content": text})

        if final is None:
            # Ran out of rounds while the model was still searching. Returning the
            # last response would hand back a tool-call message with empty content
            # (the user sees a blank reply), so force one answer with tools off.
            sys.stderr.write(f"  [max rounds] forcing final answer\n")
            closing = {**work, "tool_choice": "none"}
            closing.pop("tools", None)
            closing["messages"] = work["messages"] + [{
                "role": "system",
                "content": "Answer now using only the search results above. "
                           "Do not search again. Cite the source URLs.",
            }]
            try:
                final = self._upstream_chat(closing)
            except Exception as exc:                            # noqa: BLE001
                return self._json({"error": {"message": f"upstream error: {exc}"}}, 502)

        if not wants_stream:
            return self._json(final)

        self._emit_sse(final)

    def _emit_sse(self, final):
        """Re-emit a completed response as an SSE stream, which is what the UI expects."""
        choice = final["choices"][0]["message"]
        content = choice.get("content") or ""
        reasoning = choice.get("reasoning_content") or ""
        created = final.get("created") or int(time.time())
        model = final.get("model", "maple")
        cid = final.get("id", "chatcmpl-proxy")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def chunk(delta, finish=None):
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        try:
            chunk({"role": "assistant", "content": ""})
            if reasoning:
                chunk({"reasoning_content": reasoning})
            # Emit in slices so the UI renders progressively rather than in one jump.
            for i in range(0, len(content), 24):
                chunk({"content": content[i:i + 24]})
            chunk({}, finish="stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8080)
    ap.add_argument("--upstream", type=int, default=8081)
    ap.add_argument("--mcp", type=int, default=8181)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-rounds", type=int, default=4)
    args = ap.parse_args()

    Proxy.upstream = f"http://{args.host}:{args.upstream}"
    Proxy.mcp_url = f"http://{args.host}:{args.mcp}/mcp"
    Proxy.max_rounds = args.max_rounds

    srv = ThreadingHTTPServer((args.host, args.listen), Proxy)
    print(f"search proxy  http://{args.host}:{args.listen}"
          f"  ->  llama-server :{args.upstream}  +  mcp :{args.mcp}", flush=True)
    print("open the web UI on the proxy port; search is injected automatically.",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
