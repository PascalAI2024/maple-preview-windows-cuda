#!/usr/bin/env python3
"""
A minimal MCP server that gives a local model web search.

Why this exists
---------------
llama-server's web UI can call MCP servers, which is how you give a local model
tools. But the UI runs in a browser, so it can only reach MCP servers over HTTP --
the usual stdio transport is not an option. This is a small streamable-HTTP MCP
server exposing a single `web_search` tool.

The search itself is delegated to a backend that already has a real search index:

  grok     (default)  xAI's Grok CLI -- web + native X/Twitter search
  minimax             MiniMax chat API with web search enabled

Cost: the Grok CLI reports a `total_cost_usd` per call (~$0.09 in testing), but
that is an API-equivalent figure printed regardless of billing mode. Signed in
via OAuth (`auth_mode: oidc`, a subscription seat) it draws against plan quota
and rate limits, not that dollar amount; only an API-key setup bills per call.
MiniMax uses MINIMAX_API_KEY and is metered per request.

Usage
-----
    python mcp/search_server.py --port 8181 --backend grok

Then start llama-server with the MCP CORS proxy enabled:

    ./scripts/04-run.ps1 -Mode server -McpProxy

and add http://127.0.0.1:8181/mcp in the UI under "MCP Servers".

Security note: this binds to localhost only. `--ui-mcp-proxy` on llama-server is
explicitly documented as unsafe in untrusted environments -- do not expose either
port to a network you do not control.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_FALLBACK = "2025-06-18"

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use this for anything that "
            "happened recently, for facts you are unsure about, or when the user "
            "asks for sources. Returns a text summary with source URLs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, phrased as a question or keywords.",
                }
            },
            "required": ["query"],
        },
    }
]


# --------------------------------------------------------------------------
# Search backends
# --------------------------------------------------------------------------

def search_grok(query: str, timeout: int) -> str:
    """Delegate to the Grok CLI, which has web and native X/Twitter search."""
    grok = os.path.expanduser("~/.grok/bin/grok")
    if not os.path.exists(grok):
        grok = "grok"

    prompt = (
        f"Search the web and answer concisely: {query}\n\n"
        "Reply with a short factual summary followed by the source URLs you used. "
        "Do not ask follow-up questions."
    )
    proc = subprocess.run(
        [grok, "-p", prompt, "--always-approve", "--output-format", "json"],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        return f"[search failed: grok exited {proc.returncode}] {proc.stderr[-400:]}"

    # The CLI emits a JSON envelope; the prose we want is under "result". That
    # field is sometimes a plain string and sometimes a content object, so handle
    # both rather than passing a raw JSON blob to the model.
    try:
        payload = json.loads(proc.stdout)
        result = payload.get("result")
        if isinstance(result, dict):
            result = result.get("text") or json.dumps(result)
        elif isinstance(result, list):
            result = "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in result
            )
        return result or proc.stdout[-2000:]
    except json.JSONDecodeError:
        # Fall back to the last JSON object embedded in mixed output.
        match = re.findall(r'"result"\s*:\s*"(.*?)"\s*,\s*"stopReason"', proc.stdout, re.S)
        if match:
            return match[-1].encode().decode("unicode_escape")
        return proc.stdout[-2000:]


def search_minimax(query: str, timeout: int) -> str:
    """MiniMax chat completions with web search enabled."""
    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        return "[search failed: MINIMAX_API_KEY is not set]"

    body = json.dumps({
        "model": "MiniMax-Text-01",
        "messages": [{"role": "user", "content":
                      f"Search the web and answer concisely, with source URLs: {query}"}],
        "tools": [{"type": "web_search"}],
    }).encode()

    req = urllib.request.Request(
        "https://api.minimax.io/v1/text/chatcompletion_v2",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]
    except Exception as exc:                                   # noqa: BLE001
        return f"[search failed: {type(exc).__name__}: {exc}]"


BACKENDS = {"grok": search_grok, "minimax": search_minimax}


# --------------------------------------------------------------------------
# MCP JSON-RPC handling
# --------------------------------------------------------------------------

class MCPHandler(BaseHTTPRequestHandler):
    backend = "grok"
    search_timeout = 180

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Expose-Headers", "Mcp-Session-Id")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        # Streamable HTTP allows a GET for the server->client stream. We are
        # request/response only, so decline it rather than hold the socket open.
        self.send_response(405)
        self._cors()
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            msg = json.loads(self.rfile.read(length) or "{}")
        except json.JSONDecodeError:
            self._send({"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": "parse error"}})
            return

        # Notifications have no id and must not get a response body.
        if "id" not in msg:
            self.send_response(202)
            self._cors()
            self.end_headers()
            return

        self._send(self._dispatch(msg))

    def _dispatch(self, msg):
        rpc_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        def ok(result):
            return {"jsonrpc": "2.0", "id": rpc_id, "result": result}

        if method == "initialize":
            # Echo the client's protocol version when it sends one, so we do not
            # fail a handshake purely over version negotiation.
            version = params.get("protocolVersion") or PROTOCOL_FALLBACK
            return ok({
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "maple-search", "version": "1.0.0"},
            })

        if method == "ping":
            return ok({})

        if method == "tools/list":
            return ok({"tools": TOOLS})

        if method == "tools/call":
            name = params.get("name")
            if name != "web_search":
                return {"jsonrpc": "2.0", "id": rpc_id,
                        "error": {"code": -32602, "message": f"unknown tool: {name}"}}

            query = (params.get("arguments") or {}).get("query", "").strip()
            if not query:
                return ok({"content": [{"type": "text", "text": "No query given."}],
                           "isError": True})

            sys.stderr.write(f"  search[{self.backend}]: {query}\n")
            try:
                text = BACKENDS[self.backend](query, self.search_timeout)
            except subprocess.TimeoutExpired:
                text = f"[search timed out after {self.search_timeout}s]"
            except Exception as exc:                            # noqa: BLE001
                text = f"[search failed: {type(exc).__name__}: {exc}]"

            return ok({"content": [{"type": "text", "text": text}]})

        return {"jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32601, "message": f"method not found: {method}"}}

    def _send(self, payload):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self._cors()
        self.end_headers()
        self.wfile.write(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="grok")
    ap.add_argument("--timeout", type=int, default=180,
                    help="seconds to allow per search (grok can take ~30-60s)")
    args = ap.parse_args()

    MCPHandler.backend = args.backend
    MCPHandler.search_timeout = args.timeout

    server = ThreadingHTTPServer((args.host, args.port), MCPHandler)
    print(f"maple-search MCP server on http://{args.host}:{args.port}/mcp "
          f"(backend: {args.backend})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
