# codex-proxy

[![CI](https://github.com/cainrus/codex-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/cainrus/codex-proxy/actions/workflows/ci.yml)

A single-file bridge from the **Anthropic Messages API** to the **OpenAI
Responses API** — run Claude Code (or any Anthropic-API client) on
`gpt-5.6-*` models while keeping the client's own tools, skills and MCP
servers.

One file, zero dependencies, Node ≥ 22, no build step:

```sh
OPENAI_API_KEY=sk-... PORT=4001 ./codex-proxy.ts
# then point the client at it:
ANTHROPIC_BASE_URL=http://127.0.0.1:4001 claude --model codex-sol
```

## Why the Responses API, not chat/completions

`gpt-5.6-*` rejects function tools on `/v1/chat/completions` when
`reasoning_effort` is set — the API answers *"Function tools with
reasoning_effort are not supported … use /v1/responses or set
reasoning_effort to 'none'"*. Killing reasoning defeats the point of the
model, so the bridge speaks Responses upstream and Messages downstream.

## What it translates

| Anthropic (downstream) | OpenAI Responses (upstream) |
|---|---|
| `messages[]` with text / image / `tool_result` blocks | `input[]` items (`input_text`, `input_image`, `function_call_output`) |
| `tool_use` blocks | `function_call` items |
| `tools` + `tool_choice` | `tools` (non-strict) + `tool_choice` |
| `system` | `instructions` |
| SSE streaming events | rebuilt event-by-event (`message_start` → `content_block_*` → `message_stop`) |
| `/v1/messages/count_tokens` | local estimate (client uses it only for context-pressure hints) |

## Carrying reasoning across tool rounds

Responses-API reasoning items are opaque: the model returns an encrypted
blob that must be echoed back verbatim on the next turn, or the chain of
thought is lost. An Anthropic-API client only round-trips content blocks it
received — so the bridge smuggles the blob through a `thinking` block's
`signature` field and rebuilds the `reasoning` item on the way back
upstream. Disable with `CODEX_CARRY_REASONING=0`.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — (required) | upstream auth |
| `PORT` | `4001` | `0` picks a free port; prints `codex-proxy ready port=<n>` on stderr |
| `CODEX_UPSTREAM` | `https://api.openai.com/v1/responses` | upstream URL |
| `CODEX_EFFORT` | `medium` | `reasoning.effort` for `gpt-5.6-*` |
| `CODEX_CARRY_REASONING` | on | `0` disables the encrypted-reasoning round-trip |
| `CODEX_FALLBACK_MODEL` | `gpt-5.4-mini` | where background calls (titles, quota probes) land instead of 400-ing |
| `CODEX_DEBUG` | off | `1` logs request/response shapes to stderr |

Model aliases: `codex-luna` → `gpt-5.6-luna`, `codex-sol` → `gpt-5.6-sol`,
`codex-terra` → `gpt-5.6-terra`; any `gpt-*` name passes through unchanged.

## Tests

```sh
npm test
```

The translation layer is exported from `codex-proxy.ts`, and the server only
starts when the file is the process entry point — so `node --test` can pin
every conversion without opening a socket or calling the upstream API. The
streaming tests replay a recorded Responses SSE stream from
`test/fixtures/`, re-fed at several chunk sizes to prove events that span a
socket boundary are reassembled.

## Non-goals

Auth beyond a bearer key, retries, rate limiting, multi-tenant use. It is a
localhost bridge (`127.0.0.1` only) meant to sit next to the client process
that spawns it.

## License

MIT
