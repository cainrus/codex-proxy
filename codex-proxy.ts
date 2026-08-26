#!/usr/bin/env -S node --experimental-strip-types --disable-warning=ExperimentalWarning --disable-warning=MODULE_TYPELESS_PACKAGE_JSON
/**
 * codex-proxy — Anthropic Messages API -> OpenAI Responses API bridge.
 *
 * Lets a Claude Code process run on gpt-5.6-* while keeping CC's own tools,
 * skills and MCP servers. Runnable standalone; also embeddable under a
 * supervisor process that parses the ready line below.
 *
 * Why Responses and not chat/completions: gpt-5.6-* rejects function tools on
 * /v1/chat/completions -- "Function tools with reasoning_effort are not
 * supported ... use /v1/responses or set reasoning_effort to 'none'". Killing
 * reasoning defeats the point of the model.
 *
 * PORT=0 picks a free port and prints "codex-proxy ready port=<n>" on stderr;
 * codex-agent parses that line.
 */
import http from 'node:http'

const PORT = Number(process.env.PORT ?? 4001)
const OPENAI_KEY = process.env.OPENAI_API_KEY
const UPSTREAM = process.env.CODEX_UPSTREAM ?? 'https://api.openai.com/v1/responses'
const EFFORT = process.env.CODEX_EFFORT ?? 'medium'
/** Carry reasoning across tool rounds via encrypted_content (phase 2). */
const CARRY_REASONING = process.env.CODEX_CARRY_REASONING !== '0'

const MODEL_MAP: Record<string, string> = {
  'codex-luna': 'gpt-5.6-luna',
  'codex-sol': 'gpt-5.6-sol',
  'codex-terra': 'gpt-5.6-terra',
}
/**
 * CC fires background calls (conversation titles, quota probes) on haiku no
 * matter what --model says; route them somewhere cheap instead of 400-ing.
 */
const FALLBACK_MODEL = process.env.CODEX_FALLBACK_MODEL ?? 'gpt-5.4-mini'

type Json = any

const DEBUG = process.env.CODEX_DEBUG === '1'
const log = (...a: unknown[]) => console.error('[codex-proxy]', ...a)
const debug = (...a: unknown[]) => { if (DEBUG) log(...a) }

if (!OPENAI_KEY) {
  log('FATAL: OPENAI_API_KEY is not set')
  process.exit(1)
}

function mapModel(m: string): string {
  if (MODEL_MAP[m]) return MODEL_MAP[m]
  if (m?.startsWith('gpt-')) return m
  return FALLBACK_MODEL
}

function textOf(content: Json): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content.filter((b: Json) => b?.type === 'text').map((b: Json) => b.text).join('\n')
}

/**
 * Reasoning items are opaque: the model returns an encrypted blob that must be
 * echoed back verbatim on the next turn or the chain of thought is lost. CC
 * only round-trips what it received, so we smuggle the blob through a
 * thinking block's signature and rebuild the item on the way back.
 */
const REASONING_MARK = '​<codex-reasoning/>'

function encodeReasoning(item: Json): Json | null {
  if (!CARRY_REASONING || !item?.encrypted_content) return null
  return {
    type: 'thinking',
    thinking: REASONING_MARK,
    signature: JSON.stringify({ id: item.id, ec: item.encrypted_content }),
  }
}

function decodeReasoning(block: Json): Json | null {
  if (!CARRY_REASONING) return null
  if (block?.type !== 'thinking' || block.thinking !== REASONING_MARK) return null
  try {
    const { id, ec } = JSON.parse(block.signature)
    return { type: 'reasoning', id, encrypted_content: ec, summary: [] }
  } catch {
    return null
  }
}

/** Anthropic messages[] -> Responses input[] */
function toResponsesInput(messages: Json[]): Json[] {
  const input: Json[] = []

  for (const msg of messages) {
    const blocks = typeof msg.content === 'string'
      ? [{ type: 'text', text: msg.content }]
      : (msg.content ?? [])

    if (msg.role === 'user') {
      const parts: Json[] = []
      for (const b of blocks) {
        if (b.type === 'text') {
          parts.push({ type: 'input_text', text: b.text })
        } else if (b.type === 'image' && b.source?.type === 'base64') {
          parts.push({ type: 'input_image', image_url: `data:${b.source.media_type};base64,${b.source.data}` })
        } else if (b.type === 'tool_result') {
          // A tool_result is a standalone item upstream, not part of a message.
          const out = typeof b.content === 'string' ? b.content : textOf(b.content)
          input.push({
            type: 'function_call_output',
            call_id: b.tool_use_id,
            output: out || (b.is_error ? 'error' : ''),
          })
        }
      }
      if (parts.length) input.push({ role: 'user', content: parts })
    }

    if (msg.role === 'assistant') {
      const parts: Json[] = []
      const flush = () => {
        if (parts.length) { input.push({ role: 'assistant', content: [...parts] }); parts.length = 0 }
      }
      for (const b of blocks) {
        const reasoning = decodeReasoning(b)
        if (reasoning) {
          // Reasoning must precede the call it produced.
          flush()
          input.push(reasoning)
        } else if (b.type === 'text' && b.text) {
          parts.push({ type: 'output_text', text: b.text })
        } else if (b.type === 'tool_use') {
          flush()
          input.push({
            type: 'function_call',
            call_id: b.id,
            name: b.name,
            arguments: JSON.stringify(b.input ?? {}),
          })
        }
      }
      flush()
    }
  }
  return input
}

function toResponsesTools(tools: Json[] | undefined): Json[] | undefined {
  if (!tools?.length) return undefined
  const out = tools
    .filter((t: Json) => t?.input_schema || t?.custom?.input_schema)
    .map((t: Json) => ({
      type: 'function',
      name: t.name,
      description: t.description ?? '',
      parameters: t.input_schema ?? t.custom?.input_schema,
      // CC's schemas aren't strict-mode clean (no additionalProperties:false).
      strict: false,
    }))
  return out.length ? out : undefined
}

function toToolChoice(tc: Json): Json | undefined {
  if (!tc) return undefined
  if (tc.type === 'auto') return 'auto'
  if (tc.type === 'any') return 'required'
  if (tc.type === 'none') return 'none'
  if (tc.type === 'tool') return { type: 'function', name: tc.name }
  return undefined
}

function buildUpstream(body: Json) {
  const model = mapModel(body.model)
  const req: Json = {
    model,
    input: toResponsesInput(body.messages ?? []),
    store: false,
    stream: Boolean(body.stream),
  }
  const sys = body.system ? textOf(body.system) : ''
  if (sys) req.instructions = sys

  const tools = toResponsesTools(body.tools)
  if (tools) req.tools = tools

  const tc = toToolChoice(body.tool_choice)
  if (tc) req.tool_choice = tc

  if (body.max_tokens) req.max_output_tokens = Math.max(16, body.max_tokens)

  // Only reasoning models accept the knob; the cheap fallback does not.
  if (model.startsWith('gpt-5.6')) {
    req.reasoning = { effort: EFFORT }
    if (CARRY_REASONING) req.include = ['reasoning.encrypted_content']
  }
  return req
}

const sse = (res: http.ServerResponse, event: string, data: Json) => {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
}

function stopReasonFrom(resp: Json): string {
  const hasCall = (resp?.output ?? []).some((i: Json) => i.type === 'function_call')
  if (hasCall) return 'tool_use'
  if (resp?.status === 'incomplete') return 'max_tokens'
  return 'end_turn'
}

function safeParse(s: string): Json {
  try { return JSON.parse(s || '{}') } catch { return {} }
}

async function handleMessages(reqBody: Json, res: http.ServerResponse) {
  const upstreamReq = buildUpstream(reqBody)
  const reasoningIn = upstreamReq.input.filter((i: Json) => i.type === 'reasoning').length
  debug('->', upstreamReq.model, 'items=' + upstreamReq.input.length,
    'reasoning=' + reasoningIn, 'tools=' + (upstreamReq.tools?.length ?? 0))

  const upstream = await fetch(UPSTREAM, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${OPENAI_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(upstreamReq),
  })

  if (!upstream.ok) {
    const errText = await upstream.text()
    log('upstream', upstream.status, errText.slice(0, 400))
    res.writeHead(upstream.status, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ type: 'error', error: { type: 'api_error', message: errText.slice(0, 2000) } }))
    return
  }

  const msgId = `msg_${Math.abs(Date.now() % 1e9)}`

  if (!reqBody.stream) {
    const data: Json = await upstream.json()
    const content: Json[] = []
    for (const item of data.output ?? []) {
      if (item.type === 'reasoning') {
        const enc = encodeReasoning(item)
        if (enc) content.push(enc)
      } else if (item.type === 'message') {
        for (const c of item.content ?? []) {
          if (c.type === 'output_text') content.push({ type: 'text', text: c.text })
        }
      } else if (item.type === 'function_call') {
        content.push({ type: 'tool_use', id: item.call_id, name: item.name, input: safeParse(item.arguments) })
      }
    }
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({
      id: msgId,
      type: 'message',
      role: 'assistant',
      model: reqBody.model,
      content,
      stop_reason: stopReasonFrom(data),
      stop_sequence: null,
      usage: {
        input_tokens: data.usage?.input_tokens ?? 0,
        output_tokens: data.usage?.output_tokens ?? 0,
      },
    }))
    return
  }

  // ---- streaming ----
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  })

  sse(res, 'message_start', {
    type: 'message_start',
    message: {
      id: msgId,
      type: 'message',
      role: 'assistant',
      model: reqBody.model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  })

  let blockIndex = -1
  const openBlocks = new Map<string, number>() // upstream item id -> anthropic index
  let stopReason = 'end_turn'
  let outputTokens = 0

  const closeBlock = (idx: number) => sse(res, 'content_block_stop', { type: 'content_block_stop', index: idx })

  const reader = upstream.body!.getReader()
  const decoder = new TextDecoder()
  let buf = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })

    const chunks = buf.split('\n\n')
    buf = chunks.pop() ?? ''

    for (const chunk of chunks) {
      const dataLine = chunk.split('\n').find((l) => l.startsWith('data:'))
      if (!dataLine) continue
      const payload = dataLine.slice(5).trim()
      if (!payload || payload === '[DONE]') continue

      let ev: Json
      try { ev = JSON.parse(payload) } catch { continue }

      switch (ev.type) {
        case 'response.output_item.added': {
          const item = ev.item
          if (item?.type === 'message') {
            blockIndex++
            openBlocks.set(item.id, blockIndex)
            sse(res, 'content_block_start', {
              type: 'content_block_start',
              index: blockIndex,
              content_block: { type: 'text', text: '' },
            })
          } else if (item?.type === 'function_call') {
            blockIndex++
            openBlocks.set(item.id, blockIndex)
            stopReason = 'tool_use'
            sse(res, 'content_block_start', {
              type: 'content_block_start',
              index: blockIndex,
              content_block: { type: 'tool_use', id: item.call_id, name: item.name, input: {} },
            })
          }
          break
        }
        case 'response.output_text.delta': {
          const idx = openBlocks.get(ev.item_id)
          if (idx === undefined) break
          sse(res, 'content_block_delta', {
            type: 'content_block_delta',
            index: idx,
            delta: { type: 'text_delta', text: ev.delta },
          })
          break
        }
        case 'response.function_call_arguments.delta': {
          const idx = openBlocks.get(ev.item_id)
          if (idx === undefined) break
          sse(res, 'content_block_delta', {
            type: 'content_block_delta',
            index: idx,
            delta: { type: 'input_json_delta', partial_json: ev.delta },
          })
          break
        }
        case 'response.output_item.done': {
          const item = ev.item
          const idx = openBlocks.get(item?.id)
          if (idx !== undefined) {
            closeBlock(idx)
            openBlocks.delete(item.id)
          } else if (item?.type === 'reasoning') {
            // Reasoning arrives complete (never streamed); emit as one block so
            // CC hands the encrypted blob back to us next turn.
            const enc = encodeReasoning(item)
            if (enc) {
              blockIndex++
              sse(res, 'content_block_start', {
                type: 'content_block_start',
                index: blockIndex,
                content_block: { type: 'thinking', thinking: '' },
              })
              sse(res, 'content_block_delta', {
                type: 'content_block_delta',
                index: blockIndex,
                delta: { type: 'thinking_delta', thinking: enc.thinking },
              })
              sse(res, 'content_block_delta', {
                type: 'content_block_delta',
                index: blockIndex,
                delta: { type: 'signature_delta', signature: enc.signature },
              })
              closeBlock(blockIndex)
            }
          }
          break
        }
        case 'response.completed':
        case 'response.incomplete': {
          outputTokens = ev.response?.usage?.output_tokens ?? 0
          stopReason = stopReasonFrom(ev.response)
          break
        }
        case 'response.failed':
        case 'error': {
          log('stream error', JSON.stringify(ev).slice(0, 300))
          break
        }
      }
    }
  }

  for (const idx of openBlocks.values()) closeBlock(idx)

  sse(res, 'message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: outputTokens },
  })
  sse(res, 'message_stop', { type: 'message_stop' })
  res.end()
}

const server = http.createServer((req, res) => {
  let body = ''
  req.on('data', (c) => { body += c })
  req.on('end', async () => {
    const url = req.url ?? ''
    debug('<-', req.method, url)

    // CC probes this before anything else; a 404 here surfaces to the user as
    // "Not logged in - Please run /login".
    if (url.startsWith('/api/hello')) {
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end('{}')
      return
    }

    if (url.startsWith('/v1/messages/count_tokens')) {
      // Rough estimate; CC uses it only for context-pressure hints.
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ input_tokens: Math.ceil(body.length / 4) }))
      return
    }

    if (url.startsWith('/v1/messages')) {
      try {
        await handleMessages(JSON.parse(body || '{}'), res)
      } catch (e: unknown) {
        log('handler crash', e)
        if (!res.headersSent) {
          res.writeHead(500, { 'Content-Type': 'application/json' })
          res.end(JSON.stringify({ type: 'error', error: { type: 'api_error', message: String(e) } }))
        } else {
          res.end()
        }
      }
      return
    }

    res.writeHead(404, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'not found' }))
  })
})

server.listen(PORT, '127.0.0.1', () => {
  const port = (server.address() as { port: number }).port
  log(`ready port=${port} effort=${EFFORT} carry_reasoning=${CARRY_REASONING ? 'on' : 'off'}`)
})
