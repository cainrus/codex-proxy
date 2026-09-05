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
 *
 * Everything above the server is a pure translation layer and is exported, so
 * the test suite can pin it without opening a socket. The server only starts
 * when this file is the process entry point -- importing it is side-effect
 * free.
 */
import http from 'node:http'
import { pathToFileURL } from 'node:url'

const PORT = Number(process.env.PORT ?? 4001)
const UPSTREAM = process.env.CODEX_UPSTREAM ?? 'https://api.openai.com/v1/responses'

export const MODEL_MAP: Record<string, string> = {
  'codex-luna': 'gpt-5.6-luna',
  'codex-sol': 'gpt-5.6-sol',
  'codex-terra': 'gpt-5.6-terra',
}

type Json = any

/**
 * Knobs read from the environment once at startup. Passed explicitly through
 * the translation layer so a caller (or a test) can pin one without touching
 * process.env.
 */
export interface Config {
  /** reasoning.effort sent upstream for gpt-5.6-* models. */
  effort: string
  /** Round-trip encrypted reasoning items across tool rounds. */
  carryReasoning: boolean
  /**
   * CC fires background calls (conversation titles, quota probes) on haiku no
   * matter what --model says; route them somewhere cheap instead of 400-ing.
   */
  fallbackModel: string
}

export function configFromEnv(env: NodeJS.ProcessEnv = process.env): Config {
  return {
    effort: env.CODEX_EFFORT ?? 'medium',
    carryReasoning: env.CODEX_CARRY_REASONING !== '0',
    fallbackModel: env.CODEX_FALLBACK_MODEL ?? 'gpt-5.4-mini',
  }
}

const CONFIG: Config = configFromEnv()

const DEBUG = process.env.CODEX_DEBUG === '1'
const log = (...a: unknown[]) => console.error('[codex-proxy]', ...a)
const debug = (...a: unknown[]) => { if (DEBUG) log(...a) }

export function mapModel(m: string, cfg: Config = CONFIG): string {
  if (MODEL_MAP[m]) return MODEL_MAP[m]
  if (m?.startsWith('gpt-')) return m
  return cfg.fallbackModel
}

export function textOf(content: Json): string {
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
/** Leading U+200B keeps the marker invisible if a client ever renders it. */
export const REASONING_MARK = '​<codex-reasoning/>'

export function encodeReasoning(item: Json, cfg: Config = CONFIG): Json | null {
  if (!cfg.carryReasoning || !item?.encrypted_content) return null
  return {
    type: 'thinking',
    thinking: REASONING_MARK,
    signature: JSON.stringify({ id: item.id, ec: item.encrypted_content }),
  }
}

export function decodeReasoning(block: Json, cfg: Config = CONFIG): Json | null {
  if (!cfg.carryReasoning) return null
  if (block?.type !== 'thinking' || block.thinking !== REASONING_MARK) return null
  try {
    const { id, ec } = JSON.parse(block.signature)
    return { type: 'reasoning', id, encrypted_content: ec, summary: [] }
  } catch {
    return null
  }
}

/** Anthropic messages[] -> Responses input[] */
export function toResponsesInput(messages: Json[], cfg: Config = CONFIG): Json[] {
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
        const reasoning = decodeReasoning(b, cfg)
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

export function toResponsesTools(tools: Json[] | undefined): Json[] | undefined {
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

export function toToolChoice(tc: Json): Json | undefined {
  if (!tc) return undefined
  if (tc.type === 'auto') return 'auto'
  if (tc.type === 'any') return 'required'
  if (tc.type === 'none') return 'none'
  if (tc.type === 'tool') return { type: 'function', name: tc.name }
  return undefined
}

/** Anthropic /v1/messages request body -> Responses API request body. */
export function buildUpstream(body: Json, cfg: Config = CONFIG) {
  const model = mapModel(body.model, cfg)
  const req: Json = {
    model,
    input: toResponsesInput(body.messages ?? [], cfg),
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
    req.reasoning = { effort: cfg.effort }
    if (cfg.carryReasoning) req.include = ['reasoning.encrypted_content']
  }
  return req
}

export function stopReasonFrom(resp: Json): string {
  const hasCall = (resp?.output ?? []).some((i: Json) => i.type === 'function_call')
  if (hasCall) return 'tool_use'
  if (resp?.status === 'incomplete') return 'max_tokens'
  return 'end_turn'
}

export function safeParse(s: string): Json {
  try { return JSON.parse(s || '{}') } catch { return {} }
}

export interface MessageIdentity {
  /** Anthropic message id echoed back to the client. */
  id: string
  /** The model name the client asked for, not the one we mapped to. */
  model: string
}

/** Responses API response body -> Anthropic non-streaming message. */
export function toAnthropicMessage(data: Json, who: MessageIdentity, cfg: Config = CONFIG): Json {
  const content: Json[] = []
  for (const item of data?.output ?? []) {
    if (item.type === 'reasoning') {
      const enc = encodeReasoning(item, cfg)
      if (enc) content.push(enc)
    } else if (item.type === 'message') {
      for (const c of item.content ?? []) {
        if (c.type === 'output_text') content.push({ type: 'text', text: c.text })
      }
    } else if (item.type === 'function_call') {
      content.push({ type: 'tool_use', id: item.call_id, name: item.name, input: safeParse(item.arguments) })
    }
  }
  return {
    id: who.id,
    type: 'message',
    role: 'assistant',
    model: who.model,
    content,
    stop_reason: stopReasonFrom(data),
    stop_sequence: null,
    usage: {
      input_tokens: data?.usage?.input_tokens ?? 0,
      output_tokens: data?.usage?.output_tokens ?? 0,
    },
  }
}

/** Sink for translated Anthropic SSE events: (eventName, payload). */
export type EmitEvent = (event: string, data: Json) => void

export interface StreamTranslator {
  /** Emit message_start. Call once, before the first push. */
  start(): void
  /** Feed a raw upstream SSE chunk; partial events are buffered. */
  push(text: string): void
  /** Close any open blocks and emit message_delta + message_stop. */
  end(): void
}

/**
 * Rebuilds an Anthropic SSE stream from an OpenAI Responses SSE stream.
 *
 * Transport-free on purpose: it takes raw text in and hands events to `emit`,
 * so both the live socket and the test fixtures drive the same code.
 */
export function createStreamTranslator(
  emit: EmitEvent,
  who: MessageIdentity,
  cfg: Config = CONFIG,
): StreamTranslator {
  let blockIndex = -1
  const openBlocks = new Map<string, number>() // upstream item id -> anthropic index
  let stopReason = 'end_turn'
  let outputTokens = 0
  let buf = ''

  const closeBlock = (idx: number) => emit('content_block_stop', { type: 'content_block_stop', index: idx })

  const onEvent = (ev: Json) => {
    switch (ev.type) {
      case 'response.output_item.added': {
        const item = ev.item
        if (item?.type === 'message') {
          blockIndex++
          openBlocks.set(item.id, blockIndex)
          emit('content_block_start', {
            type: 'content_block_start',
            index: blockIndex,
            content_block: { type: 'text', text: '' },
          })
        } else if (item?.type === 'function_call') {
          blockIndex++
          openBlocks.set(item.id, blockIndex)
          stopReason = 'tool_use'
          emit('content_block_start', {
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
        emit('content_block_delta', {
          type: 'content_block_delta',
          index: idx,
          delta: { type: 'text_delta', text: ev.delta },
        })
        break
      }
      case 'response.function_call_arguments.delta': {
        const idx = openBlocks.get(ev.item_id)
        if (idx === undefined) break
        emit('content_block_delta', {
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
          const enc = encodeReasoning(item, cfg)
          if (enc) {
            blockIndex++
            emit('content_block_start', {
              type: 'content_block_start',
              index: blockIndex,
              content_block: { type: 'thinking', thinking: '' },
            })
            emit('content_block_delta', {
              type: 'content_block_delta',
              index: blockIndex,
              delta: { type: 'thinking_delta', thinking: enc.thinking },
            })
            emit('content_block_delta', {
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

  return {
    start() {
      emit('message_start', {
        type: 'message_start',
        message: {
          id: who.id,
          type: 'message',
          role: 'assistant',
          model: who.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      })
    },

    push(text: string) {
      buf += text
      const chunks = buf.split('\n\n')
      buf = chunks.pop() ?? ''

      for (const chunk of chunks) {
        const dataLine = chunk.split('\n').find((l) => l.startsWith('data:'))
        if (!dataLine) continue
        const payload = dataLine.slice(5).trim()
        if (!payload || payload === '[DONE]') continue

        let ev: Json
        try { ev = JSON.parse(payload) } catch { continue }
        onEvent(ev)
      }
    },

    end() {
      for (const idx of openBlocks.values()) closeBlock(idx)

      emit('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens },
      })
      emit('message_stop', { type: 'message_stop' })
    },
  }
}

const sse = (res: http.ServerResponse, event: string, data: Json) => {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
}

export function newMessageId(): string {
  return `msg_${Math.abs(Date.now() % 1e9)}`
}

async function handleMessages(reqBody: Json, res: http.ServerResponse) {
  const upstreamReq = buildUpstream(reqBody)
  const reasoningIn = upstreamReq.input.filter((i: Json) => i.type === 'reasoning').length
  debug('->', upstreamReq.model, 'items=' + upstreamReq.input.length,
    'reasoning=' + reasoningIn, 'tools=' + (upstreamReq.tools?.length ?? 0))

  const upstream = await fetch(UPSTREAM, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(upstreamReq),
  })

  if (!upstream.ok) {
    const errText = await upstream.text()
    log('upstream', upstream.status, errText.slice(0, 400))
    res.writeHead(upstream.status, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ type: 'error', error: { type: 'api_error', message: errText.slice(0, 2000) } }))
    return
  }

  const who: MessageIdentity = { id: newMessageId(), model: reqBody.model }

  if (!reqBody.stream) {
    const data: Json = await upstream.json()
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify(toAnthropicMessage(data, who)))
    return
  }

  // ---- streaming ----
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  })

  const stream = createStreamTranslator((event, data) => sse(res, event, data), who)
  stream.start()

  const reader = upstream.body!.getReader()
  const decoder = new TextDecoder()

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    stream.push(decoder.decode(value, { stream: true }))
  }

  stream.end()
  res.end()
}

/** Route one collected request. Split out from the server so it is callable without a socket. */
export async function handleRequest(url: string, body: string, res: http.ServerResponse): Promise<void> {
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
}

export function createProxyServer(): http.Server {
  return http.createServer((req, res) => {
    let body = ''
    req.on('data', (c) => { body += c })
    req.on('end', () => {
      const url = req.url ?? ''
      debug('<-', req.method, url)
      void handleRequest(url, body, res)
    })
  })
}

function main() {
  if (!process.env.OPENAI_API_KEY) {
    log('FATAL: OPENAI_API_KEY is not set')
    process.exit(1)
  }

  const server = createProxyServer()
  server.listen(PORT, '127.0.0.1', () => {
    const port = (server.address() as { port: number }).port
    log(`ready port=${port} effort=${CONFIG.effort} carry_reasoning=${CONFIG.carryReasoning ? 'on' : 'off'}`)
  })
}

const entry = process.argv[1] ? pathToFileURL(process.argv[1]).href : ''
if (entry === import.meta.url) main()
