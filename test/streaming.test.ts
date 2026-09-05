/**
 * SSE translation, driven by a recorded upstream stream in
 * fixtures/responses-stream.sse.txt. The translator takes raw text and hands
 * events to a callback, so these tests exercise the same code the socket does
 * without opening one.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { describe, test } from 'node:test'

import { createStreamTranslator, REASONING_MARK, type Config } from '../codex-proxy.ts'

const CFG: Config = { effort: 'medium', carryReasoning: true, fallbackModel: 'gpt-5.4-mini' }
const NO_CARRY: Config = { ...CFG, carryReasoning: false }
const WHO = { id: 'msg_fixture_0001', model: 'codex-sol' }

const FIXTURE = readFileSync(new URL('./fixtures/responses-stream.sse.txt', import.meta.url), 'utf8')

interface Emitted { event: string; data: any }

function run(chunks: string[], cfg: Config = CFG): Emitted[] {
  const out: Emitted[] = []
  const stream = createStreamTranslator((event, data) => out.push({ event, data }), WHO, cfg)
  stream.start()
  for (const c of chunks) stream.push(c)
  stream.end()
  return out
}

/** Split a string into fixed-size pieces, to fake arbitrary socket framing. */
function slice(text: string, size: number): string[] {
  const out: string[] = []
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size))
  return out
}

const names = (events: Emitted[]) => events.map((e) => e.event)

describe('a recorded reasoning + text + tool-call stream', () => {
  const events = run([FIXTURE])

  test('opens with message_start carrying the client-facing id and model', () => {
    assert.equal(events[0].event, 'message_start')
    assert.equal(events[0].data.message.id, 'msg_fixture_0001')
    assert.equal(events[0].data.message.model, 'codex-sol')
    assert.equal(events[0].data.message.stop_reason, null)
  })

  test('produces the full Anthropic event sequence, blocks balanced', () => {
    assert.deepEqual(names(events), [
      'message_start',
      // reasoning: never streamed, so it arrives as one complete block
      'content_block_start', 'content_block_delta', 'content_block_delta', 'content_block_stop',
      // assistant text
      'content_block_start', 'content_block_delta', 'content_block_delta', 'content_block_stop',
      // tool call
      'content_block_start', 'content_block_delta', 'content_block_delta', 'content_block_stop',
      'message_delta', 'message_stop',
    ])
    const starts = events.filter((e) => e.event === 'content_block_start').length
    const stops = events.filter((e) => e.event === 'content_block_stop').length
    assert.equal(starts, stops)
  })

  test('block indices are contiguous from zero and never reused', () => {
    const indices = events
      .filter((e) => e.event === 'content_block_start')
      .map((e) => e.data.index)
    assert.deepEqual(indices, [0, 1, 2])
    for (const e of events) {
      if (e.event === 'content_block_delta' || e.event === 'content_block_stop') {
        assert.ok(indices.includes(e.data.index), `index ${e.data.index} was never opened`)
      }
    }
  })

  test('the reasoning block carries the marker and the encrypted blob', () => {
    const [thinking, signature] = events.filter((e) => e.event === 'content_block_delta' && e.data.index === 0)
    assert.equal(events[1].data.content_block.type, 'thinking')
    assert.equal(thinking.data.delta.type, 'thinking_delta')
    assert.equal(thinking.data.delta.thinking, REASONING_MARK)
    assert.equal(signature.data.delta.type, 'signature_delta')
    assert.deepEqual(JSON.parse(signature.data.delta.signature), {
      id: 'rs_fixture_0001',
      ec: 'ZW5jcnlwdGVkLXJlYXNvbmluZy1ibG9i',
    })
  })

  test('text deltas reassemble into the sentence the model wrote', () => {
    const text = events
      .filter((e) => e.event === 'content_block_delta' && e.data.delta.type === 'text_delta')
      .map((e) => e.data.delta.text)
      .join('')
    assert.equal(text, 'Reading the file.')
  })

  test('the tool block announces call_id and name up front, arguments as JSON deltas', () => {
    const start = events.filter((e) => e.event === 'content_block_start')[2]
    assert.deepEqual(start.data.content_block, {
      type: 'tool_use',
      id: 'call_fixture_0001',
      name: 'Read',
      input: {},
    })
    const partial = events
      .filter((e) => e.event === 'content_block_delta' && e.data.delta.type === 'input_json_delta')
      .map((e) => e.data.delta.partial_json)
      .join('')
    assert.deepEqual(JSON.parse(partial), { file_path: 'README.md' })
  })

  test('closes with tool_use and the upstream output-token count', () => {
    const delta = events.at(-2)!
    assert.equal(delta.event, 'message_delta')
    assert.deepEqual(delta.data.delta, { stop_reason: 'tool_use', stop_sequence: null })
    assert.deepEqual(delta.data.usage, { output_tokens: 48 })
    assert.equal(events.at(-1)!.event, 'message_stop')
  })
})

describe('chunk framing', () => {
  test('a stream split mid-event yields exactly the same output', () => {
    // The socket hands over arbitrary byte runs; SSE events span them.
    const whole = run([FIXTURE])
    for (const size of [1, 7, 64, 997]) {
      assert.deepEqual(run(slice(FIXTURE, size)), whole, `framing broke at chunk size ${size}`)
    }
  })

  test('a chunk boundary inside a JSON string is not treated as an event end', () => {
    const cut = FIXTURE.indexOf('README.md') + 4
    assert.deepEqual(run([FIXTURE.slice(0, cut), FIXTURE.slice(cut)]), run([FIXTURE]))
  })
})

describe('stream edge cases', () => {
  test('unparseable and terminator payloads are skipped, not fatal', () => {
    const events = run([
      'event: error\ndata: not json at all\n\n',
      'data: [DONE]\n\n',
      ': keep-alive comment\n\n',
      'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","output":[],"usage":{"output_tokens":3}}}\n\n',
    ])
    assert.deepEqual(names(events), ['message_start', 'message_delta', 'message_stop'])
    assert.equal(events[1].data.delta.stop_reason, 'end_turn')
    assert.equal(events[1].data.usage.output_tokens, 3)
  })

  test('a stream that only errors still terminates the message', () => {
    const events = run(['event: response.failed\ndata: {"type":"response.failed","response":{"error":{"message":"upstream boom"}}}\n\n'])
    assert.deepEqual(names(events), ['message_start', 'message_delta', 'message_stop'])
    assert.equal(events[1].data.delta.stop_reason, 'end_turn')
  })

  test('response.incomplete reports the token cap downstream', () => {
    const events = run([
      'event: response.incomplete\ndata: {"type":"response.incomplete","response":{"status":"incomplete","output":[],"usage":{"output_tokens":16}}}\n\n',
    ])
    assert.equal(events[1].data.delta.stop_reason, 'max_tokens')
    assert.equal(events[1].data.usage.output_tokens, 16)
  })

  test('a block left open by a dropped connection is still closed', () => {
    // Upstream died after output_item.added; the client would hang forever on
    // an unclosed block.
    const events = run([
      'event: response.output_item.added\ndata: {"type":"response.output_item.added","item":{"id":"msg_upstream_0001","type":"message"}}\n\n',
      'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"msg_upstream_0001","delta":"half a sen"}\n\n',
    ])
    assert.deepEqual(names(events), [
      'message_start', 'content_block_start', 'content_block_delta',
      'content_block_stop', 'message_delta', 'message_stop',
    ])
    assert.equal(events.at(-3)!.data.index, 0)
  })

  test('a delta for an item that was never opened is ignored', () => {
    const events = run([
      'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"msg_unknown","delta":"orphan"}\n\n',
    ])
    assert.deepEqual(names(events), ['message_start', 'message_delta', 'message_stop'])
  })

  test('an empty stream is still a well-formed Anthropic message', () => {
    assert.deepEqual(names(run([])), ['message_start', 'message_delta', 'message_stop'])
    assert.deepEqual(names(run([''])), ['message_start', 'message_delta', 'message_stop'])
  })

  test('with carrying off no thinking block is streamed at all', () => {
    const events = run([FIXTURE], NO_CARRY)
    assert.equal(events.some((e) => e.event === 'content_block_start' && e.data.content_block.type === 'thinking'), false)
    // Indices stay contiguous: text and tool call shift down to 0 and 1.
    assert.deepEqual(
      events.filter((e) => e.event === 'content_block_start').map((e) => e.data.index),
      [0, 1],
    )
  })
})
