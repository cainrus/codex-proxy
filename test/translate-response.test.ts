/**
 * Upstream -> downstream: a Responses API response body becomes an Anthropic
 * non-streaming message. No network; the response bodies are literals.
 */
import assert from 'node:assert/strict'
import { describe, test } from 'node:test'

import {
  decodeReasoning,
  safeParse,
  stopReasonFrom,
  toAnthropicMessage,
  REASONING_MARK,
  type Config,
} from '../codex-proxy.ts'

const CFG: Config = { effort: 'medium', carryReasoning: true, fallbackModel: 'gpt-5.4-mini' }
const NO_CARRY: Config = { ...CFG, carryReasoning: false }
const WHO = { id: 'msg_fixture_0001', model: 'codex-sol' }

const textResponse = {
  id: 'resp_fixture_0001',
  status: 'completed',
  output: [{
    id: 'msg_upstream_0001',
    type: 'message',
    role: 'assistant',
    content: [{ type: 'output_text', text: 'The readme is 69 lines.' }],
  }],
  usage: { input_tokens: 1204, output_tokens: 48 },
}

describe('safeParse', () => {
  test('parses JSON, and never throws on rubbish', () => {
    assert.deepEqual(safeParse('{"a":1}'), { a: 1 })
    // A truncated tool call must not take the whole response down.
    assert.deepEqual(safeParse('{"a":'), {})
    assert.deepEqual(safeParse(''), {})
  })
})

describe('stopReasonFrom', () => {
  test('a completed answer ends the turn', () => {
    assert.equal(stopReasonFrom({ status: 'completed', output: [{ type: 'message' }] }), 'end_turn')
  })

  test('an incomplete response reports the token cap', () => {
    assert.equal(stopReasonFrom({ status: 'incomplete', output: [] }), 'max_tokens')
  })

  test('a pending tool call outranks everything else', () => {
    // The client must run the tool; telling it max_tokens would end the turn.
    assert.equal(
      stopReasonFrom({ status: 'incomplete', output: [{ type: 'function_call' }] }),
      'tool_use',
    )
  })

  test('a response with no output at all still yields a valid stop reason', () => {
    assert.equal(stopReasonFrom({}), 'end_turn')
    assert.equal(stopReasonFrom(undefined), 'end_turn')
  })
})

describe('toAnthropicMessage', () => {
  test('a plain answer becomes one text block with usage carried over', () => {
    assert.deepEqual(toAnthropicMessage(textResponse, WHO, CFG), {
      id: 'msg_fixture_0001',
      type: 'message',
      role: 'assistant',
      model: 'codex-sol',
      content: [{ type: 'text', text: 'The readme is 69 lines.' }],
      stop_reason: 'end_turn',
      stop_sequence: null,
      usage: { input_tokens: 1204, output_tokens: 48 },
    })
  })

  test('the model echoed back is the alias the client asked for', () => {
    // The client validates the echo against what it sent; answering
    // "gpt-5.6-sol" to a request for "codex-sol" confuses it.
    assert.equal(toAnthropicMessage(textResponse, WHO, CFG).model, 'codex-sol')
  })

  test('a function_call becomes a tool_use block with parsed input', () => {
    const msg = toAnthropicMessage({
      status: 'completed',
      output: [{
        id: 'fc_upstream_0001',
        type: 'function_call',
        call_id: 'call_fixture_0001',
        name: 'Read',
        arguments: '{"file_path":"README.md"}',
      }],
    }, WHO, CFG)
    assert.deepEqual(msg.content, [{
      type: 'tool_use',
      // The client sends this id back as tool_use_id, and Responses matches
      // function_call_output on call_id -- so it is call_id, not the item id.
      id: 'call_fixture_0001',
      name: 'Read',
      input: { file_path: 'README.md' },
    }])
    assert.equal(msg.stop_reason, 'tool_use')
  })

  test('truncated tool arguments degrade to an empty input, not a crash', () => {
    const msg = toAnthropicMessage({
      status: 'incomplete',
      output: [{ type: 'function_call', call_id: 'call_fixture_0002', name: 'Read', arguments: '{"file_pa' }],
    }, WHO, CFG)
    assert.deepEqual(msg.content[0].input, {})
  })

  test('text and a tool call keep their upstream order', () => {
    const msg = toAnthropicMessage({
      status: 'completed',
      output: [
        { type: 'message', content: [{ type: 'output_text', text: 'Reading it.' }] },
        { type: 'function_call', call_id: 'call_fixture_0003', name: 'Read', arguments: '{}' },
      ],
    }, WHO, CFG)
    assert.deepEqual(msg.content.map((b: any) => b.type), ['text', 'tool_use'])
  })

  test('several output_text parts in one message become separate blocks', () => {
    const msg = toAnthropicMessage({
      output: [{
        type: 'message',
        content: [
          { type: 'output_text', text: 'one' },
          { type: 'refusal', refusal: 'nope' },
          { type: 'output_text', text: 'two' },
        ],
      }],
    }, WHO, CFG)
    assert.deepEqual(msg.content, [{ type: 'text', text: 'one' }, { type: 'text', text: 'two' }])
  })

  test('a reasoning item survives as a thinking block the client will hand back', () => {
    const msg = toAnthropicMessage({
      status: 'completed',
      output: [
        { id: 'rs_fixture_0001', type: 'reasoning', encrypted_content: 'ZW5jcnlwdGVkLWJsb2I=' },
        { type: 'message', content: [{ type: 'output_text', text: 'done' }] },
      ],
    }, WHO, CFG)
    assert.equal(msg.content[0].type, 'thinking')
    assert.equal(msg.content[0].thinking, REASONING_MARK)
    // The whole point: the next request can rebuild the upstream item from it.
    assert.deepEqual(decodeReasoning(msg.content[0], CFG), {
      type: 'reasoning',
      id: 'rs_fixture_0001',
      encrypted_content: 'ZW5jcnlwdGVkLWJsb2I=',
      summary: [],
    })
  })

  test('with carrying off the reasoning item is dropped from the answer', () => {
    const msg = toAnthropicMessage({
      output: [
        { id: 'rs_fixture_0001', type: 'reasoning', encrypted_content: 'ZW5jcnlwdGVkLWJsb2I=' },
        { type: 'message', content: [{ type: 'output_text', text: 'done' }] },
      ],
    }, WHO, NO_CARRY)
    assert.deepEqual(msg.content, [{ type: 'text', text: 'done' }])
  })

  test('an unsummarised reasoning item adds no empty block', () => {
    const msg = toAnthropicMessage({ output: [{ id: 'rs_fixture_0002', type: 'reasoning', summary: [] }] }, WHO, CFG)
    assert.deepEqual(msg.content, [])
  })

  test('missing usage reports zeros rather than undefined', () => {
    const msg = toAnthropicMessage({ status: 'completed', output: [] }, WHO, CFG)
    assert.deepEqual(msg.usage, { input_tokens: 0, output_tokens: 0 })
    assert.deepEqual(msg.content, [])
    assert.equal(msg.stop_reason, 'end_turn')
  })

  test('an empty upstream body still produces a valid Anthropic message', () => {
    const msg = toAnthropicMessage({}, WHO, CFG)
    assert.equal(msg.type, 'message')
    assert.equal(msg.role, 'assistant')
    assert.equal(msg.stop_sequence, null)
    assert.deepEqual(msg.content, [])
  })
})
