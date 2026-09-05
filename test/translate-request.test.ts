/**
 * Downstream -> upstream: an Anthropic /v1/messages body becomes a Responses
 * API request body. Pure functions only; nothing here opens a socket.
 */
import assert from 'node:assert/strict'
import { describe, test } from 'node:test'

import {
  buildUpstream,
  configFromEnv,
  decodeReasoning,
  encodeReasoning,
  mapModel,
  REASONING_MARK,
  textOf,
  toResponsesInput,
  toResponsesTools,
  toToolChoice,
  type Config,
} from '../codex-proxy.ts'

const CFG: Config = { effort: 'medium', carryReasoning: true, fallbackModel: 'gpt-5.4-mini' }
const NO_CARRY: Config = { ...CFG, carryReasoning: false }

describe('mapModel', () => {
  test('resolves the codex-* aliases to gpt-5.6-* names', () => {
    assert.equal(mapModel('codex-luna', CFG), 'gpt-5.6-luna')
    assert.equal(mapModel('codex-sol', CFG), 'gpt-5.6-sol')
    assert.equal(mapModel('codex-terra', CFG), 'gpt-5.6-terra')
  })

  test('passes any gpt-* name through unchanged', () => {
    assert.equal(mapModel('gpt-5.6-sol', CFG), 'gpt-5.6-sol')
    assert.equal(mapModel('gpt-4.1-mini', CFG), 'gpt-4.1-mini')
  })

  test('routes an unknown model to the fallback instead of failing', () => {
    // This is the whole reason the fallback exists: the client fires
    // background calls (titles, quota probes) on a model we never mapped.
    assert.equal(mapModel('claude-haiku-4-5-20251001', CFG), 'gpt-5.4-mini')
    assert.equal(mapModel('', CFG), 'gpt-5.4-mini')
    assert.equal(mapModel('claude-haiku-4-5-20251001', { ...CFG, fallbackModel: 'gpt-4.1-nano' }), 'gpt-4.1-nano')
  })
})

describe('configFromEnv', () => {
  test('defaults with an empty environment', () => {
    assert.deepEqual(configFromEnv({}), {
      effort: 'medium',
      carryReasoning: true,
      fallbackModel: 'gpt-5.4-mini',
    })
  })

  test('CODEX_CARRY_REASONING is off only for the exact string "0"', () => {
    assert.equal(configFromEnv({ CODEX_CARRY_REASONING: '0' }).carryReasoning, false)
    assert.equal(configFromEnv({ CODEX_CARRY_REASONING: 'false' }).carryReasoning, true)
    assert.equal(configFromEnv({ CODEX_CARRY_REASONING: '' }).carryReasoning, true)
  })

  test('reads effort and fallback model from the environment', () => {
    const cfg = configFromEnv({ CODEX_EFFORT: 'high', CODEX_FALLBACK_MODEL: 'gpt-4.1-nano' })
    assert.equal(cfg.effort, 'high')
    assert.equal(cfg.fallbackModel, 'gpt-4.1-nano')
  })
})

describe('textOf', () => {
  test('accepts both content shapes and ignores non-text blocks', () => {
    assert.equal(textOf('plain'), 'plain')
    assert.equal(textOf([{ type: 'text', text: 'a' }, { type: 'text', text: 'b' }]), 'a\nb')
    assert.equal(textOf([{ type: 'image' }, { type: 'text', text: 'only' }]), 'only')
    assert.equal(textOf(undefined), '')
    assert.equal(textOf({ type: 'text', text: 'not an array' }), '')
  })
})

describe('system -> instructions', () => {
  test('a string system prompt becomes instructions', () => {
    const req = buildUpstream({ model: 'codex-sol', system: 'Be terse.', messages: [] }, CFG)
    assert.equal(req.instructions, 'Be terse.')
  })

  test('the client sends system as text blocks; they join with newlines', () => {
    const req = buildUpstream({
      model: 'codex-sol',
      system: [
        { type: 'text', text: 'You are a CLI assistant.' },
        { type: 'text', text: 'Never guess file contents.' },
      ],
      messages: [],
    }, CFG)
    assert.equal(req.instructions, 'You are a CLI assistant.\nNever guess file contents.')
  })

  test('no system prompt leaves instructions unset', () => {
    const req = buildUpstream({ model: 'codex-sol', messages: [] }, CFG)
    assert.equal('instructions' in req, false)
  })
})

describe('messages -> input items', () => {
  test('a string user message becomes one input_text part', () => {
    assert.deepEqual(
      toResponsesInput([{ role: 'user', content: 'hello' }], CFG),
      [{ role: 'user', content: [{ type: 'input_text', text: 'hello' }] }],
    )
  })

  test('a base64 image becomes an input_image data URL', () => {
    const input = toResponsesInput([{
      role: 'user',
      content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'aGVsbG8=' } }],
    }], CFG)
    assert.deepEqual(input, [{
      role: 'user',
      content: [{ type: 'input_image', image_url: 'data:image/png;base64,aGVsbG8=' }],
    }])
  })

  test('a URL image is dropped rather than sent as a broken part', () => {
    const input = toResponsesInput([{
      role: 'user',
      content: [{ type: 'image', source: { type: 'url', url: 'https://example.invalid/a.png' } }],
    }], CFG)
    assert.deepEqual(input, [])
  })

  test('an assistant turn with text and a tool call splits into two items in order', () => {
    const input = toResponsesInput([{
      role: 'assistant',
      content: [
        { type: 'text', text: 'Let me look.' },
        { type: 'tool_use', id: 'toolu_fixture_0001', name: 'Read', input: { file_path: 'README.md' } },
      ],
    }], CFG)
    assert.deepEqual(input, [
      { role: 'assistant', content: [{ type: 'output_text', text: 'Let me look.' }] },
      {
        type: 'function_call',
        call_id: 'toolu_fixture_0001',
        name: 'Read',
        arguments: '{"file_path":"README.md"}',
      },
    ])
  })

  test('a tool_use with no input serialises to an empty object, not undefined', () => {
    const input = toResponsesInput([{
      role: 'assistant',
      content: [{ type: 'tool_use', id: 'toolu_fixture_0002', name: 'ListDir' }],
    }], CFG)
    assert.equal(input[0].arguments, '{}')
  })

  test('empty assistant text emits no message item', () => {
    const input = toResponsesInput([{ role: 'assistant', content: [{ type: 'text', text: '' }] }], CFG)
    assert.deepEqual(input, [])
  })

  test('a tool_result leaves the user message and becomes a standalone item', () => {
    // Anthropic nests tool results inside a user turn; Responses wants them
    // as top-level function_call_output items. This is the shape that breaks
    // the round trip if it regresses.
    const input = toResponsesInput([{
      role: 'user',
      content: [
        { type: 'tool_result', tool_use_id: 'toolu_fixture_0001', content: '# codex-proxy' },
        { type: 'text', text: 'and now summarise it' },
      ],
    }], CFG)
    assert.deepEqual(input, [
      { type: 'function_call_output', call_id: 'toolu_fixture_0001', output: '# codex-proxy' },
      { role: 'user', content: [{ type: 'input_text', text: 'and now summarise it' }] },
    ])
  })

  test('a tool_result carrying content blocks is flattened to text', () => {
    const input = toResponsesInput([{
      role: 'user',
      content: [{
        type: 'tool_result',
        tool_use_id: 'toolu_fixture_0003',
        content: [{ type: 'text', text: 'line one' }, { type: 'text', text: 'line two' }],
      }],
    }], CFG)
    assert.equal(input[0].output, 'line one\nline two')
  })

  test('an empty error tool_result still sends a non-empty output', () => {
    // The Responses API rejects a function_call_output with an empty string,
    // so an errored tool with no body has to say something.
    const input = toResponsesInput([{
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: 'toolu_fixture_0004', content: '', is_error: true }],
    }], CFG)
    assert.equal(input[0].output, 'error')
  })

  test('a full tool round trip keeps call ids paired across turns', () => {
    const input = toResponsesInput([
      { role: 'user', content: 'read the readme' },
      {
        role: 'assistant',
        content: [{ type: 'tool_use', id: 'toolu_fixture_0005', name: 'Read', input: { file_path: 'README.md' } }],
      },
      {
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'toolu_fixture_0005', content: '# codex-proxy' }],
      },
    ], CFG)
    assert.deepEqual(input.map((i: any) => i.type ?? i.role), ['user', 'function_call', 'function_call_output'])
    assert.equal(input[1].call_id, input[2].call_id)
  })
})

describe('reasoning carried through a thinking block', () => {
  const item = { id: 'rs_fixture_0001', type: 'reasoning', encrypted_content: 'ZW5jcnlwdGVkLWJsb2I=' }

  test('encode then decode reproduces the upstream reasoning item', () => {
    const block = encodeReasoning(item, CFG)
    assert.equal(block?.type, 'thinking')
    assert.equal(block?.thinking, REASONING_MARK)
    assert.deepEqual(decodeReasoning(block, CFG), {
      type: 'reasoning',
      id: 'rs_fixture_0001',
      encrypted_content: 'ZW5jcnlwdGVkLWJsb2I=',
      summary: [],
    })
  })

  test('the marker is invisible: a zero-width space, then a tag', () => {
    assert.equal(REASONING_MARK.codePointAt(0), 0x200b)
    assert.equal(REASONING_MARK.slice(1), '<codex-reasoning/>')
  })

  test('a reasoning item without an encrypted blob encodes to nothing', () => {
    assert.equal(encodeReasoning({ id: 'rs_fixture_0002', type: 'reasoning' }, CFG), null)
  })

  test('a real thinking block from another provider is not mistaken for ours', () => {
    assert.equal(decodeReasoning({ type: 'thinking', thinking: 'Let me think...', signature: 'abc' }, CFG), null)
    assert.equal(decodeReasoning({ type: 'text', text: 'hi' }, CFG), null)
  })

  test('a corrupt signature decodes to nothing instead of throwing', () => {
    assert.equal(decodeReasoning({ type: 'thinking', thinking: REASONING_MARK, signature: 'not json' }, CFG), null)
  })

  test('the rebuilt reasoning item is hoisted ahead of the call it produced', () => {
    // Responses rejects the pair in the other order.
    const input = toResponsesInput([{
      role: 'assistant',
      content: [
        encodeReasoning(item, CFG),
        { type: 'text', text: 'Looking.' },
        { type: 'tool_use', id: 'toolu_fixture_0006', name: 'Read', input: {} },
      ],
    }], CFG)
    assert.deepEqual(input.map((i: any) => i.type ?? i.role), ['reasoning', 'assistant', 'function_call'])
  })

  test('with carrying off the marker block is dropped, not sent as text', () => {
    assert.equal(encodeReasoning(item, NO_CARRY), null)
    const input = toResponsesInput([{ role: 'assistant', content: [encodeReasoning(item, CFG)] }], NO_CARRY)
    assert.deepEqual(input, [])
  })
})

describe('tool definitions', () => {
  const tools = [
    {
      name: 'Read',
      description: 'Read a file',
      input_schema: { type: 'object', properties: { file_path: { type: 'string' } }, required: ['file_path'] },
    },
  ]

  test('an Anthropic tool becomes a non-strict Responses function tool', () => {
    assert.deepEqual(toResponsesTools(tools), [{
      type: 'function',
      name: 'Read',
      description: 'Read a file',
      parameters: tools[0].input_schema,
      // The client's schemas omit additionalProperties:false, so strict mode
      // would reject every one of them.
      strict: false,
    }])
  })

  test('a missing description becomes an empty string, never undefined', () => {
    assert.equal(toResponsesTools([{ name: 'Bare', input_schema: { type: 'object' } }])?.[0].description, '')
  })

  test('an MCP-style tool nesting its schema under custom is still mapped', () => {
    const out = toResponsesTools([{ name: 'mcp__srv__do', custom: { input_schema: { type: 'object' } } }])
    assert.equal(out?.[0].name, 'mcp__srv__do')
    assert.deepEqual(out?.[0].parameters, { type: 'object' })
  })

  test('a tool with no schema at all is filtered out', () => {
    const out = toResponsesTools([{ name: 'NoSchema' }, ...tools])
    assert.equal(out?.length, 1)
    assert.equal(out?.[0].name, 'Read')
  })

  test('nothing left after filtering means no tools key upstream', () => {
    assert.equal(toResponsesTools([{ name: 'NoSchema' }]), undefined)
    assert.equal(toResponsesTools([]), undefined)
    assert.equal(toResponsesTools(undefined), undefined)
  })
})

describe('tool_choice', () => {
  test('maps every Anthropic form to its Responses equivalent', () => {
    assert.equal(toToolChoice({ type: 'auto' }), 'auto')
    assert.equal(toToolChoice({ type: 'any' }), 'required')
    assert.equal(toToolChoice({ type: 'none' }), 'none')
    assert.deepEqual(toToolChoice({ type: 'tool', name: 'Read' }), { type: 'function', name: 'Read' })
  })

  test('absent or unknown forms send nothing rather than a guess', () => {
    assert.equal(toToolChoice(undefined), undefined)
    assert.equal(toToolChoice({ type: 'something_new' }), undefined)
  })
})

describe('buildUpstream envelope', () => {
  test('never stores the conversation upstream', () => {
    assert.equal(buildUpstream({ model: 'codex-sol', messages: [] }, CFG).store, false)
  })

  test('stream is always a boolean, mirroring the request', () => {
    assert.equal(buildUpstream({ model: 'codex-sol', messages: [] }, CFG).stream, false)
    assert.equal(buildUpstream({ model: 'codex-sol', messages: [], stream: true }, CFG).stream, true)
  })

  test('max_tokens becomes max_output_tokens with a floor of 16', () => {
    assert.equal(buildUpstream({ model: 'codex-sol', messages: [], max_tokens: 4096 }, CFG).max_output_tokens, 4096)
    // The client probes with max_tokens:1; Responses rejects anything under 16.
    assert.equal(buildUpstream({ model: 'codex-sol', messages: [], max_tokens: 1 }, CFG).max_output_tokens, 16)
    assert.equal('max_output_tokens' in buildUpstream({ model: 'codex-sol', messages: [] }, CFG), false)
  })

  test('the reasoning knob is sent only to models that accept it', () => {
    const reasoning = buildUpstream({ model: 'codex-sol', messages: [] }, CFG)
    assert.deepEqual(reasoning.reasoning, { effort: 'medium' })
    assert.deepEqual(reasoning.include, ['reasoning.encrypted_content'])

    // The cheap fallback 400s on reasoning.effort.
    const fallback = buildUpstream({ model: 'claude-haiku-4-5-20251001', messages: [] }, CFG)
    assert.equal('reasoning' in fallback, false)
    assert.equal('include' in fallback, false)
  })

  test('the configured effort reaches the request', () => {
    assert.deepEqual(
      buildUpstream({ model: 'codex-sol', messages: [] }, { ...CFG, effort: 'high' }).reasoning,
      { effort: 'high' },
    )
  })

  test('with carrying off the encrypted blob is not requested back', () => {
    const req = buildUpstream({ model: 'codex-sol', messages: [] }, NO_CARRY)
    assert.deepEqual(req.reasoning, { effort: 'medium' })
    assert.equal('include' in req, false)
  })

  test('temperature and other unmapped knobs are not forwarded blind', () => {
    // Responses rejects temperature on gpt-5.6-*, so dropping it is the point.
    const req = buildUpstream({ model: 'codex-sol', messages: [], temperature: 0.7, top_p: 0.9, metadata: {} }, CFG)
    assert.equal('temperature' in req, false)
    assert.equal('top_p' in req, false)
    assert.equal('metadata' in req, false)
  })

  test('an empty body still produces a well-formed upstream request', () => {
    const req = buildUpstream({}, CFG)
    assert.equal(req.model, 'gpt-5.4-mini')
    assert.deepEqual(req.input, [])
    assert.equal(req.stream, false)
  })
})
