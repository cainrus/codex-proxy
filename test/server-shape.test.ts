/**
 * The thin HTTP layer: routing decisions and the module's import contract.
 * The routes exercised here answer locally, so no socket and no upstream call
 * is involved -- the response object is a recorder.
 */
import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { describe, test } from 'node:test'
import { promisify } from 'node:util'
import type { ServerResponse } from 'node:http'

import { createProxyServer, handleRequest, newMessageId } from '../codex-proxy.ts'

const execFileAsync = promisify(execFile)

interface Recorded {
  status: number
  headers: Record<string, string>
  body: string
}

function recorder(): { rec: Recorded; res: ServerResponse } {
  const rec: Recorded = { status: 0, headers: {}, body: '' }
  const res = {
    headersSent: false,
    writeHead(status: number, headers: Record<string, string>) {
      rec.status = status
      rec.headers = headers
      this.headersSent = true
      return this
    },
    write(chunk: string) { rec.body += chunk; return true },
    end(chunk?: string) { if (chunk) rec.body += chunk; return this },
  }
  return { rec, res: res as unknown as ServerResponse }
}

describe('routing', () => {
  test('/api/hello answers 200 so the client does not report a logged-out state', async () => {
    // A 404 here surfaces to the user as "Not logged in - Please run /login".
    const { rec, res } = recorder()
    await handleRequest('/api/hello', '', res)
    assert.equal(rec.status, 200)
    assert.equal(rec.headers['Content-Type'], 'application/json')
    assert.deepEqual(JSON.parse(rec.body), {})
  })

  test('/api/hello matches with a query string attached', async () => {
    const { rec, res } = recorder()
    await handleRequest('/api/hello?client=cli', '', res)
    assert.equal(rec.status, 200)
  })

  test('count_tokens answers locally with a length-based estimate', async () => {
    const body = JSON.stringify({ messages: [{ role: 'user', content: 'hello' }] })
    const { rec, res } = recorder()
    await handleRequest('/v1/messages/count_tokens', body, res)
    assert.equal(rec.status, 200)
    assert.deepEqual(JSON.parse(rec.body), { input_tokens: Math.ceil(body.length / 4) })
  })

  test('count_tokens is matched before the /v1/messages prefix it starts with', async () => {
    // Falling through to /v1/messages would try to reach the upstream API.
    const { rec, res } = recorder()
    await handleRequest('/v1/messages/count_tokens', '', res)
    assert.equal(rec.status, 200)
    assert.deepEqual(JSON.parse(rec.body), { input_tokens: 0 })
  })

  test('an unknown path is a JSON 404, not an HTML error page', async () => {
    const { rec, res } = recorder()
    await handleRequest('/v1/complete', '', res)
    assert.equal(rec.status, 404)
    assert.equal(rec.headers['Content-Type'], 'application/json')
    assert.deepEqual(JSON.parse(rec.body), { error: 'not found' })
  })
})

describe('message ids', () => {
  test('carry the msg_ prefix the client expects', () => {
    assert.match(newMessageId(), /^msg_\d+$/)
  })
})

describe('module contract', () => {
  test('the server is built on demand, not at import time', () => {
    const server = createProxyServer()
    assert.equal(server.listening, false)
    server.close()
  })

  test('importing the bridge without OPENAI_API_KEY neither exits nor binds a port', async () => {
    // The CLI still refuses to start without a key; only the import is quiet.
    const env = { ...process.env }
    delete env.OPENAI_API_KEY
    delete env.PORT

    const { stdout, stderr } = await execFileAsync(
      process.execPath,
      ['--experimental-strip-types', '--disable-warning=ExperimentalWarning', 'test/fixtures/import-probe.ts'],
      { env, cwd: new URL('..', import.meta.url) },
    )
    assert.match(stdout, /imported-without-side-effects gpt-5\.6-sol/)
    assert.doesNotMatch(stderr, /ready port=/)
    assert.doesNotMatch(stderr, /FATAL/)
  })
})
