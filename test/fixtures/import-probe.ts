/**
 * Probe for the "importing the bridge starts nothing" contract.
 *
 * Run as its own process with OPENAI_API_KEY unset: the module must import
 * cleanly, without exiting on the missing key and without binding a port.
 */
import { mapModel } from '../../codex-proxy.ts'

console.log(`imported-without-side-effects ${mapModel('codex-sol')}`)
