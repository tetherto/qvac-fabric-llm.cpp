'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const os = require('os')
const path = require('path')
const {
  CLAP_POLICY_VERSION,
  buildClapText,
  mergeClapScore,
  parseClapBatchOutput,
  shouldScoreRound
} = require('../lib/clap')
const { aggregateRounds, groupByPrompt } = require('../lib/aggregate')
const { applyScores, collectPendingRounds, persistRounds, replaceScoredRounds } = require('../score-clap')
const { loadRoundRecords, writeReports } = require('../lib/results')

test('comparison policy matches the shared Python scorer', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '../../../../scripts/benchmarks/music_alignment.py'), 'utf8')
  assert.equal(source.match(/^POLICY_VERSION = '([^']+)'/m)[1], CLAP_POLICY_VERSION)
})

test('buildClapText uses caption or caption plus lyrics', () => {
  const prompt = { caption: 'folk guitar', lyrics: '[Verse]\nHello' }
  assert.equal(buildClapText(prompt, 'caption'), 'folk guitar')
  assert.equal(buildClapText(prompt, 'caption+lyrics'), 'folk guitar\n[Verse]\nHello')
  assert.throws(() => buildClapText(prompt, 'tokens'), /unknown CLAP text policy/)
})

test('parseClapBatchOutput requires ok scores JSON', () => {
  const parsed = parseClapBatchOutput('{"ok":true,"scores":[{"id":"a","ok":true,"score":0.4}]}')
  assert.equal(parsed.scores[0].score, 0.4)
  assert.throws(() => parseClapBatchOutput(''), /no JSON/)
  assert.throws(() => parseClapBatchOutput('{"ok":false,"error":"missing torch"}'), /missing torch/)
})

test('shouldScoreRound only reuses finite scores from the current policy', () => {
  const timed = { kind: 'timed', wavPath: '/tmp/a.wav' }
  assert.equal(shouldScoreRound(timed, { timedOnly: true, force: false }), true)
  assert.equal(shouldScoreRound({ kind: 'warmup', wavPath: '/tmp/a.wav' }, { timedOnly: true, force: false }), false)
  for (const policyVersion of [undefined, null, 'clap-music-v1', 'clap-music-v3']) {
    assert.equal(shouldScoreRound({ ...timed, clap: { score: 0.2, policyVersion } }, {}), true)
  }
  const current = { ...timed, clap: { score: 0.2, policyVersion: CLAP_POLICY_VERSION } }
  assert.equal(shouldScoreRound(current, {}), false)
  assert.equal(shouldScoreRound(current, { force: true }), true)
  for (const score of [null, NaN, Infinity, -Infinity, '0.2']) {
    assert.equal(shouldScoreRound({ ...current, clap: { ...current.clap, score } }, {}), true)
  }
  assert.equal(shouldScoreRound({ ...current, clap: { ...current.clap, ok: false } }, {}), true)
  assert.equal(shouldScoreRound({ kind: 'timed' }, { force: true }), false)
  assert.equal(shouldScoreRound({ kind: 'warmup', wavPath: '/tmp/a.wav' }, { timedOnly: false }), true)
})

test('mergeClapScore keeps generation times untouched', () => {
  const merged = mergeClapScore(
    { engine: 'qvac', generationMs: 1000, e2eMs: 1100 },
    { ok: true, score: 0.31, elapsedMs: 50 },
    { model: 'laion/larger_clap_music_and_speech', revision: 'abc', samplingRate: 48000, policyVersion: CLAP_POLICY_VERSION, device: 'cpu', textPolicy: 'caption' }
  )
  assert.equal(merged.generationMs, 1000)
  assert.equal(merged.e2eMs, 1100)
  assert.equal(merged.clap.score, 0.31)
  assert.equal(merged.clap.elapsedMs, 50)
  assert.equal(merged.clap.policyVersion, CLAP_POLICY_VERSION)
})

test('aggregateRounds summarises CLAP separately from RTF', () => {
  const stats = aggregateRounds([
    { engine: 'qvac', ok: true, generationMs: 10, e2eMs: 12, rtf: 0.5, clap: { score: 0.2 }, audio: { durationSeconds: 8, silenceRatio: 0, clippingRatio: 0 } },
    { engine: 'qvac', ok: true, generationMs: 14, e2eMs: 16, rtf: 0.6, clap: { score: 0.4 }, audio: { durationSeconds: 8, silenceRatio: 0, clippingRatio: 0 } }
  ]).qvac
  assert.ok(Math.abs(stats.clap.median - 0.3) < 1e-12)
  assert.ok(Math.abs(stats.rtf.median - 0.55) < 1e-12)
  assert.equal(stats.clap.policyVersion, null)
})

test('aggregates reject mixed policies across engines and prompts', () => {
  const current = { engine: 'qvac', promptId: 'a', ok: true, clap: { score: 0.2, policyVersion: CLAP_POLICY_VERSION } }
  for (const engine of ['qvac', 'acestep']) {
    for (const policyVersion of [undefined, null, 'clap-music-v1']) {
      const rounds = [current, { ...current, engine, promptId: 'b', clap: { score: 0.4, policyVersion } }]
      assert.throws(() => aggregateRounds(rounds), /mixed CLAP policies.*score-clap.js --force/)
      assert.throws(() => groupByPrompt(rounds), /mixed CLAP policies/)
    }
  }
  const stats = aggregateRounds([current, { ...current, clap: { score: null } },
    { ...current, clap: { score: Infinity } }, { ...current, ok: false, clap: { score: 0.4 } }]).qvac
  assert.equal(stats.clap.n, 1)
  assert.equal(stats.clap.policyVersion, CLAP_POLICY_VERSION)
})

test('applyScores requires the expected batch policy and versions failed items', () => {
  const pending = [{ record: {}, item: { id: 'a' } }]
  for (const policyVersion of [undefined, 'clap-music-v1']) {
    assert.throws(() => applyScores(pending, { scores: [], policyVersion }, 'caption'), /incompatible CLAP scorer policy/)
  }
  const [round] = applyScores(pending, { scores: [], policyVersion: CLAP_POLICY_VERSION }, 'caption')
  assert.equal(round.clap.policyVersion, CLAP_POLICY_VERSION)
  assert.equal(round.clap.score, null)
  assert.match(round.clap.error, /omitted this item/)
})

test('legacy rescore persists the batch policy and produces compatible reports', t => {
  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'clap-policy-'))
  t.after(() => fs.rmSync(outDir, { recursive: true, force: true }))
  const config = { outDir, backend: 'cpu', models: {} }
  const root = path.join(outDir, 'rounds', 'cpu')
  fs.mkdirSync(root, { recursive: true })
  const legacy = { engine: 'qvac', promptId: 'a', kind: 'timed', index: 0, ok: true,
    wavPath: '/tmp/a.wav', generationMs: 100, clap: { score: 0.1 } }
  const current = { ...legacy, index: 1, clap: { score: 0.4, policyVersion: CLAP_POLICY_VERSION } }
  for (const round of [legacy, current]) fs.writeFileSync(path.join(root, `${round.index}.json`), JSON.stringify(round))
  const records = loadRoundRecords(root)
  assert.throws(() => writeReports(config, records), /mixed CLAP policies/)
  assert.equal(fs.existsSync(path.join(outDir, 'cpu.json')), false)
  const pending = collectPendingRounds(records, [{ id: 'a', caption: 'folk guitar' }], {}, 'caption')
  assert.equal(pending.length, 1)
  const scored = applyScores(pending, { policyVersion: CLAP_POLICY_VERSION,
    scores: [{ id: pending[0].item.id, ok: true, score: 0.3 }] }, 'caption')
  persistRounds(config, replaceScoredRounds(records, scored))
  const reloaded = loadRoundRecords(root)
  assert.equal(reloaded.find(round => round.index === 0).generationMs, 100)
  assert.ok(reloaded.every(round => round.clap.policyVersion === CLAP_POLICY_VERSION))
  assert.equal(collectPendingRounds(reloaded, [], {}, 'caption').length, 0)
  const written = writeReports(config, reloaded)
  const report = JSON.parse(fs.readFileSync(written.jsonPath, 'utf8'))
  assert.equal(report.overall.qvac.clap.policyVersion, CLAP_POLICY_VERSION)
  assert.equal(report.byPrompt.a.qvac.clap.policyVersion, CLAP_POLICY_VERSION)
  assert.match(fs.readFileSync(written.markdownPath, 'utf8'), /clap-music-v2/)
})
