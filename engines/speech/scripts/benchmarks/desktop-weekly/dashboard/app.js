(function () {
  const TABS = ['overview', 'asr', 'tts', 'music', 'runs'];
  const QUALITY_FLAG_PCT = 10;
  const DURATION_TOLERANCE = 0.02;
  const ENGINE_COLORS = {
    'qvac-fabric-speech.cpp': '#2f6fdb',
    'QVAC bundled whisper.cpp': '#2f6fdb',
    'audio.cpp': '#e0663a',
    'LocalAI / parakeet.cpp': '#2aa5c9',
    'transcribe.cpp': '#c2a020',
    'NeMo-Speech.cpp': '#4a5d70',
    'whisper.cpp (upstream)': '#c93d3d',
    'cosyvoice.cpp': '#a8733a',
    'TTS.cpp': '#4f9a3c',
    'acestep.cpp': '#1c9e8a',
    'minimaxmusic.cpp': '#b04fc7',
    'faster-whisper (CTranslate2)': '#8b5cf6',
    'vLLM': '#e09a1f',
    'SGLang-Omni': '#d14b8f',
    'OpenMOSS PyTorch reference': '#7d8f1a',
  };
  const ENGINE_ORDER = Object.keys(ENGINE_COLORS);
  const MACHINE_ORDER = ['DGX Spark', 'RTX 5090', 'Strix Halo', 'Jetson Orin', 'Mac Studio M3 Ultra', 'Mac mini M4'];
  const BACKEND_ORDER = ['cpu', 'vulkan', 'cuda', 'metal', 'coreml'];
  const BACKEND_COLORS = { cpu: '#4f52c9', cuda: '#4d8a12', vulkan: '#b9303b', metal: '#c45a22', coreml: '#2aa5c9' };
  const TASK_ORDER = ['ASR', 'TTS', 'TTS cloning', 'Music'];
  const FIELD = { backend: 'backend', model: 'model', scenario: 'scenario_label' };
  const SCEN_SHORT = { jfk: 'JFK 11 s', abcba: 'ABCBA 160.6 s', long_clip: '98 s clip', fixed_prompt: 'fixed prompt', fixed_prompt_short: '9-word prompt', long_prompt: '76-word prompt', clone_jfk: 'clone, JFK ref', clone_bwav: 'clone, b.wav ref', music_30s: '30 s request', music_10s: '10 s request', music_set: 'campaign prompt set' };
  const SCOPE_SHORT = { campaign: 'campaign', fresh_process: 'fresh process', fresh_process_ref: 'fresh process + ref encode', native_excl_load: 'native, excl. load', resident_api: 'resident API', http: 'HTTP, full response' };
  const SET_LABEL = { weekly: 'weekly run', report: 'report', campaign: 'campaign' };
  const TREND_HEIGHT = 220;
  const TREND_MIN_WIDTH = 280;
  const COV_CLASS = { measured: 'ok', failed: 'bad', superseded: 'mute', 'unsupported by competitor': 'mute', 'budget-excluded': 'warn', 'not attempted': 'mute', inaccessible: 'warn', excluded: 'mute', 'install failed': 'bad', 'configuration failed': 'bad' };

  const num = (v) => (v === null || v === undefined || v === '' || Number.isNaN(Number(v)) ? null : Number(v));
  const yes = (v) => v === true || v === 'true' || v === 1 || v === '1';
  const fmtInt = d3.format(',');
  const fmtMs = (v) => (v == null ? 'n/a' : v < 10 ? d3.format('.2f')(v) + ' ms' : v < 1000 ? d3.format('.1f')(v) + ' ms' : v < 100000 ? d3.format('.2f')(v / 1000) + ' s' : d3.format('.1f')(v / 1000) + ' s');
  const fmtX = (v) => (v == null ? 'n/a' : (v >= 100 ? fmtInt(Math.round(v)) : v >= 10 ? d3.format('.1f')(v) : d3.format('.2f')(v)) + '×');
  const fmtRtf = (v) => (v == null ? 'n/a' : v < 0.001 ? d3.format('.5f')(v) : v < 0.01 ? d3.format('.4f')(v) : d3.format('.3f')(v));
  const fmtMib = (v) => (v == null ? 'n/a' : fmtInt(Math.round(v)) + ' MiB');
  const fmtPct = (v) => (v == null ? 'n/a' : d3.format('.1f')(v) + '%');
  const fmtS = (v) => (v == null ? 'n/a' : d3.format('.2f')(v) + ' s');
  const fmtRatio = (v) => (v == null || !isFinite(v) ? 'n/a' : d3.format('.2f')(v));

  const METRICS = {
    latency: { field: 'latency_ms', label: 'Latency (median)', fmt: fmtMs, log: true, better: 'lower is better' },
    speed: { field: 'speed_x', label: 'Speed × real time', fmt: fmtX, log: true, better: 'higher is better' },
    rtf: { field: 'rtf', label: 'RTF', fmt: fmtRtf, log: true, better: 'lower is better' },
    rss: { field: 'rss_mib', label: 'Peak host RSS', fmt: fmtMib, log: false, better: 'lower is better' },
    gpu: { field: 'gpu_mib', label: 'Device memory', fmt: fmtMib, log: false, better: 'lower is better' },
    wer: { field: 'wer_pct', label: 'WER', fmt: fmtPct, log: false, better: 'lower is better' },
  };

  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null || v === false) continue;
        if (k === 'text') el.textContent = v;
        else if (k === 'class') el.className = v;
        else if (k === 'style') Object.assign(el.style, v);
        else el.setAttribute(k, v === true ? '' : v);
      }
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  const rowsOf = (id) => {
    const d = dash.data(id);
    if (d.status !== 'ok') return null;
    return Array.isArray(d.data) ? d.data : d.data && d.data.rows ? d.data.rows : [];
  };
  const dash0 = (el) => el.replaceChildren(document.createTextNode('—'));
  const taskOf = (el) => el.dataset.task || (el.closest('[data-task]') || {}).dataset?.task || 'all';
  const curTab = () => (TABS.includes(dash.params().tab) ? dash.params().tab : 'overview');
  const metricKey = () => (METRICS[dash.params().metric] ? dash.params().metric : 'speed');
  const color = (engine) => ENGINE_COLORS[engine] || 'var(--color-fg-muted)';
  const engineRank = (e) => { const i = ENGINE_ORDER.indexOf(e); return i < 0 ? 99 : i; };
  const machineRank = (m) => { const i = MACHINE_ORDER.indexOf(m); return i < 0 ? 99 : i; };
  const backendRank = (b) => { const i = BACKEND_ORDER.indexOf(b); return i < 0 ? 99 : i; };
  const dot = (engine) => h('span', { class: 'dot', style: { background: color(engine) }, 'aria-hidden': 'true' });

  function matches(r, P, skip) {
    for (const f of Object.keys(FIELD)) {
      if (f !== skip && P[f] && r[FIELD[f]] !== P[f]) return false;
    }
    return true;
  }
  function taskRows(all, task) {
    const P = dash.params();
    return all.filter((r) => r.dashboard === task && matches(r, P, null));
  }
  function metricValue(r, key) {
    if (!yes(r.accepted)) return null;
    if (key === 'wer' && r.task === 'Music') return null;
    return num(r[METRICS[key].field]);
  }
  function naReason(r, key) {
    if (!yes(r.accepted)) return 'Failed run: excluded from comparison bars';
    if (key === 'gpu') return (r.gpu_kind || 'unavailable') + ': ' + (r.gpu_note || '');
    if (key === 'wer') return r.wer_reason || 'Not measured';
    if (key === 'rss') return 'Host RSS not recorded for this row';
    return 'Not recorded for this row';
  }
  function fmtMetric(r, key) {
    const v = metricValue(r, key);
    if (v == null) return yes(r.accepted) ? 'n/a' : 'failed';
    const s = METRICS[key].fmt(v);
    return key === 'gpu' && r.gpu_kind === 'sampled lower bound' ? '≥ ' + s : s;
  }

  // Tooltip shared by bar rows and matrix cells; it is placed inside the drawing's positioned box.
  function tipFor(box) {
    let tip = box.querySelector(':scope > .tip');
    if (!tip) { tip = h('div', { class: 'tip', hidden: true, role: 'tooltip' }); box.append(tip); }
    return tip;
  }
  function showTip(box, target, lines) {
    const tip = tipFor(box);
    tip.replaceChildren(...lines.map((l, i) => h('div', { class: i === 0 ? 'tt' : null, text: l })));
    tip.hidden = false;
    const b = box.getBoundingClientRect();
    const t = target.getBoundingClientRect();
    const w = tip.offsetWidth;
    let left = t.left - b.left + Math.min(t.width, 120);
    left = Math.max(0, Math.min(left, b.width - w));
    tip.style.left = left + 'px';
    tip.style.top = t.bottom - b.top + 4 + 'px';
  }
  function hideTip(box) { const tip = box.querySelector(':scope > .tip'); if (tip) tip.hidden = true; }
  function bindTip(box, target, lines) {
    target.addEventListener('mouseenter', () => showTip(box, target, lines()));
    target.addEventListener('mouseleave', () => hideTip(box));
    target.addEventListener('focus', () => showTip(box, target, lines()));
    target.addEventListener('blur', () => hideTip(box));
  }
  function runLines(r) {
    return [
      r.machine + ' · ' + r.backend + ' · ' + r.scenario_label,
      r.model + ' · ' + r.engine + (r.variant ? ' (' + r.variant + ')' : ''),
      'Precision: ' + r.precision,
      'Timing: ' + r.timing_label,
      'Latency ' + fmtMs(num(r.latency_ms)) + ' · audio ' + fmtS(num(r.audio_s)) + ' · speed ' + fmtX(num(r.speed_x)) + ' · RTF ' + fmtRtf(num(r.rtf)),
      'Host RSS ' + fmtMib(num(r.rss_mib)) + ' · device ' + (num(r.gpu_mib) == null ? r.gpu_kind : fmtMetric(r, 'gpu')) + ' · WER ' + (num(r.wer_pct) == null ? 'n/a' : fmtPct(num(r.wer_pct)) + ' (' + r.wer_kind + ')'),
      (r.runs || 0) + ' timed runs, ' + (r.warmups == null ? 'unknown' : r.warmups) + ' warmup · ' + r.date + ' · ' + r.status,
    ];
  }

  function scaleFor(values, log) {
    const vals = values.filter((v) => v != null && (!log || v > 0));
    if (!vals.length) return { pos: () => 0, text: '' };
    if (log) {
      let lo = Math.floor(Math.log10(d3.min(vals)));
      let hi = Math.ceil(Math.log10(d3.max(vals)));
      if (hi - lo < 1) hi = lo + 1;
      return { pos: (v) => (v > 0 ? Math.max(1.5, ((Math.log10(v) - lo) / (hi - lo)) * 100) : 0), lo: Math.pow(10, lo), hi: Math.pow(10, hi), log: true };
    }
    const max = d3.max(vals) || 1;
    return { pos: (v) => (v > 0 ? Math.max(1, (v / max) * 100) : 0), lo: 0, hi: max, log: false };
  }
  function barRow(box, r, key, scale, showModel) {
    const v = metricValue(r, key);
    const lab = h('div', { class: 'lab' },
      h('div', { class: 'l1' }, dot(r.engine), h('span', { class: 't', text: showModel ? r.model + ' · ' + r.engine : r.engine + (r.variant ? ' (' + r.variant + ')' : '') })),
      h('div', { class: 'l2', text: [SCEN_SHORT[r.scenario] || r.scenario_label, SCOPE_SHORT[r.timing_class] || r.timing_label, showModel && r.variant ? r.variant : null, showModel ? null : r.precision].filter(Boolean).join(' · ') }));
    const track = h('div', { class: 'track' });
    if (v != null) {
      track.append(h('div', { class: 'bar' + (key === 'gpu' && r.gpu_kind === 'sampled lower bound' ? ' lb' : ''), style: { width: scale.pos(v) + '%', background: color(r.engine) } }));
    }
    const val = h('div', { class: 'val' + (v == null ? ' na' : ''), text: fmtMetric(r, key) });
    const row = h('div', { class: 'brow', tabindex: '0' }, lab, track, val);
    bindTip(box, row, () => (v == null ? runLines(r).concat(['No bar: ' + naReason(r, key)]) : runLines(r)));
    return row;
  }

  function sourcePanel(list, summaryText) {
    const det = h('details', null, h('summary', { text: summaryText || 'Sources and exact values' }));
    const body = h('div', { class: 'src' });
    det.append(body);
    det.addEventListener('toggle', () => {
      if (det.open && !body.childElementCount) body.append(...list.map(sourceBlock));
    });
    return det;
  }
  function sourceBlock(r) {
    const pairs = [
      ['Run', r.id + ' · ' + r.model + ' · ' + r.engine + ' · ' + r.machine + ' ' + r.backend + ' · ' + r.status],
      ['Source', r.source_file + ' row ' + r.source_row + (r.raw_source ? ' · raw: ' + r.raw_source : '')],
      ['Latency', fmtMs(num(r.latency_ms)) + ' median (' + r.latency_ms + ' ms)' + (num(r.min_ms) != null ? ' · min ' + fmtMs(num(r.min_ms)) + ' · max ' + fmtMs(num(r.max_ms)) : '')],
      ['Samples (ms)', r.samples_ms],
      ['Audio duration', fmtS(num(r.audio_s)) + (r.audio_samples ? ' (per run: ' + r.audio_samples + ' s)' : '')],
      ['Speed / RTF', fmtX(num(r.speed_x)) + ' / ' + fmtRtf(num(r.rtf))],
      ['Model load', num(r.load_ms) == null ? '' : fmtMs(num(r.load_ms)) + ' (separate from the timed scope; not cold-disk)'],
      ['Host RSS', fmtMib(num(r.rss_mib))],
      ['Device memory', (num(r.gpu_mib) == null ? 'unavailable' : fmtMetric(r, 'gpu')) + ' · ' + r.gpu_kind + ' · ' + r.gpu_note],
      ['WER', (num(r.wer_pct) == null ? 'not measured' : fmtPct(num(r.wer_pct))) + ' · ' + r.wer_kind + (r.wer_scored_outputs ? ' · ' + r.wer_scored_outputs + ' output(s) scored' : '') + ' · ' + (r.wer_scope || '') + (r.wer_reason ? ' · ' + r.wer_reason : '')],
      ['Precision', r.precision + ' (' + r.precision_source + ')'],
      ['Timing scope', r.timing_label + (r.timing_includes ? '. ' + r.timing_includes : '')],
      ['Iterations', (r.runs || 0) + ' timed · ' + (r.warmups == null ? 'warmups not recorded' : r.warmups + ' warmup') + (r.threads ? ' · ' + r.threads + ' CPU threads' : '')],
      ['Backend observed', r.observed_backend],
      ['Measured', r.date + (r.started_utc ? ' · ' + r.started_utc + ' to ' + r.finished_utc : '') + ' · ' + r.date_source],
      ['Engine pins', r.pins],
      ['Model revisions', r.model_revisions],
      ['Hashes', r.hashes],
      ['Configuration', r.config],
      ['Command', r.command],
      ['Host conditions', r.host_conditions],
      ['Comparison limit', r.comparison_limitation],
      ['Caveats', r.caveats],
    ].filter((p) => p[1] != null && p[1] !== '');
    return h('dl', null, ...pairs.flatMap(([k, v]) => [h('dt', { text: k }), h('dd', null, ['Command', 'Hashes', 'Engine pins'].includes(k) ? h('code', { text: v }) : v)]));
  }

  // ---------- page-level state: tabs, metric, filters ----------
  const hashTab = location.hash.slice(1);
  if (TABS.includes(hashTab)) dash.setParams({ tab: hashTab });

  document.querySelectorAll('.fsb .tabs button[data-tab]').forEach((b) => b.addEventListener('click', () => {
    dash.setParams({ tab: b.dataset.tab, model: null, scenario: null });
    dash.setLink(b.dataset.tab);
  }));
  document.querySelectorAll('.fsb [data-metric]').forEach((b) => b.addEventListener('click', () => dash.setParams({ metric: b.dataset.metric })));
  document.querySelectorAll('.fsb .reset').forEach((b) => b.addEventListener('click', () => dash.setParams({ backend: null, model: null, scenario: null })));

  let covStatus = '';
  document.querySelectorAll('.fsb [data-cov]').forEach((b) => b.addEventListener('click', () => {
    covStatus = b.dataset.cov;
    document.querySelectorAll('.fsb [data-cov]').forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.cov === covStatus)));
    dash.draw('coverage-table');
  }));
  let runQuery = '';
  let runStatus = '';
  let runSort = { key: 'id', dir: 1 };
  let selectedRun = null;
  document.querySelectorAll('.fsb .runs-q').forEach((i) => i.addEventListener('input', () => { runQuery = i.value.trim().toLowerCase(); dash.draw('runs-table'); }));
  document.querySelectorAll('.fsb .runs-status').forEach((s) => s.addEventListener('change', () => { runStatus = s.value; dash.draw('runs-table'); }));
  document.querySelectorAll('.fsb button.sort').forEach((b) => b.addEventListener('click', () => {
    const k = b.dataset.sort;
    runSort = { key: k, dir: runSort.key === k ? -runSort.dir : 1 };
    document.querySelectorAll('.fsb button.sort').forEach((x) => x.closest('th').setAttribute('aria-sort', x.dataset.sort === k ? (runSort.dir > 0 ? 'ascending' : 'descending') : 'none'));
    dash.draw('runs-table');
  }));
  document.querySelectorAll('.fsb .runs-table').forEach((t) => t.addEventListener('click', (ev) => {
    const tr = ev.target.closest('tr[data-key]');
    if (!tr) return;
    selectedRun = tr.dataset.key;
    dash.draw('runs-table');
    dash.draw('run-detail');
  }));
  let histQuery = '';
  document.querySelectorAll('.fsb .hist-q').forEach((i) => i.addEventListener('input', () => { histQuery = i.value.trim().toLowerCase(); dash.draw('prior-campaign'); }));

  dash.onData(() => {
    const tab = curTab();
    document.querySelectorAll('.fsb section.page').forEach((s) => { s.hidden = s.dataset.page !== tab; });
    document.querySelectorAll('.fsb .tabs button[data-tab]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
    const mk = metricKey();
    document.querySelectorAll('.fsb [data-metric]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.metric === mk)));
    document.querySelectorAll('.fsb [data-cov]').forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.cov === covStatus)));
  });

  dash.calc('ratios', {
    title: 'Competitor ÷ QVAC time',
    description: 'For each matched comparison with exactly one accepted QVAC row, every other accepted row\'s median latency divided by the QVAC median latency (the RTF ratio where a campaign row has no latency).',
    inputs: ['runs'],
    fn: (runs) => {
      const ok = runs.filter((r) => r.accepted === true || r.accepted === 'true');
      const byGroup = {};
      ok.forEach((r) => { (byGroup[r.group] = byGroup[r.group] || []).push(r); });
      const out = [];
      Object.values(byGroup).forEach((g) => {
        const q = g.filter((r) => r.is_qvac === true || r.is_qvac === 'true');
        if (q.length !== 1) return;
        g.forEach((r) => {
          if (r === q[0]) return;
          const timed = Number(r.latency_ms) > 0 && Number(q[0].latency_ms) > 0;
          out.push({ id: r.id, group: r.group, engine: r.engine, qvac_id: q[0].id, ratio: timed ? Number(r.latency_ms) / Number(q[0].latency_ms) : Number(r.rtf) / Number(q[0].rtf), qualifier: r.ratio_qualifier });
        });
      });
      return out;
    },
  });

  // ---------- drawings ----------
  dash.draw('fact', (el) => {
    const m = rowsOf('machine');
    if (!m || !m.length) return dash0(el);
    const v = m[0][el.dataset.key];
    if (v == null || v === '') return dash0(el);
    el.textContent = el.dataset.key === 'snapshot_utc' ? String(v).slice(0, 16).replace('T', ' ') + ' UTC' : typeof v === 'number' ? fmtInt(v) : String(v);
  });

  dash.draw('mstatus', (el) => {
    const m = rowsOf('machine');
    if (!m || !m.length) return dash0(el);
    const x = m[0];
    const backends = String(x.backends || '').split(',').map((s) => s.trim()).filter(Boolean);
    el.replaceChildren(...[
      h('div', { style: { display: 'flex', gap: '6px', flexWrap: 'wrap' } }, ...backends.map((b) => h('span', { class: 'chip', text: b })),
        h('span', { class: 'pill ok', text: fmtInt(num(x.accepted)) + ' accepted' }), num(x.failed) ? h('span', { class: 'pill bad', text: fmtInt(num(x.failed)) + ' failed' }) : null),
      h('p', { class: 'small', text: x.work }),
      h('p', { class: 'small muted', text: 'Host ' + x.host + ' · state at collection: ' + x.status }),
      x.conditions ? h('p', { class: 'note small', text: x.conditions }) : null,
    ].filter(Boolean));
  });

  dash.draw('gaps', (el) => {
    const cov = rowsOf('coverage');
    if (!cov) return dash0(el);
    const list = cov.filter((c) => c.status !== 'measured');
    if (!list.length) return el.replaceChildren(h('p', { class: 'small muted', text: 'No gaps recorded.' }));
    el.replaceChildren(h('ul', { class: 'plain small' }, ...list.map((c) => h('li', null, h('span', { class: 'pill ' + (COV_CLASS[c.status] || 'mute'), text: c.status }), ' ', h('b', { text: c.task + ' · ' + c.model }), ' · ' + c.engine + ' · ' + c.backend + (c.scenario ? ' · ' + c.scenario : '') + '. ' + c.reason))));
  });

  dash.draw('engines-here', (el) => {
    const e = rowsOf('engines');
    const m = rowsOf('machine');
    if (!e || !m || !m.length) return el.replaceChildren();
    const label = m[0].machine;
    const here = (x) => String(x.measured_on || '').split(',').map((s) => s.trim()).includes(label);
    const list = e.filter(here).concat(e.filter((x) => !here(x) && x.role !== 'QVAC engine'));
    el.replaceChildren(...list.map((x) => h('tr', null,
      h('td', null, h('span', { style: { display: 'inline-flex', gap: '6px', alignItems: 'center' } }, ENGINE_COLORS[x.engine] ? dot(x.engine) : null, h('b', { text: x.engine }))),
      h('td', null, h('span', { class: 'pill ' + (here(x) ? 'ok' : 'mute'), text: here(x) ? 'measured here' : x.role })),
      h('td', { text: x.measured_on || 'none' }),
      h('td', { text: x.details }))));
  });

  dash.draw('legend', (el) => {
    const all = rowsOf('runs');
    if (!all) return dash0(el);
    const task = taskOf(el);
    const engines = [...new Set(all.filter((r) => task === 'all' || r.dashboard === task).map((r) => r.engine))].sort((a, b) => engineRank(a) - engineRank(b));
    el.replaceChildren(...engines.map((e) => h('span', null, dot(e), e)));
  });

  dash.draw('notes', (el) => {
    const c = rowsOf('commentary');
    if (!c) return dash0(el);
    const task = taskOf(el);
    const list = c.filter((x) => x.dashboard === task);
    if (!list.length) return el.replaceChildren(h('p', { class: 'small muted', text: 'No commentary recorded for this task.' }));
    el.replaceChildren(h('ul', { class: 'plain small' }, ...list.map((x) => h('li', null, h('b', { text: x.topic + '. ' }), x.comment, x.scope ? h('span', { class: 'muted', text: ' Scope: ' + x.scope }) : null, h('span', { class: 'muted' }, ' Source: ', h('code', { text: x.source }))))));
  });

  dash.draw('qflags', (el) => {
    const all = rowsOf('runs');
    if (!all) return dash0(el);
    const flagged = all.filter((r) => yes(r.accepted) && (r.task === 'TTS' || r.task === 'TTS cloning') && num(r.wer_pct) != null && num(r.wer_pct) >= QUALITY_FLAG_PCT)
      .sort((a, b) => num(b.wer_pct) - num(a.wer_pct));
    const box = h('div', { class: 'rows' });
    const items = flagged.map((r) => {
      const li = h('li', { tabindex: '0' }, dot(r.engine), ' ', h('b', { text: r.model }), ' · ' + r.engine + ' on ' + r.machine + ' ' + r.backend + (r.task === 'TTS cloning' ? ' (cloning)' : '') + ': intelligibility WER ',
        h('b', { text: fmtPct(num(r.wer_pct)) }), ' at ' + fmtX(num(r.speed_x)) + ' speed (' + r.wer_kind + ', ' + (r.wer_scored_outputs || 0) + ' output scored). ');
      bindTip(box, li, () => runLines(r));
      return li;
    });
    if (!items.length) return el.replaceChildren(h('p', { class: 'small muted', text: 'No TTS row reaches the ' + QUALITY_FLAG_PCT + '% intelligibility WER flag threshold here.' }));
    box.append(h('ul', { class: 'plain small' }, ...items));
    el.replaceChildren(box);
  });

  dash.draw('machines', (el) => {
    const m = rowsOf('machines');
    if (!m) return dash0(el);
    const list = m.slice().sort((a, b) => machineRank(a.machine) - machineRank(b.machine));
    el.replaceChildren(...list.map((x) => {
      const backends = String(x.backends || '').split(',').map((s) => s.trim()).filter(Boolean);
      const card = h('div', { class: 'card', id: 'machine-' + x.host },
        h('div', { class: 'mcard-head' },
          h('div', { style: { display: 'flex', gap: '6px', flexWrap: 'wrap' } }, ...(backends.length ? backends : ['no fresh data']).map((b) => h('span', { class: 'chip', text: b }))),
          h('span', { class: 'pill ' + (num(x.accepted) > 0 ? 'ok' : 'warn'), text: num(x.accepted) > 0 ? fmtInt(num(x.accepted)) + ' accepted' + (num(x.failed) ? ', ' + num(x.failed) + ' failed' : '') : x.status })),
        h('h3', { text: x.machine }),
        h('p', { class: 'small muted', text: x.hardware + ' · host ' + x.host }),
        h('p', { class: 'small', text: x.work }),
        x.conditions ? h('p', { class: 'note small', text: x.conditions }) : null);
      if (num(x.accepted) > 0) {
        const nav = h('div', { style: { display: 'flex', gap: '6px', flexWrap: 'wrap' } });
        [['asr', 'ASR'], ['tts', 'TTS'], ['music', 'Music']].forEach(([t, label]) => {
          const b = h('button', { type: 'button', class: 'btn', text: label + ' on ' + x.machine + ' →' });
          b.addEventListener('click', () => { dash.setParams({ tab: t, machine: x.machine, backend: null, model: null, scenario: null }); dash.setLink(t); });
          nav.append(b);
        });
        card.append(nav);
      }
      return card;
    }));
  });

  dash.draw('matrix', (el) => {
    const all = rowsOf('runs');
    const cov = rowsOf('coverage');
    if (!all || !cov) return dash0(el);
    const colKeys = [...new Set(all.map((r) => r.machine + '|' + r.backend))]
      .sort((a, b) => { const [ma, ba] = a.split('|'); const [mb, bb] = b.split('|'); return machineRank(ma) - machineRank(mb) || backendRank(ba) - backendRank(bb); })
      .map((k) => { const [machine, backend] = k.split('|'); return { machine, backend, host: all.find((r) => r.machine === machine).host }; });
    cov.filter((c) => (c.status === 'inaccessible' || c.status === 'excluded') && c.model === 'All models')
      .forEach((c) => colKeys.push({ machine: c.machine, backend: c.backend, host: c.host, whole: c }));
    const rowKeys = [...new Set(all.map((r) => r.task + '|' + r.model))]
      .sort((a, b) => TASK_ORDER.indexOf(a.split('|')[0]) - TASK_ORDER.indexOf(b.split('|')[0]) || a.localeCompare(b));
    const box = h('div', { style: { position: 'relative' } });
    const table = h('table', { class: 'ctable matrix' });
    table.append(h('thead', null, h('tr', null, h('th', { text: 'Task · model' }), ...colKeys.map((c) => h('th', null, c.machine, h('br'), h('span', { class: 'muted', text: c.backend }))))));
    const tbody = h('tbody');
    rowKeys.forEach((rk) => {
      const [task, model] = rk.split('|');
      const req = all.some((r) => r.task === task && r.model === model && yes(r.required_family));
      const tr = h('tr', null, h('td', null, h('span', { class: 'muted small', text: task + ' ' }), h('b', { text: model }), req ? h('span', null, ' ', h('span', { class: 'tag req', text: 'required' })) : null));
      colKeys.forEach((c) => {
        const td = h('td', { class: 'cell' });
        if (c.whole) {
          const p = h('span', { class: 'pill ' + (COV_CLASS[c.whole.status] || 'mute'), tabindex: '0', text: c.whole.status });
          bindTip(box, p, () => [c.machine, c.whole.reason]);
          td.append(p);
          tr.append(td);
          return;
        }
        const here = all.filter((r) => r.task === task && r.model === model && r.machine === c.machine && r.backend === c.backend);
        if (here.length) {
          const dots = h('div', { class: 'dots' });
          const okEng = [...new Set(here.filter((r) => yes(r.accepted)).map((r) => r.engine))].sort((a, b) => engineRank(a) - engineRank(b));
          okEng.forEach((e) => {
            const d = h('span', { class: 'dot', tabindex: '0', style: { background: color(e), width: '11px', height: '11px' }, 'aria-label': e });
            bindTip(box, d, () => [c.machine + ' · ' + c.backend, model + ' · ' + e, here.filter((r) => r.engine === e && yes(r.accepted)).length + ' accepted row(s): ' + [...new Set(here.filter((r) => r.engine === e).map((r) => SCEN_SHORT[r.scenario] + ', ' + SCOPE_SHORT[r.timing_class]))].join('; ')]);
            dots.append(d);
          });
          here.filter((r) => !yes(r.accepted)).forEach((r) => {
            const x = h('span', { class: 'small', tabindex: '0', style: { color: 'var(--color-bad)', fontWeight: '600' }, text: '✕' });
            bindTip(box, x, () => ['Failed: ' + r.engine + (r.variant ? ' (' + r.variant + ')' : ''), r.comparison_limitation || r.caveats || 'Run failed']);
            dots.append(x);
          });
          td.append(dots);
        } else {
          const gap = cov.find((g) => g.host === c.host && g.status !== 'measured' && g.status !== 'failed' &&
            (g.model === model || model.startsWith(String(g.model).split(' ')[0]) || String(g.model).includes(model.split(' ')[0])) &&
            (String(g.backend).includes(c.backend)) && (g.task === task || String(g.task).includes(task.split(' ')[0])));
          if (gap) {
            const p = h('span', { class: 'pill ' + (COV_CLASS[gap.status] || 'mute'), tabindex: '0', text: gap.status });
            bindTip(box, p, () => [c.machine + ' · ' + c.backend + ' · ' + gap.engine, gap.reason]);
            td.append(p);
          }
        }
        tr.append(td);
      });
      tbody.append(tr);
    });
    table.append(tbody);
    box.append(table);
    el.replaceChildren(box);
  });

  dash.draw('metrics-table', (el) => {
    const m = rowsOf('metrics');
    if (!m) return el.replaceChildren();
    el.replaceChildren(...m.map((x) => h('tr', null, h('td', null, h('code', { text: x.key })), h('td', { text: x.definition }), h('td', { text: x.better }),
      h('td', null, h('span', { class: 'pill ' + (x.available === 'yes' ? 'ok' : x.available === 'partial' ? 'warn' : 'mute'), text: x.available })))));
  });

  function engineRows(el, filter, roleCol) {
    const e = rowsOf('engines');
    if (!e) return el.replaceChildren();
    el.replaceChildren(...e.filter(filter).map((x) => h('tr', null,
      h('td', null, h('span', { style: { display: 'inline-flex', gap: '6px', alignItems: 'center' } }, ENGINE_COLORS[x.engine] ? dot(x.engine) : null, h('b', { text: x.engine }))),
      h('td', null, h('span', { class: 'pill ' + (x.status === 'measured' ? 'ok' : 'mute'), text: roleCol ? x.role : x.status })),
      h('td', { text: x.measured_on || 'none' }),
      h('td', null, x.details, h('br'), h('span', { class: 'muted' }, 'Source: ', h('code', { text: x.source }))))));
  }
  dash.draw('engines-prior', (el) => engineRows(el, (x) => x.role === 'prior-report competitor', false));
  dash.draw('engines-new', (el) => engineRows(el, (x) => x.role === 'new competitor, measured', true));

  dash.draw('filter', (sel) => {
    const all = rowsOf('runs');
    if (!all) return sel.replaceChildren();
    const f = sel.dataset.f;
    const task = taskOf(sel);
    const P = dash.params();
    const base = all.filter((r) => r.dashboard === task && matches(r, P, f));
    let vals = [...new Set(base.map((r) => r[FIELD[f]]))];
    vals.sort((a, b) => (f === 'machine' ? machineRank(a) - machineRank(b) : f === 'backend' ? backendRank(a) - backendRank(b) : String(a).localeCompare(String(b))));
    sel.replaceChildren(new Option('All', ''), ...vals.map((v) => new Option(v, v)));
    sel.value = P[f] && vals.includes(P[f]) ? P[f] : '';
    sel.onchange = () => dash.setParams({ [f]: sel.value || null });
  });

  dash.draw('mb-cards', (el) => {
    const all = rowsOf('runs');
    const mach = rowsOf('machine');
    const cov = rowsOf('coverage');
    if (!all || !mach || !cov) return dash0(el);
    const key = metricKey();
    const M = METRICS[key];
    const task = taskOf(el);
    const rs = taskRows(all, task);
    if (!rs.length) {
      const words = { asr: ['ASR'], tts: ['TTS'], music: ['Music'] }[task] || [];
      const reasons = cov.filter((c) => c.status !== 'measured' && (c.task === 'All' || words.some((w) => String(c.task).includes(w))));
      const anyRows = all.some((r) => r.dashboard === task);
      return el.replaceChildren(h('div', { class: 'card' }, h('p', { class: 'small', text: anyRows ? 'No rows match these filters.' : 'Not measured on this machine in this collection.' }),
        ...(anyRows ? [] : reasons.map((c) => h('p', { class: 'small muted' }, h('span', { class: 'pill ' + (COV_CLASS[c.status] || 'mute'), text: c.status }), ' ' + c.model + ' · ' + c.engine + ': ' + c.reason)))));
    }
    const groups = d3.groups(rs, (r) => r.machine + '|' + r.backend)
      .sort((a, b) => { const [ma, ba] = a[0].split('|'); const [mb, bb] = b[0].split('|'); return machineRank(ma) - machineRank(mb) || backendRank(ba) - backendRank(bb); });
    el.replaceChildren(...groups.map(([k, list]) => {
      const [machine, backend] = k.split('|');
      const info = mach.find((x) => x.machine === machine) || {};
      list.sort((a, b) => a.model.localeCompare(b.model) || a.scenario.localeCompare(b.scenario) || a.timing_class.localeCompare(b.timing_class) || engineRank(a.engine) - engineRank(b.engine) || a.id.localeCompare(b.id));
      const scale = scaleFor(list.map((r) => metricValue(r, key)), M.log);
      const box = h('div', { class: 'rows' });
      box.append(...list.map((r) => barRow(box, r, key, scale, true)));
      const ok = list.filter((r) => yes(r.accepted)).length;
      const shared = list.some((r) => r.host_conditions);
      return h('div', { class: 'card', id: 'card-' + info.host + '-' + backend + '-' + taskOf(el) },
        h('div', { class: 'mcard-head' }, h('span', { class: 'chip', text: backend }), h('span', { class: 'small muted', text: ok + ' measured' + (list.length - ok ? ', ' + (list.length - ok) + ' failed' : '') })),
        h('h3', { text: machine }),
        h('p', { class: 'small muted', text: info.hardware || '' }),
        h('div', { class: 'meta' }, h('span', null, h('b', { text: M.label }), ' · ' + M.better),
          h('span', { text: scale.hi ? (scale.log ? 'log scale ' : 'linear scale ') + (scale.log ? M.fmt(scale.lo) + ' to ' + M.fmt(scale.hi) : 'from 0 to ' + M.fmt(scale.hi)) : 'no values for this metric' })),
        shared ? h('p', { class: 'note', text: 'Shared host: another user\'s GPU1 workload was active during these runs. Not an idle-host ranking.' }) : null,
        box,
        sourcePanel(list));
    }));
  });

  function durationNote(ok) {
    const withAudio = ok.filter((r) => num(r.audio_s) != null && num(r.latency_ms) != null);
    if (withAudio.length < 2) return null;
    const durs = withAudio.map((r) => num(r.audio_s));
    if ((d3.max(durs) - d3.min(durs)) / d3.max(durs) <= DURATION_TOLERANCE) return null;
    const fastest = d3.least(withAudio, (r) => num(r.latency_ms));
    const speediest = d3.greatest(withAudio, (r) => num(r.speed_x));
    const parts = withAudio.map((r) => r.engine + (r.variant ? ' (' + r.variant + ')' : '') + ' ' + fmtS(num(r.audio_s)));
    if (fastest.id !== speediest.id) {
      return 'Ordering flips: ' + fastest.engine + ' finishes this request sooner (' + fmtMs(num(fastest.latency_ms)) + '), while ' + speediest.engine + ' produces more audio per processing second (' + fmtX(num(speediest.speed_x)) + '). Generated durations differ: ' + parts.join(', ') + '.';
    }
    return 'Generated durations differ (' + parts.join(', ') + '), so latency and speed ratios are not identical.';
  }

  dash.draw('groups', (el) => {
    const all = rowsOf('runs');
    const ratios = rowsOf('ratios');
    if (!all || !ratios) return dash0(el);
    const key = metricKey();
    const M = METRICS[key];
    const rs = taskRows(all, taskOf(el));
    const groups = d3.groups(rs, (r) => r.group).filter(([, list]) => list.filter((r) => yes(r.accepted)).length >= 2)
      .sort((a, b) => a[1][0].model.localeCompare(b[1][0].model) || machineRank(a[1][0].machine) - machineRank(b[1][0].machine) || backendRank(a[1][0].backend) - backendRank(b[1][0].backend) || a[0].localeCompare(b[0]));
    if (!groups.length) return el.replaceChildren(h('p', { class: 'small muted', text: 'No matched comparison has two or more accepted rows under these filters.' }));
    el.replaceChildren(...groups.map(([gk, list]) => {
      list.sort((a, b) => (yes(b.is_qvac) - yes(a.is_qvac)) || engineRank(a.engine) - engineRank(b.engine) || a.id.localeCompare(b.id));
      const f = list[0];
      const ok = list.filter((r) => yes(r.accepted));
      const scale = scaleFor(list.map((r) => metricValue(r, key)), false);
      const box = h('div', { class: 'rows' });
      box.append(...list.map((r) => barRow(box, r, key, scale, false)));
      const tbl = h('table', { class: 'ctable' },
        h('thead', null, h('tr', null, ...['Engine', 'Latency median (min to max)', 'Audio', 'Speed ×', 'RTF', 'Host RSS', 'Device', 'WER', 'Runs / warmups'].map((t, i) => h('th', { class: i ? 'r' : null, text: t })))),
        h('tbody', null, ...list.map((r) => h('tr', null,
          h('td', null, h('span', { style: { display: 'inline-flex', gap: '6px', alignItems: 'center' } }, dot(r.engine), r.engine + (r.variant ? ' (' + r.variant + ')' : '')), yes(r.accepted) ? null : h('span', null, ' ', h('span', { class: 'pill bad', text: 'failed' }))),
          h('td', { class: 'r', text: yes(r.accepted) ? fmtMs(num(r.latency_ms)) + (num(r.min_ms) != null ? ' (' + fmtMs(num(r.min_ms)) + ' to ' + fmtMs(num(r.max_ms)) + ')' : '') : 'failed' }),
          h('td', { class: 'r', text: fmtS(num(r.audio_s)) }),
          h('td', { class: 'r', text: yes(r.accepted) ? fmtX(num(r.speed_x)) : 'n/a' }),
          h('td', { class: 'r', text: yes(r.accepted) ? fmtRtf(num(r.rtf)) : 'n/a' }),
          h('td', { class: 'r', text: fmtMib(num(r.rss_mib)) }),
          h('td', { class: 'r', text: num(r.gpu_mib) == null ? r.gpu_kind : fmtMetric(r, 'gpu') }),
          h('td', { class: 'r', text: num(r.wer_pct) == null ? 'n/a' : fmtPct(num(r.wer_pct)) }),
          h('td', { class: 'r', text: (r.runs || 0) + ' / ' + (r.warmups == null ? '?' : r.warmups) })))));
      const precisions = list.map((r) => h('li', null, dot(r.engine), ' ' + r.engine + (r.variant ? ' (' + r.variant + ')' : '') + ': ' + r.precision));
      const rr = ratios.filter((x) => x.group === gk);
      const ratioBlock = rr.length ? h('div', { class: 'ratio' },
        h('div', null, h('b', { text: 'Competitor ÷ QVAC time' }), h('span', { class: 'muted', text: ' (median latency; above 1 means QVAC took less time)' })),
        ...rr.map((x) => h('div', null, dot(x.engine), ' ' + x.engine + (list.find((r) => r.id === x.id)?.variant ? ' (' + list.find((r) => r.id === x.id).variant + ')' : '') + ': ', h('b', { 'data-source': 'ratios', 'data-where': 'id=' + x.id, 'data-field': 'ratio', text: fmtRatio(num(x.ratio)) }))),
        h('div', { class: 'muted', text: 'Qualifier: ' + (rr[0].qualifier || '') })) : null;
      const dn = durationNote(ok);
      const limits = [...new Set(list.map((r) => r.comparison_limitation).filter(Boolean))];
      const hostCond = [...new Set(list.map((r) => r.host_conditions).filter(Boolean))];
      const failed = list.filter((r) => !yes(r.accepted));
      const threads = [...new Set(ok.map((r) => r.threads).filter((t) => t != null))];
      return h('div', { class: 'card', id: 'grp-' + f.id },
        h('div', { class: 'mcard-head' }, h('span', { class: 'chip', text: f.backend }), h('span', { class: 'small muted', text: f.machine })),
        h('h3', null, f.model + (f.task === 'TTS cloning' ? ' · voice cloning' : ''), yes(f.required_family) ? h('span', null, ' ', h('span', { class: 'tag req', text: 'required' })) : null),
        h('div', { class: 'meta' },
          h('span', null, h('b', { text: 'Input ' }), f.scenario_label),
          h('span', null, h('b', { text: 'Timing ' }), f.timing_label),
          h('span', null, h('b', { text: 'Metric ' }), M.label + ', ' + M.better),
          threads.length ? h('span', null, h('b', { text: 'CPU threads ' }), threads.join(', ')) : null,
          h('span', null, h('b', { text: 'Measured ' }), [...new Set(list.map((r) => r.date))].join(', '))),
        f.timing_includes ? h('p', { class: 'small muted', text: 'Timed region: ' + f.timing_includes }) : null,
        box,
        h('div', { class: 'scroll' }, tbl),
        h('div', null, h('div', { class: 'small muted', text: 'Precision by row' }), h('ul', { class: 'plain small' }, ...precisions)),
        ratioBlock,
        dn ? h('p', { class: 'note', text: dn }) : null,
        ...hostCond.map((t) => h('p', { class: 'note', text: t })),
        ...limits.map((t) => h('p', { class: 'note info', text: 'Comparison limit: ' + t })),
        failed.length ? h('p', { class: 'note', text: 'Failed rows in this setup are listed without bars: ' + failed.map((r) => r.engine + (r.variant ? ' (' + r.variant + ')' : '')).join(', ') + '.' }) : null,
        sourcePanel(list));
    }));
  });

  // ---------- per-model comparison sections ----------
  const WHAT = { ASR: 'transcription speed', TTS: 'speech synthesis speed', 'TTS cloning': 'voice cloning speed', Music: 'music generation speed' };
  const ONE_LINE = {
    ASR: 'Seconds of audio transcribed per second of processing, higher is better. Each chart is one backend and clip from one source: the weekly run, whose bars are QVAC variants, or a dated report or campaign, whose bars are QVAC and the other engines on the same input and timing scope.',
    TTS: 'Seconds of speech generated per second of processing, higher is better. Each chart is one backend and prompt from one source: the weekly run, whose bars are QVAC variants, or a dated report or campaign, whose bars are QVAC and the other engines on the same prompt and timing scope.',
    'TTS cloning': 'Seconds of cloned speech generated per second of processing, higher is better. Each chart is one backend and reference recording from one source: the weekly run, whose bars are QVAC variants, or a dated report, whose bars are QVAC and the other engines under the same timing scope.',
    Music: 'Seconds of music generated per second of processing, higher is better. Each chart is one backend from one source: the weekly run, a fresh process per render including model load, a dated report with the other engines on the same request, or the dated campaign against acestep.cpp over its ten prompts.',
  };
  const LATE_MODELS = ['Nemotron'];
  const modelRank = (m) => { const i = LATE_MODELS.findIndex((p) => m.startsWith(p)); return i < 0 ? 0 : i + 1; };
  const SCEN_RANK = ['jfk', 'long_clip', 'fixed_prompt', 'fixed_prompt_short', 'long_prompt', 'clone_jfk', 'music_30s', 'music_set'];
  const BACKEND_LABEL = { cpu: 'CPU', cuda: 'CUDA', vulkan: 'Vulkan', metal: 'Metal', coreml: 'Core ML' };
  const label = (r) => r.engine + (r.variant ? ' (' + r.variant + ')' : '');

  function isComparison(list) {
    const ok = list.filter((r) => yes(r.accepted));
    return (ok.length >= 2 && ok.some((r) => !yes(r.is_qvac))) || (ok.length >= 1 && ok.every((r) => r.comparison_set === 'weekly'));
  }
  const isWeekly = (list) => list.every((r) => r.comparison_set === 'weekly');
  function panelTitle(f) {
    return (BACKEND_LABEL[f.backend] || f.backend) + ' · ' + (SCEN_SHORT[f.scenario] || f.scenario_label) + ' · ' + (SET_LABEL[f.comparison_set] || f.comparison_set) + (f.comparison_set === 'weekly' ? '' : ' ' + f.date);
  }
  function speedPanel(list) {
    const weekly = isWeekly(list);
    const f = list.find((r) => yes(r.is_qvac)) || list[0];
    const speed = (r) => (yes(r.accepted) ? num(r.speed_x) : null);
    const rows = list.slice().sort((a, b) => (speed(b) ?? -1) - (speed(a) ?? -1));
    const max = d3.max(rows, speed) || 1;
    const ticks = d3.ticks(0, max, 4);
    const top = Math.max(max, ticks[ticks.length - 1] || max);
    const pos = (v) => (v / top) * 100;
    const box = h('div', { class: 'hb' });
    rows.forEach((r) => {
      const v = speed(r);
      const row = h('div', { class: 'hb-row', tabindex: '0' },
        h('div', { class: 'hb-lab' }, h('span', { text: weekly ? (r.variant || r.engine) : r.engine }), !weekly && r.variant ? h('small', { text: r.variant }) : null),
        h('div', { class: 'hb-track' }, ...ticks.map((t) => h('i', { class: 'hb-grid', style: { left: pos(t) + '%' } })),
          v != null ? h('div', { class: 'hb-bar', style: { width: Math.max(1, pos(v)) + '%', background: color(r.engine) } }) : null),
        h('div', { class: 'hb-val' + (v == null ? ' na' : ''), text: v == null ? (yes(r.accepted) ? 'n/a' : 'failed') : fmtX(v) }));
      bindTip(box, row, () => runLines(r));
      box.append(row);
    });
    box.append(h('div', { class: 'hb-axis' }, h('span'), h('div', { class: 'hb-ticks' }, ...ticks.map((t) => h('span', { style: { left: pos(t) + '%' }, text: d3.format('~g')(t) }))), h('span')));
    return h('div', { class: 'panel' },
      h('h4', { text: panelTitle(f) }),
      h('p', { class: 'small muted', text: SCOPE_SHORT[f.timing_class] || f.timing_label }),
      box);
  }
  function detailTable(list) {
    return h('table', { class: 'ctable' },
      h('thead', null, h('tr', null, ...['Engine', 'Latency median (min to max)', 'Audio', 'Speed ×', 'RTF', 'Host RSS', 'Device', 'WER', 'Runs / warmups'].map((t, i) => h('th', { class: i ? 'r' : null, text: t })))),
      h('tbody', null, ...list.map((r) => h('tr', null,
        h('td', null, h('span', { style: { display: 'inline-flex', gap: '6px', alignItems: 'center' } }, dot(r.engine), label(r)), yes(r.accepted) ? null : h('span', null, ' ', h('span', { class: 'pill bad', text: 'failed' }))),
        h('td', { class: 'r', text: yes(r.accepted) ? fmtMs(num(r.latency_ms)) + (num(r.min_ms) != null ? ' (' + fmtMs(num(r.min_ms)) + ' to ' + fmtMs(num(r.max_ms)) + ')' : '') : 'failed' }),
        h('td', { class: 'r', text: fmtS(num(r.audio_s)) }),
        h('td', { class: 'r', text: yes(r.accepted) ? fmtX(num(r.speed_x)) : 'n/a' }),
        h('td', { class: 'r', text: yes(r.accepted) ? fmtRtf(num(r.rtf)) : 'n/a' }),
        h('td', { class: 'r', text: fmtMib(num(r.rss_mib)) }),
        h('td', { class: 'r', text: num(r.gpu_mib) == null ? r.gpu_kind : fmtMetric(r, 'gpu') }),
        h('td', { class: 'r', text: num(r.wer_pct) == null ? 'n/a' : fmtPct(num(r.wer_pct)) }),
        h('td', { class: 'r', text: (r.runs || 0) + ' / ' + (r.warmups == null ? '?' : r.warmups) })))));
  }
  function ratioBlock(gk, list, ratios) {
    const rr = ratios.filter((x) => x.group === gk);
    if (!rr.length) return h('p', { class: 'small muted', text: isWeekly(list) ? 'No time ratio: every row in this chart is a QVAC variant of the weekly run.' : 'No time ratio: this chart has no single QVAC reference row.' });
    return h('div', { class: 'ratio' },
      h('div', null, h('b', { text: 'Competitor ÷ QVAC time' }), h('span', { class: 'muted', text: ' (median latency; above 1 means QVAC took less time)' })),
      ...rr.map((x) => h('div', null, dot(x.engine), ' ' + label(list.find((r) => r.id === x.id) || x) + ': ', h('b', { 'data-source': 'ratios', 'data-where': 'id=' + x.id, 'data-field': 'ratio', text: fmtRatio(num(x.ratio)) }))),
      h('div', { class: 'muted', text: 'Qualifier: ' + (rr[0].qualifier || '') }));
  }
  function groupDetail(gk, list, ratios) {
    list = list.slice().sort((a, b) => (yes(b.is_qvac) - yes(a.is_qvac)) || engineRank(a.engine) - engineRank(b.engine) || a.id.localeCompare(b.id));
    const f = list[0];
    const ok = list.filter((r) => yes(r.accepted));
    const dn = durationNote(ok);
    const limits = [...new Set(list.map((r) => r.comparison_limitation).filter(Boolean))];
    const hostCond = [...new Set(list.map((r) => r.host_conditions).filter(Boolean))];
    const failed = list.filter((r) => !yes(r.accepted));
    const threads = [...new Set(ok.map((r) => r.threads).filter((t) => t != null))];
    return h('div', { class: 'gdetail' },
      h('h4', { text: panelTitle(f) + ' · ' + (SCOPE_SHORT[f.timing_class] || f.timing_label) }),
      h('div', { class: 'meta' },
        h('span', null, h('b', { text: 'Input ' }), f.scenario_label),
        h('span', null, h('b', { text: 'Timing ' }), f.timing_label),
        threads.length ? h('span', null, h('b', { text: 'CPU threads ' }), threads.join(', ')) : null,
        h('span', null, h('b', { text: 'Measured ' }), [...new Set(list.map((r) => r.date))].join(', '))),
      f.timing_includes ? h('p', { class: 'small muted', text: 'Timed region: ' + f.timing_includes }) : null,
      h('div', { class: 'scroll' }, detailTable(list)),
      h('div', null, h('div', { class: 'small muted', text: 'Precision by row' }), h('ul', { class: 'plain small' }, ...list.map((r) => h('li', null, dot(r.engine), ' ' + label(r) + ': ' + r.precision)))),
      ratioBlock(gk, list, ratios),
      dn ? h('p', { class: 'note', text: dn }) : null,
      ...hostCond.map((t) => h('p', { class: 'note', text: t })),
      ...limits.map((t) => h('p', { class: 'note info', text: 'Comparison limit: ' + t })),
      failed.length ? h('p', { class: 'note', text: 'Failed rows, shown without bars: ' + failed.map(label).join(', ') + '.' }) : null,
      sourcePanel(list));
  }
  function soloDetail(solo) {
    return h('div', { class: 'gdetail' }, h('h4', { text: 'QVAC-only timings for this model (no competitor in the same setup)' }),
      h('ul', { class: 'plain small' }, ...solo.map((r) => h('li', null, dot(r.engine), ' ' + label(r) + ' · ' + r.backend + ' · ' + (SCEN_SHORT[r.scenario] || r.scenario_label) + ' · ' +
        (SCOPE_SHORT[r.timing_class] || r.timing_label) + ': ' + (yes(r.accepted) ? fmtMs(num(r.latency_ms)) + ', ' + fmtX(num(r.speed_x)) : 'failed')))),
      sourcePanel(solo));
  }
  function notMeasured(el, cov, task) {
    const words = { asr: ['ASR'], tts: ['TTS'], music: ['Music'] }[task] || [];
    const reasons = cov.filter((c) => c.status !== 'measured' && (c.task === 'All' || words.some((w) => String(c.task).includes(w))));
    el.replaceChildren(h('div', { class: 'card' }, h('p', { class: 'small', text: 'Not measured on this machine in this collection.' }),
      ...reasons.map((c) => h('p', { class: 'small muted' }, h('span', { class: 'pill ' + (COV_CLASS[c.status] || 'mute'), text: c.status }), ' ' + c.model + ' · ' + c.engine + ': ' + c.reason))));
  }
  function modelSection(task, idx, list, ratios, soloLines) {
    const f = list[0];
    const groups = d3.groups(list, (r) => r.group)
      .sort((a, b) => backendRank(a[1][0].backend) - backendRank(b[1][0].backend) || SCEN_RANK.indexOf(a[1][0].scenario) - SCEN_RANK.indexOf(b[1][0].scenario) || a[0].localeCompare(b[0]));
    const comp = groups.filter(([, l]) => isComparison(l));
    const solo = groups.filter(([, l]) => !isComparison(l)).flatMap(([, l]) => l);
    if (!comp.length) {
      const ok = solo.filter((r) => yes(r.accepted));
      soloLines.push(h('li', null, h('b', { text: f.model + (f.task === 'TTS cloning' ? ', voice cloning' : '') }),
        ': ' + [...new Set(ok.map((r) => BACKEND_LABEL[r.backend] || r.backend))].join(', ') + (ok.length ? '' : ' (failed runs only)')));
      return null;
    }
    const details = h('details', { class: 'mdetail' }, h('summary', { text: 'Details: exact values, precision, time ratios, comparison limits and sources' }));
    const inner = h('div', { class: 'mdetail-body' });
    details.append(inner);
    details.addEventListener('toggle', () => {
      if (!details.open || inner.childElementCount) return;
      inner.append(...comp.map(([gk, l]) => groupDetail(gk, l, ratios)));
      if (solo.length) inner.append(soloDetail(solo));
    });
    return h('section', { class: 'card msec', id: 'm-' + task + '-' + idx },
      h('h2', null, f.model + ': ' + WHAT[f.task] + ' (higher is better)', yes(f.required_family) ? h('span', null, ' ', h('span', { class: 'tag req', text: 'required' })) : null),
      h('p', { class: 'small muted', text: ONE_LINE[f.task] }),
      h('div', { class: 'panels' }, ...comp.map(([, l]) => speedPanel(l))),
      details);
  }

  dash.draw('models', (el) => {
    const all = rowsOf('runs');
    const ratios = rowsOf('ratios');
    const cov = rowsOf('coverage');
    if (!all || !ratios || !cov) return dash0(el);
    const task = taskOf(el);
    const rs = all.filter((r) => r.dashboard === task);
    if (!rs.length) return notMeasured(el, cov, task);
    const byModel = d3.groups(rs, (r) => r.model + '|' + r.task)
      .sort((a, b) => modelRank(a[0]) - modelRank(b[0]) || a[0].localeCompare(b[0]));
    const soloLines = [];
    const sections = byModel.map(([, list], i) => modelSection(task, i, list, ratios, soloLines)).filter(Boolean);
    if (soloLines.length) {
      sections.push(h('section', { class: 'card msec' }, h('h3', { text: 'Measured in QVAC only on this machine' }),
        h('p', { class: 'small muted', text: 'No competitor ran these models here, so there is nothing to compare. Their timings are in the Runs and coverage tab.' }),
        h('ul', { class: 'plain small' }, ...soloLines)));
    }
    el.replaceChildren(...sections);
  });

  // ---------- runs and coverage ----------
  dash.draw('runs-table', (el) => {
    const all = rowsOf('runs');
    if (!all) return el.replaceChildren();
    let list = all.filter((r) => (!runStatus || r.status === runStatus) && (!runQuery || JSON.stringify(r).toLowerCase().includes(runQuery)));
    const k = runSort.key;
    list = list.slice().sort((a, b) => {
      const va = num(a[k]) != null && k !== 'id' ? num(a[k]) : String(a[k] ?? '');
      const vb = num(b[k]) != null && k !== 'id' ? num(b[k]) : String(b[k] ?? '');
      return (va < vb ? -1 : va > vb ? 1 : 0) * runSort.dir;
    });
    el.replaceChildren(...list.map((r) => h('tr', { 'data-key': r.id, class: r.id === selectedRun ? 'sel' : null, tabindex: '0' },
      h('td', { text: r.id }), h('td', { text: r.machine }), h('td', { text: r.backend }), h('td', { text: r.task }),
      h('td', null, dot(r.engine), ' ', r.model), h('td', { text: r.engine }), h('td', { text: r.variant }),
      h('td', { text: r.scenario_label }), h('td', { text: r.timing_label }),
      h('td', null, h('span', { class: 'pill ' + (yes(r.accepted) ? 'ok' : 'bad'), text: r.status })),
      h('td', { class: 'r', text: yes(r.accepted) ? fmtMs(num(r.latency_ms)) : 'n/a' }),
      h('td', { class: 'r', text: num(r.audio_s) == null ? 'n/a' : d3.format('.2f')(num(r.audio_s)) }),
      h('td', { class: 'r', text: yes(r.accepted) ? fmtX(num(r.speed_x)) : 'n/a' }),
      h('td', { class: 'r', text: yes(r.accepted) ? fmtRtf(num(r.rtf)) : 'n/a' }),
      h('td', { class: 'r', text: num(r.rss_mib) == null ? 'n/a' : fmtInt(Math.round(num(r.rss_mib))) }),
      h('td', { class: 'r', text: num(r.gpu_mib) == null ? 'n/a' : (r.gpu_kind === 'sampled lower bound' ? '≥ ' : '') + fmtInt(Math.round(num(r.gpu_mib))) }),
      h('td', { class: 'r', text: num(r.wer_pct) == null ? 'n/a' : d3.format('.1f')(num(r.wer_pct)) }),
      h('td', { class: 'r', text: r.runs }), h('td', { class: 'r', text: r.warmups == null ? '?' : r.warmups }), h('td', { text: r.date }))));
  });

  dash.draw('run-detail', (el) => {
    const all = rowsOf('runs');
    if (!all) return dash0(el);
    const r = all.find((x) => x.id === selectedRun);
    if (!r) return el.replaceChildren(h('p', { class: 'small muted', text: 'Select a row to see its full provenance here.' }));
    el.replaceChildren(h('div', { class: 'src' }, h('h4', { text: 'Provenance for ' + r.id }), sourceBlock(r)));
  });

  dash.draw('coverage-table', (el) => {
    const cov = rowsOf('coverage');
    if (!cov) return el.replaceChildren();
    const other = ['excluded', 'install failed', 'configuration failed'];
    const list = cov.filter((c) => !covStatus || (covStatus === 'other' ? other.includes(c.status) : c.status === covStatus))
      .slice().sort((a, b) => machineRank(a.machine) - machineRank(b.machine) || String(a.task).localeCompare(String(b.task)) || String(a.model).localeCompare(String(b.model)));
    el.replaceChildren(...list.map((c) => h('tr', null,
      h('td', { text: c.id }), h('td', { text: c.machine }), h('td', { text: c.backend }), h('td', { text: c.task }), h('td', { text: c.model }),
      h('td', null, ENGINE_COLORS[c.engine] ? dot(c.engine) : null, ' ', c.engine),
      h('td', null, h('span', { class: 'pill ' + (COV_CLASS[c.status] || 'mute'), text: c.status })),
      h('td', { class: 'r', text: c.rows }), h('td', { text: c.scenario }), h('td', { text: c.reason }), h('td', null, h('code', { text: c.source })))));
  });

  // ---------- history ----------
  dash.draw('prior-models', (el) => {
    const p = rowsOf('prior_models');
    if (!p) return el.replaceChildren();
    el.replaceChildren(...p.map((x) => h('tr', null, h('td', { text: x.id }), h('td', null, h('b', { text: x.model })), h('td', { text: x.campaign_date }), h('td', { text: x.competitors }), h('td', { class: 'r', text: x.rows }), h('td', { text: x.note }))));
  });
  dash.draw('prior-campaign', (el) => {
    const p = rowsOf('prior_campaign');
    if (!p) return el.replaceChildren();
    const list = p.filter((x) => !histQuery || JSON.stringify(x).toLowerCase().includes(histQuery));
    el.replaceChildren(...list.map((x) => h('tr', null, h('td', { text: x.id }), h('td', { text: x.model }), h('td', { text: x.campaign_date }), h('td', { text: x.device }), h('td', { text: x.backend }), h('td', { text: x.qvac_rtf_as_reported }), h('td', { text: x.competitor }), h('td', { text: x.ratio_as_reported }))));
  });
  dash.draw('export-table', (el) => {
    const p = rowsOf('dashboard_export');
    if (!p) return el.replaceChildren();
    el.replaceChildren(...p.map((x) => h('tr', null, h('td', { text: x.id }), h('td', { text: x.task }), h('td', { text: x.machine }), h('td', { text: x.backend }), h('td', { text: x.model }), h('td', { class: 'r', text: fmtInt(num(x.speed_x_displayed)) + '×' }), h('td', { text: x.displayed_date }))));
  });

  // ---------- week over week ----------
  const trendFamily = {};
  const familyLabel = (r) => r.model + (r.variant ? ' (' + r.variant + ')' : '');
  dash.draw('trend-model', (sel) => {
    const hist = rowsOf('history');
    if (!hist) return sel.replaceChildren();
    const task = taskOf(sel);
    const fams = [...new Map(hist.filter((r) => r.dashboard === task && r.status === 'measured').map((r) => [r.family, r])).values()]
      .sort((a, b) => a.model.localeCompare(b.model) || String(a.variant).localeCompare(String(b.variant)));
    sel.replaceChildren(...fams.map((r) => new Option(familyLabel(r), r.family)));
    if (!fams.some((r) => r.family === trendFamily[task])) trendFamily[task] = fams.length ? fams[0].family : null;
    sel.value = trendFamily[task] || '';
    sel.onchange = () => { trendFamily[task] = sel.value; dash.draw('trend'); };
  });
  function trendSeries(rows, key) {
    const byBackend = d3.groups(rows.filter((r) => r.status === 'measured' && num(r[METRICS[key].field]) != null), (r) => r.backend)
      .sort((a, b) => backendRank(a[0]) - backendRank(b[0]));
    return byBackend.map(([backend, list]) => ({ backend, color: BACKEND_COLORS[backend] || 'var(--color-fg-muted)',
      pts: list.map((r) => ({ x: new Date(r.started_utc || r.date), y: num(r[METRICS[key].field]), row: r })).sort((p, q) => p.x - q.x) }));
  }
  function trendLines(s, p, M) {
    return [(BACKEND_LABEL[s.backend] || s.backend) + ' · run ' + p.row.run + ' · ' + p.row.date, M.label + ': ' + M.fmt(p.y), p.row.pins];
  }
  function trendChart(el, box, series, key) {
    const M = METRICS[key];
    const W = Math.max(TREND_MIN_WIDTH, el.clientWidth), H = TREND_HEIGHT, padL = 56, padR = 16, padT = 10, padB = 28;
    const all = series.flatMap((s) => s.pts);
    const x = d3.scaleUtc(d3.extent(all, (p) => p.x), [padL, W - padR]);
    if (+x.domain()[0] === +x.domain()[1]) x.domain([d3.utcDay.offset(x.domain()[0], -7), d3.utcDay.offset(x.domain()[1], 7)]);
    const y = (M.log ? d3.scaleLog().domain([d3.min(all, (p) => p.y) / 1.5, d3.max(all, (p) => p.y) * 1.3]) : d3.scaleLinear().domain([0, d3.max(all, (p) => p.y) * 1.1]).nice()).range([H - padB, padT]);
    const svg = d3.create('svg').attr('width', '100%').attr('viewBox', '0 0 ' + W + ' ' + H).attr('role', 'img').attr('aria-label', M.label + ' by run');
    svg.append('g').attr('class', 'tgrid').selectAll('line').data(y.ticks(4)).join('line').attr('x1', padL).attr('x2', W - padR).attr('y1', (t) => y(t)).attr('y2', (t) => y(t)).attr('stroke', 'var(--color-border-line)');
    svg.append('g').selectAll('text').data(y.ticks(4)).join('text').attr('x', padL - 6).attr('y', (t) => y(t) + 4).attr('text-anchor', 'end').attr('fill', 'var(--color-fg-muted)').attr('font-size', '11').text((t) => M.fmt(t));
    svg.append('g').selectAll('text').data(x.ticks(Math.min(6, Math.max(2, Math.round(W / 120))))).join('text').attr('x', (t) => x(t)).attr('y', H - 8).attr('text-anchor', 'middle').attr('fill', 'var(--color-fg-muted)').attr('font-size', '11').text(d3.utcFormat('%b %-d'));
    const line = d3.line().x((p) => x(p.x)).y((p) => y(p.y));
    series.forEach((s) => {
      svg.append('path').attr('d', line(s.pts)).attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', 2);
      svg.append('g').selectAll('circle').data(s.pts).join('circle').attr('cx', (p) => x(p.x)).attr('cy', (p) => y(p.y)).attr('r', 4.5).attr('fill', s.color).attr('tabindex', '0')
        .each(function (p) { bindTip(box, this, () => trendLines(s, p, M)); });
    });
    return svg.node();
  }
  dash.draw('trend', (el) => {
    const hist = rowsOf('history');
    if (!hist) return dash0(el);
    const task = taskOf(el);
    const key = metricKey();
    const family = trendFamily[task] || (hist.find((r) => r.dashboard === task) || {}).family;
    const rows = hist.filter((r) => r.family === family);
    const series = trendSeries(rows, key);
    if (!series.length) return el.replaceChildren(h('p', { class: 'small muted', text: 'No stored run has this metric for this model yet.' }));
    const legend = h('div', { class: 'legend' }, ...series.map((s) => h('span', null, h('span', { class: 'dot', style: { background: s.color } }), BACKEND_LABEL[s.backend] || s.backend)));
    const box = h('div', { class: 'rows' });
    box.append(legend, trendChart(el, box, series, key));
    el.replaceChildren(box);
  });

  const named = document.getElementById(location.hash.slice(1));
  if (named && !TABS.includes(location.hash.slice(1))) named.scrollIntoView();
})();
