/* ============================================================
   Portfolio Pulse — frontend logic (vanilla JS, Chart.js v4)
   ============================================================ */
'use strict';

/* ---------- Tiny DOM helpers (XSS-safe: textContent only) ---------- */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}

/** Only allow http(s) URLs for outbound links; otherwise inert '#'. */
function safeUrl(u) {
  try {
    const x = new URL(u, window.location.origin);
    return (x.protocol === 'http:' || x.protocol === 'https:') ? x.href : '#';
  } catch {
    return '#';
  }
}

function clamp01(v) {
  const n = Number(v);
  if (!isFinite(n)) return 0;
  return Math.min(1, Math.max(0, n));
}

function pct(v) { return Math.round(clamp01(v) * 100) + '%'; }

function fmtDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d)) return String(iso).slice(0, 10);
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

function fmtShortDate(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return String(iso).slice(5, 10);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/* ---------- Chart.js global theme ---------- */
if (window.Chart) {
  Chart.defaults.color = '#8b98a9';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
  Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
  Chart.defaults.font.size = 11;
}
let charts = {};
function destroyCharts() {
  Object.values(charts).forEach(c => { try { c && c.destroy(); } catch {} });
  charts = {};
}

const PALETTE = ['#10b981', '#22d3ee', '#f59e0b', '#8b5cf6', '#ec4899', '#3b82f6', '#14b8a6', '#f97316', '#a3e635', '#64748b'];

/* ---------- Pipeline steps ---------- */
const STEPS = [
  { key: 'fetch',       label: 'Fetching news',              desc: 'Pulling up to 25 days of coverage' },
  { key: 'relevance',   label: 'Relevance & NER filtering',  desc: 'Is this story really about the company?' },
  { key: 'materiality', label: 'Financial materiality scoring', desc: 'What matters to a portfolio manager' },
  { key: 'embed',       label: 'Embedding & retrieval',      desc: 'Vector search over material news' },
  { key: 'agents',      label: 'Agent debate',               desc: 'Bull vs bear, moderated verdict' },
];

/** Map job status/progress -> active step index (-1 = none active, 5 = all done). */
function stepIndexFor(job) {
  switch (job.status) {
    case 'queued':        return 0;
    case 'fetching':      return 0;
    case 'preprocessing': return (Number(job.progress) || 0) >= 50 ? 2 : 1;
    case 'embedding':     return 3;
    case 'agents':        return 4;
    case 'done':          return 5;
    default:              return 0;
  }
}

function initSteps() {
  const ol = $('#steps');
  ol.innerHTML = '';
  STEPS.forEach((s, i) => {
    const li = el('li', 'step');
    li.id = 'step-' + i;
    const dot = el('span', 'step-dot', String(i + 1));
    const h = el('h3', null, s.label);
    const p = el('p', null, s.desc);
    li.append(dot, h, p);
    ol.appendChild(li);
  });
}

function renderPipeline(job) {
  const progress = Math.min(100, Math.max(0, Number(job.progress) || 0));
  $('#progress-fill').style.width = progress + '%';
  $('#progress-pct').textContent = Math.round(progress) + '%';
  $('#progress-aria').setAttribute('aria-valuenow', String(Math.round(progress)));
  $('#stage-msg').textContent = job.message || job.stage || 'Working…';

  const active = stepIndexFor(job);
  STEPS.forEach((_, i) => {
    const li = $('#step-' + i);
    li.classList.toggle('done', i < active);
    li.classList.toggle('active', i === active);
    const dot = li.querySelector('.step-dot');
    dot.textContent = i < active ? '✓' : String(i + 1);
  });
}

/* ---------- Analysis (single synchronous request) ---------- */
let stageTimers = [];

function stopStages() {
  stageTimers.forEach(t => clearTimeout(t));
  stageTimers = [];
}

// The backend runs the whole pipeline in one request (serverless-friendly),
// so the UI walks through staged progress messages while awaiting the result.
const STAGE_MESSAGES = [
  [0, 'Fetching news from configured sources…'],
  [9000, 'Deduplicating and filtering for entity relevance…'],
  [20000, 'Scoring financial materiality…'],
  [32000, 'Bull agent building the upside case…'],
  [55000, 'Bear agent building the downside case…'],
  [85000, 'Moderator weighing both cases…'],
  [120000, 'Assembling your dashboard…'],
];

function playStages() {
  stopStages();
  const stepCount = STEPS.length;
  STAGE_MESSAGES.forEach(([delay, message], idx) => {
    stageTimers.push(setTimeout(() => {
      renderPipeline({ status: 'working', progress: Math.min(8 + idx * 13, 94), message });
      const active = Math.min(idx + 1, stepCount - 1);
      STEPS.forEach((_, i) => {
        const li = $('#step-' + i);
        li.classList.toggle('done', i < active);
        li.classList.toggle('active', i === active);
        const dot = li.querySelector('.step-dot');
        dot.textContent = i < active ? '✓' : String(i + 1);
      });
    }, delay));
  });
}

async function startAnalysis(payload) {
  stopStages();
  destroyCharts();

  $('#dashboard').hidden = true;
  $('#error-card').hidden = true;
  $('#pipeline').hidden = false;
  $('#news-list').innerHTML = '';
  initSteps();
  renderPipeline({ status: 'queued', progress: 4, message: 'Starting analysis…' });
  playStages();

  const btn = $('#run-btn');
  btn.disabled = true;
  btn.querySelector('span').textContent = 'Analyzing…';

  $('#pipeline').scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    const r = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      let detail = 'Server returned ' + r.status;
      try { detail = (await r.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const data = await r.json();
    stopStages();
    resetRunButton();
    renderResults(data);
    $('#pipeline').hidden = true;
    $('#dashboard').hidden = false;
    $('#dashboard').scrollIntoView({ behavior: 'smooth', block: 'start' });
    observeReveals();
  } catch (err) {
    stopStages();
    resetRunButton();
    showError('The analysis could not be completed: ' + err.message);
  }
}

function resetRunButton() {
  const btn = $('#run-btn');
  btn.disabled = false;
  btn.querySelector('span').textContent = 'Run Analysis';
}

function showError(msg) {
  $('#pipeline').hidden = true;
  $('#dashboard').hidden = true;
  $('#error-card').hidden = false;
  $('#error-msg').textContent = msg;
  $('#error-card').scrollIntoView({ behavior: 'smooth', block: 'center' });
  observeReveals();
}

/* ---------- Results ---------- */

function renderResults(d) {
  renderStats(d);
  renderVerdict(d);
  renderBrief(d);
  buildCharts(d);
  renderAgent($('#bull-score'), $('#bull-thesis'), $('#bull-bullets'), $('#bull-evidence'), d.bull_case, 'bull');
  renderAgent($('#bear-score'), $('#bear-thesis'), $('#bear-bullets'), $('#bear-evidence'), d.bear_case, 'bear');
  renderNews(d.news || []);
  renderMeta(d);
}

function renderStats(d) {
  const strip = $('#stats-strip');
  strip.innerHTML = '';
  const s = d.stats || {};

  const stat = (num, label, emph) => {
    const box = el('div', 'stat');
    box.append(el('div', 'stat-num' + (emph ? ' em' : ''), String(num)), el('div', 'stat-label', label));
    return box;
  };
  const arrow = () => {
    const a = el('span', 'stat-arrow');
    a.setAttribute('aria-hidden', 'true');
    a.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none"><path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    return a;
  };

  strip.append(
    stat(s.fetched ?? 0, 'Fetched'),
    arrow(),
    stat(s.relevant ?? 0, 'Relevant'),
    arrow(),
    stat(s.material ?? 0, 'Material', true),
  );
  const range = el('div', 'stat-range');
  range.append(
    el('div', 'stat-num mono', (s.date_from ? fmtDate(s.date_from) : '—') + ' → ' + (s.date_to ? fmtDate(s.date_to) : '—')),
    el('div', 'stat-label', 'Coverage window')
  );
  strip.appendChild(range);
}

function renderVerdict(d) {
  const v = d.verdict || {};
  const label = String(v.label || 'NEUTRAL').toUpperCase();
  const pill = $('#verdict-label');
  pill.textContent = label;
  pill.className = 'verdict-pill ' + (label === 'BULLISH' ? 'bullish' : label === 'BEARISH' ? 'bearish' : 'neutral');
  const card = $('#verdict-card');
  card.classList.remove('verdict-bearish', 'verdict-neutral');
  if (label === 'BEARISH') card.classList.add('verdict-bearish');
  if (label === 'NEUTRAL') card.classList.add('verdict-neutral');

  const conf = clamp01(v.confidence);
  $('#verdict-conf-fill').style.width = '0%';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $('#verdict-conf-fill').style.width = (conf * 100) + '%';
  }));
  $('#verdict-conf-pct').textContent = pct(conf);
  $('#verdict-summary').textContent = v.summary || 'No summary available.';
  const subj = [d.company_name, d.ticker ? '(' + String(d.ticker).toUpperCase() + ')' : '', d.generated_at ? '· generated ' + fmtDate(d.generated_at) : '']
    .filter(Boolean).join(' ');
  $('#verdict-subject').textContent = subj;
}

function fillList(ul, items) {
  ul.innerHTML = '';
  (items || []).forEach(t => {
    if (t === null || t === undefined || String(t).trim() === '') return;
    ul.appendChild(el('li', null, String(t)));
  });
  if (!ul.children.length) ul.appendChild(el('li', null, 'None reported.'));
}

function renderBrief(d) {
  const v = d.verdict || {};
  fillList($('#brief-bullets'), v.bullets);
  fillList($('#brief-risks'), v.risks);
  fillList($('#brief-watch'), v.watch_items);
}

function renderAgent(scoreEl, thesisEl, bulletsEl, evEl, agent, kind) {
  const a = agent || {};
  const sc = clamp01(a.score);
  const conf = a.confidence !== undefined ? ' · confidence ' + pct(a.confidence) : '';
  scoreEl.textContent = 'Conviction ' + pct(sc) + conf;
  thesisEl.textContent = a.thesis || 'No thesis provided.';
  fillList(bulletsEl, a.bullets);
  evEl.innerHTML = '';
  (a.evidence || []).forEach(e => {
    const li = el('li');
    li.append(el('span', 'ev-title', String(e.title || 'Untitled')), el('span', 'ev-why', String(e.why || '')));
    evEl.appendChild(li);
  });
  if (!evEl.children.length) evEl.appendChild(el('li', null, 'No cited evidence.'));
}

function sentimentClass(label) {
  const l = String(label || '').toLowerCase();
  if (l.includes('pos') || l.includes('bull')) return 'positive';
  if (l.includes('neg') || l.includes('bear')) return 'negative';
  return 'neutral';
}

function meter(label, value) {
  const box = el('div', 'meter');
  const row = el('div', 'meter-row');
  row.append(el('span', null, label), el('span', 'mono', pct(value)));
  const track = el('div', 'meter-track');
  const fill = el('div', 'meter-fill');
  fill.style.width = '0%';
  const target = pct(value);
  requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = target; }));
  track.appendChild(fill);
  box.append(row, track);
  return box;
}

function renderNews(news) {
  const list = $('#news-list');
  list.innerHTML = '';
  if (!news.length) {
    list.appendChild(el('p', null, 'No material news items.')).style.color = 'var(--muted)';
    return;
  }
  news.forEach(n => {
    const item = el('article', 'news-item');

    const top = el('div', 'news-top');
    if (n.priority) top.appendChild(el('span', 'chip priority-tag', 'Priority'));
    top.appendChild(el('span', 'sent-badge ' + sentimentClass(n.sentiment_label), String(n.sentiment_label || 'neutral').toUpperCase()));
    (n.categories || []).slice(0, 4).forEach(c => top.appendChild(el('span', 'chip', String(c))));

    const title = el('h4', 'news-title');
    const a = el('a', null, String(n.title || 'Untitled'));
    a.href = safeUrl(n.url);
    if (a.href !== '#') { a.target = '_blank'; a.rel = 'noopener noreferrer'; }
    title.appendChild(a);

    const meta = el('div', 'news-meta',
      [n.source || 'Unknown source', fmtDate(n.published_at)].filter(Boolean).join(' · '));

    const bars = el('div', 'news-bars');
    bars.append(meter('Relevance', n.relevance), meter('Materiality', n.materiality));

    item.append(top, title);
    if (n.summary) item.appendChild(el('p', 'news-summary', String(n.summary)));
    item.append(meta, bars);
    list.appendChild(item);
  });
}

function renderMeta(d) {
  const m = d.meta || {};
  const line = $('#meta-line');
  line.innerHTML = '';
  const parts = [];
  if ((m.sources_used || []).length) parts.push('Sources: ' + m.sources_used.join(', '));
  if (m.models && Object.keys(m.models).length) {
    parts.push('Models: ' + Object.entries(m.models).map(([k, v]) => k + '=' + v).join(', '));
  }
  line.textContent = parts.join('  ·  ');
  if (m.demo_mode) {
    const badge = el('span', 'demo-badge', 'DEMO DATA');
    line.appendChild(badge);
  }
}

/* ---------- Charts ---------- */
function buildCharts(d) {
  destroyCharts();
  const c = d.charts || {};

  /* (a) News by category — doughnut */
  const cats = c.category_breakdown || [];
  charts.categories = new Chart($('#chart-categories'), {
    type: 'doughnut',
    data: {
      labels: cats.map(x => String(x.label)),
      datasets: [{
        data: cats.map(x => Number(x.count) || 0),
        backgroundColor: cats.map((_, i) => PALETTE[i % PALETTE.length]),
        borderColor: '#0a0e14',
        borderWidth: 3,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '62%',
      plugins: {
        legend: { position: 'right', labels: { boxWidth: 12, padding: 14, usePointStyle: true } },
        tooltip: { callbacks: { label: ctx => ' ' + ctx.label + ': ' + ctx.parsed + ' articles' } },
      },
    },
  });

  /* (b) Coverage timeline — bars (count) + line (sentiment), priority days highlighted */
  const tl = c.timeline || [];
  const dates = tl.map(t => fmtShortDate(t.date));
  charts.timeline = new Chart($('#chart-timeline'), {
    data: {
      labels: dates,
      datasets: [
        {
          type: 'bar', label: 'Articles', yAxisID: 'y',
          data: tl.map(t => Number(t.count) || 0),
          backgroundColor: tl.map(t => t.priority ? 'rgba(34,211,238,0.9)' : 'rgba(139,152,169,0.32)'),
          hoverBackgroundColor: tl.map(t => t.priority ? '#22d3ee' : 'rgba(139,152,169,0.6)'),
          borderRadius: 4, barPercentage: 0.75,
        },
        {
          type: 'line', label: 'Avg sentiment', yAxisID: 'y1',
          data: tl.map(t => (t.avg_sentiment === null || t.avg_sentiment === undefined) ? null : Number(t.avg_sentiment)),
          borderColor: '#10b981', backgroundColor: '#10b981',
          borderWidth: 2, tension: 0.35, pointRadius: 2.5, spanGaps: true,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { ticks: { maxTicksLimit: 10, maxRotation: 45 } },
        y: { beginAtZero: true, title: { display: true, text: 'Articles' }, ticks: { precision: 0 } },
        y1: { position: 'right', min: -1, max: 1, title: { display: true, text: 'Sentiment' }, grid: { drawOnChartArea: false } },
      },
      plugins: {
        tooltip: {
          callbacks: {
            afterBody: ctx => {
              const t = tl[ctx[0].dataIndex];
              return t && t.material_count !== undefined ? 'Material: ' + t.material_count : '';
            },
          },
        },
      },
    },
  });

  /* (c) Materiality distribution — bars */
  const md = c.materiality_dist || [];
  charts.materiality = new Chart($('#chart-materiality'), {
    type: 'bar',
    data: {
      labels: md.map(x => String(x.bucket)),
      datasets: [{
        label: 'Articles',
        data: md.map(x => Number(x.count) || 0),
        backgroundColor: 'rgba(16,185,129,0.75)',
        hoverBackgroundColor: '#10b981',
        borderRadius: 6, barPercentage: 0.7,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Materiality score' } },
        y: { beginAtZero: true, title: { display: true, text: 'Articles' }, ticks: { precision: 0 } },
      },
      plugins: { legend: { display: false } },
    },
  });

  /* (d) Bull vs Bear — opposing horizontal bars */
  const scores = c.agent_scores || {};
  const bull = clamp01(scores.bull), bear = clamp01(scores.bear);
  charts.duel = new Chart($('#chart-duel'), {
    type: 'bar',
    data: {
      labels: ['Bull conviction', 'Bear conviction'],
      datasets: [{
        data: [bull, bear],
        backgroundColor: ['rgba(16,185,129,0.9)', 'rgba(239,68,68,0.9)'],
        hoverBackgroundColor: ['#10b981', '#ef4444'],
        borderRadius: 8, barThickness: 30,
      }],
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      scales: { x: { min: 0, max: 1, ticks: { stepSize: 0.25 }, title: { display: true, text: 'Conviction (0–1)' } } },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ' ' + ctx.parsed.x.toFixed(2) } },
      },
    },
  });
}

/* ---------- Reveal on scroll ---------- */
let revealObserver = null;
function observeReveals() {
  if (!('IntersectionObserver' in window)) {
    $$('.reveal').forEach(n => n.classList.add('visible'));
    return;
  }
  if (!revealObserver) {
    revealObserver = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); revealObserver.unobserve(e.target); } });
    }, { threshold: 0.08 });
  }
  $$('.reveal:not(.visible)').forEach(n => revealObserver.observe(n));
}

/* ---------- Demo-mode note (best effort; backend may not expose it) ---------- */
async function checkDemoMode() {
  try {
    const r = await fetch('/api/health', { cache: 'no-store' });
    if (!r.ok) return;
    const h = await r.json();
    if (h && h.demo_mode) $('#demo-note').hidden = false;
  } catch { /* backend without /api/health — ignore */ }
}

/* ---------- Init ---------- */
document.addEventListener('DOMContentLoaded', () => {
  observeReveals();
  checkDemoMode();

  $('#analyze-form').addEventListener('submit', e => {
    e.preventDefault();
    const company = $('#company').value.trim();
    const ticker = $('#ticker').value.trim().toUpperCase();
    const query = $('#query').value.trim();
    const days = Math.min(60, Math.max(1, parseInt($('#days').value, 10) || 25));
    if (!company || !ticker) return;
    $('#ticker').value = ticker;
    startAnalysis({ ticker, company_name: company, query: query || 'What is the financial risk for this company?', days });
  });
});
