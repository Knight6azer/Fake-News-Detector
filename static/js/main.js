/* ═══════════════════════════════════════════════════════════════════════════
   TruthLens — Main JavaScript
   Handles: predictions, charts, particles, word highlights, history, batch
   ═══════════════════════════════════════════════════════════════════════════ */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  selectedModel: 'tfidf',
  theme: localStorage.getItem('theme') || 'dark',
  history: JSON.parse(localStorage.getItem('predHistory') || '[]'),
  pieChart: null,
  sessionStats: { total: 0, fake: 0, real: 0, totalConf: 0 },
};

// ── DOM Ready ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initParticles();
  initNavScroll();
  initModelTabs();
  initTextInput();
  initPieChart();
  renderHistory();
  updateHeroStats();
  checkModelStatus();

  // Scroll animations
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) e.target.classList.add('visible');
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.glass-card, .step-card, .model-card, .stat-card')
    .forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(20px)';
      el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
      observer.observe(el);
    });

  // Add visible style
  const style = document.createElement('style');
  style.textContent = `.visible { opacity: 1 !important; transform: translateY(0) !important; }`;
  document.head.appendChild(style);
});

// ── Theme ───────────────────────────────────────────────────────────────────
function initTheme() {
  applyTheme(state.theme);
  document.getElementById('themeToggle').addEventListener('click', () => {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('theme', state.theme);
    applyTheme(state.theme);
  });
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  document.getElementById('themeToggle').textContent = theme === 'dark' ? '🌙' : '☀️';
}

// ── Navbar Scroll ───────────────────────────────────────────────────────────
function initNavScroll() {
  const nav = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 60);
  }, { passive: true });
}

// ── Model Tabs ──────────────────────────────────────────────────────────────
function initModelTabs() {
  document.querySelectorAll('.model-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.selectedModel = tab.dataset.model;
    });
  });
}

// ── Text Input Counter ──────────────────────────────────────────────────────
function initTextInput() {
  const textarea = document.getElementById('textInput');
  const counter  = document.getElementById('charCount');

  textarea.addEventListener('input', () => {
    const words = textarea.value.trim().split(/\s+/).filter(Boolean).length;
    counter.textContent = `${words} word${words !== 1 ? 's' : ''}`;
    counter.style.color = words < 5 ? 'var(--fake-color)' : 'var(--text-muted)';
  });

  textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) runAnalysis();
  });
}

// ── Check Model Status ──────────────────────────────────────────────────────
async function checkModelStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();
    const badge = document.getElementById('navModelBadge');
    if (data.ready) {
      badge.innerHTML = '<span class="pulse-dot"></span> AI Ready';
      badge.style.color = 'var(--real-color)';
    } else {
      badge.innerHTML = '⏳ Training…';
      badge.style.color = 'var(--accent-blue)';
      setTimeout(checkModelStatus, 5000);
    }
  } catch (e) {
    document.getElementById('navModelBadge').innerHTML = '🔴 Offline';
  }
}

// ── Sample Texts ────────────────────────────────────────────────────────────
const SAMPLES = {
  fake: {
    title: 'BOMBSHELL: Government ADMITS 5G Towers Cause COVID-19, Mainstream Media HIDING Truth!',
    text:  'SHOCKING new leaked documents from a Pentagon whistleblower PROVE beyond any doubt that 5G radiation activates dormant viruses in the human body. Mainstream media is CENSORING this explosive bombshell. The deep state and globalist elites are desperately trying to suppress this information. Share before they DELETE this! Scientists who challenged this secret have been SILENCED. The truth is finally being REVEALED — the government has known since 2020 and has been HIDING it from you. Wake up! This is not a drill. Your family\'s lives are at stake.',
  },
  real: {
    title: 'Federal Reserve Holds Interest Rates Steady Amid Stable Inflation Data',
    text:  'The Federal Reserve voted unanimously on Wednesday to maintain its benchmark interest rate at current levels, citing stable inflation figures and a resilient labor market. Fed Chair Jerome Powell noted that while inflation has moderated significantly from its 2022 peak, the committee wants additional data before making further adjustments. The decision was in line with market expectations. Consumer price index data released last week showed annual inflation at 2.4%, approaching the Fed\'s 2% target. Economists broadly interpreted this pause as a sign of a measured approach to monetary policy normalization.',
  }
};

function loadSample(type) {
  const sample = SAMPLES[type];
  if (!sample) return;
  document.getElementById('titleInput').value = sample.title;
  document.getElementById('textInput').value = sample.text;
  document.getElementById('textInput').dispatchEvent(new Event('input'));
  showToast(`Loaded ${type === 'fake' ? '🚫 Fake' : '✅ Real'} sample`);
}

function clearAll() {
  document.getElementById('titleInput').value = '';
  document.getElementById('textInput').value = '';
  document.getElementById('charCount').textContent = '0 words';
  document.getElementById('resultPlaceholder').classList.remove('hidden');
  document.getElementById('resultContent').classList.add('hidden');
}

// ── Main Analysis ───────────────────────────────────────────────────────────
async function runAnalysis() {
  const text  = document.getElementById('textInput').value.trim();
  const title = document.getElementById('titleInput').value.trim();

  if (!text && !title) {
    showToast('⚠️ Please enter some text to analyze.', 3000);
    document.getElementById('textInput').focus();
    return;
  }

  if ((text + title).split(/\s+/).filter(Boolean).length < 3) {
    showToast('⚠️ Please enter at least 3 words.', 3000);
    return;
  }

  setAnalyzing(true);

  try {
    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text, title, model: state.selectedModel }),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showToast(`❌ ${data.error || 'Server error'}`, 4000);
      return;
    }

    displayResult(data, title || text);
    addToHistory({ ...data, text: title || text.slice(0, 80) });
    await refreshStats();

  } catch (err) {
    showToast('❌ Connection error. Is the server running?', 4000);
    console.error(err);
  } finally {
    setAnalyzing(false);
  }
}

function setAnalyzing(loading) {
  const btn     = document.getElementById('analyzeBtn');
  const btnText = document.getElementById('analyzeBtnText');
  const spinner = document.getElementById('btnSpinner');
  btn.disabled = loading;
  btnText.classList.toggle('hidden', loading);
  spinner.classList.toggle('hidden', !loading);
  if (loading) btnText.textContent = '🔍 Analyze Now';
}

// ── Display Result ──────────────────────────────────────────────────────────
function displayResult(data, inputText) {
  const isFake   = data.label === 'FAKE';
  const confidence = parseFloat(data.confidence) || 0;

  // Show result panel
  document.getElementById('resultPlaceholder').classList.add('hidden');
  document.getElementById('resultContent').classList.remove('hidden');

  // Verdict header
  const header = document.getElementById('verdictHeader');
  header.className = `verdict-header ${isFake ? 'fake-verdict' : 'real-verdict'}`;

  document.getElementById('verdictIcon').textContent  = isFake ? '🚫' : '✅';
  document.getElementById('verdictLabel').textContent = `${data.label} News`;
  document.getElementById('verdictLabel').style.color = isFake ? 'var(--fake-color)' : 'var(--real-color)';
  document.getElementById('verdictModel').textContent = `via ${data.model_used || 'TF-IDF'}`;
  document.getElementById('verdictTime').textContent  = new Date().toLocaleTimeString();

  // Gauge
  animateGauge(confidence, isFake);

  // Confidence bars
  const fakeConf = isFake ? confidence : (100 - confidence);
  const realConf = isFake ? (100 - confidence) : confidence;

  setTimeout(() => {
    document.getElementById('fakeBar').style.width = `${fakeConf}%`;
    document.getElementById('realBar').style.width = `${realConf}%`;
  }, 100);

  document.getElementById('fakePct').textContent = `${fakeConf.toFixed(1)}%`;
  document.getElementById('realPct').textContent = `${realConf.toFixed(1)}%`;

  // Word highlights
  renderWordHighlights(data.word_highlights || [], inputText);

  // Meta info
  document.getElementById('wordCount').textContent  = `${data.input_length || '—'} words`;
  document.getElementById('elapsedMs').textContent  = `${data.elapsed_ms || '—'}ms`;
  document.getElementById('modelName').textContent  = data.model_used || '—';

  // Scroll to result on mobile
  if (window.innerWidth < 768) {
    document.getElementById('resultPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ── Gauge Animation ─────────────────────────────────────────────────────────
function animateGauge(pct, isFake) {
  // SVG arc: total arc length of the semi-circle path ≈ 251
  const FULL_DASH = 251;
  const fill = (pct / 100) * FULL_DASH;

  const gaugeFill = document.getElementById('gaugeFill');
  const gaugePct  = document.getElementById('gaugePct');

  // Inject gradient defs once
  const svg = gaugeFill.closest('svg');
  if (!svg.querySelector('#gaugeGrad')) {
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
      <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="${isFake ? '#f43f5e' : '#10b981'}"/>
        <stop offset="100%" stop-color="${isFake ? '#fb7185' : '#34d399'}"/>
      </linearGradient>`;
    svg.prepend(defs);
  } else {
    // Update gradient color
    const stops = svg.querySelectorAll('#gaugeGrad stop');
    stops[0].setAttribute('stop-color', isFake ? '#f43f5e' : '#10b981');
    stops[1].setAttribute('stop-color', isFake ? '#fb7185' : '#34d399');
  }

  // Animate dash
  gaugeFill.style.strokeDasharray = `${fill} ${FULL_DASH}`;
  gaugePct.textContent = `${Math.round(pct)}%`;
  gaugePct.style.fill  = isFake ? 'var(--fake-color)' : 'var(--real-color)';
}

// ── Word Highlights ──────────────────────────────────────────────────────────
function renderWordHighlights(highlights, inputText) {
  const container = document.getElementById('highlightedText');

  if (!highlights || highlights.length === 0) {
    container.textContent = inputText || 'No text to display.';
    return;
  }

  // Build a map: word → direction (uses cleaned version)
  const wordMap = {};
  highlights.forEach(h => {
    if (h.word && h.direction !== 'neutral') {
      wordMap[h.word.toLowerCase()] = h.direction;
    }
  });

  // Tokenize original input preserving spaces
  const words = inputText.split(/(\s+)/);
  const fragment = document.createDocumentFragment();

  words.forEach(part => {
    if (/^\s+$/.test(part)) {
      fragment.appendChild(document.createTextNode(part));
      return;
    }
    const cleanedPart = part.replace(/[^a-zA-Z]/g, '').toLowerCase();
    const direction   = wordMap[cleanedPart];

    if (direction) {
      const span = document.createElement('span');
      span.className = `hl-word hl-${direction}`;
      span.textContent = part;
      span.title = `${direction === 'fake' ? '⚠️ Fake indicator' : '✅ Real indicator'}`;
      fragment.appendChild(span);
    } else {
      fragment.appendChild(document.createTextNode(part));
    }
  });

  container.innerHTML = '';
  container.appendChild(fragment);
}

// ── History ─────────────────────────────────────────────────────────────────
function addToHistory(entry) {
  const item = {
    id:         entry.id || Date.now(),
    text:       entry.text || '—',
    label:      entry.label,
    confidence: entry.confidence,
    model:      entry.model_used || state.selectedModel,
    time:       new Date().toLocaleTimeString(),
  };

  state.history.unshift(item);
  if (state.history.length > 50) state.history.pop();
  localStorage.setItem('predHistory', JSON.stringify(state.history));

  // Update session stats
  state.sessionStats.total++;
  if (item.label === 'FAKE') state.sessionStats.fake++;
  else state.sessionStats.real++;
  state.sessionStats.totalConf += parseFloat(item.confidence) || 0;

  renderHistory();
  updateDashboard();
  updateHeroStats();
}

function renderHistory() {
  const list = document.getElementById('historyList');
  if (!list) return;

  if (state.history.length === 0) {
    list.innerHTML = `<div class="empty-history"><span>📭</span><span>No predictions yet. Analyze some news!</span></div>`;
    return;
  }

  list.innerHTML = state.history.slice(0, 20).map(item => `
    <div class="history-item">
      <span class="hi-badge ${item.label === 'FAKE' ? 'fake' : 'real'}">${item.label}</span>
      <span class="hi-text" title="${escHtml(item.text)}">${escHtml(item.text)}</span>
      <span class="hi-conf">${parseFloat(item.confidence).toFixed(0)}%</span>
    </div>
  `).join('');
}

function clearHistory() {
  state.history = [];
  state.sessionStats = { total: 0, fake: 0, real: 0, totalConf: 0 };
  localStorage.removeItem('predHistory');
  renderHistory();
  updateDashboard();
  updateHeroStats();
  showToast('🗑 History cleared');
}

// ── Dashboard ────────────────────────────────────────────────────────────────
async function refreshStats() {
  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();
    document.getElementById('statTotal').textContent = data.total_predictions ?? 0;
    document.getElementById('statFake').textContent  = data.fake_count ?? 0;
    document.getElementById('statReal').textContent  = data.real_count ?? 0;
    document.getElementById('statAvgConf').textContent =
      data.total_predictions ? `${parseFloat(data.avg_confidence).toFixed(1)}%` : '—';

    document.getElementById('heroTotal').textContent = data.total_predictions ?? 0;
    updatePieChart(data.fake_count || 0, data.real_count || 0);
  } catch (e) {
    updateDashboard();
  }
}

function updateDashboard() {
  const s = state.sessionStats;
  document.getElementById('statTotal').textContent = s.total;
  document.getElementById('statFake').textContent  = s.fake;
  document.getElementById('statReal').textContent  = s.real;
  const avg = s.total ? (s.totalConf / s.total).toFixed(1) : '—';
  document.getElementById('statAvgConf').textContent = s.total ? `${avg}%` : '—';
  updatePieChart(s.fake, s.real);
}

function updateHeroStats() {
  document.getElementById('heroTotal').textContent = state.sessionStats.total;
}

// ── Pie Chart ─────────────────────────────────────────────────────────────────
function initPieChart() {
  const ctx = document.getElementById('pieChart');
  if (!ctx) return;

  state.pieChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Fake', 'Real'],
      datasets: [{
        data: [0, 0],
        backgroundColor: ['rgba(244,63,94,0.8)', 'rgba(16,185,129,0.8)'],
        borderColor:     ['rgba(244,63,94,1)',   'rgba(16,185,129,1)'],
        borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: '70%',
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(13,15,28,0.95)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#f0f2ff',
          bodyColor: '#8892b0',
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.raw} articles`
          }
        }
      },
      animation: { animateScale: true, animateRotate: true, duration: 600 }
    }
  });
}

function updatePieChart(fake, real) {
  if (!state.pieChart) return;
  state.pieChart.data.datasets[0].data = [fake, real];
  state.pieChart.update();
}

// ── Batch Analyzer ───────────────────────────────────────────────────────────
async function runBatch() {
  const raw   = document.getElementById('batchInput').value.trim();
  const model = document.getElementById('batchModel').value;

  if (!raw) { showToast('⚠️ Please enter some texts first.'); return; }

  const texts = raw.split('\n').map(t => t.trim()).filter(Boolean);
  if (texts.length === 0) { showToast('⚠️ No valid lines found.'); return; }
  if (texts.length > 50) { showToast('⚠️ Maximum 50 texts per batch.'); return; }

  const btn    = document.getElementById('batchBtnText');
  const resDiv = document.getElementById('batchResults');

  btn.textContent = '⏳ Analyzing…';
  resDiv.classList.add('hidden');
  resDiv.innerHTML = '';

  try {
    const res  = await fetch('/api/batch', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ texts, model }),
    });
    const data = await res.json();

    if (!res.ok) { showToast(`❌ ${data.error}`); return; }

    const { results, summary } = data;

    // Summary bar
    const summaryEl = document.createElement('div');
    summaryEl.className = 'batch-summary';
    summaryEl.innerHTML = `
      📊 Analyzed <strong>${summary.total}</strong> articles —
      <span style="color:var(--fake-color)">🚫 ${summary.fake} Fake</span> /
      <span style="color:var(--real-color)">✅ ${summary.real} Real</span>
      ${summary.errors ? ` / ❌ ${summary.errors} errors` : ''}
    `;
    resDiv.appendChild(summaryEl);

    results.forEach(r => {
      const item = document.createElement('div');
      const cls  = (r.label || '').toLowerCase();
      item.className = `batch-result-item ${cls}`;
      item.innerHTML = `
        <span class="br-badge ${cls}">${r.label || '?'}</span>
        <span class="br-text" title="${escHtml(r.text_preview || '')}">${escHtml((r.text_preview || r.error || '—').slice(0, 120))}</span>
        <span class="br-conf">${r.confidence ? r.confidence.toFixed(0) + '%' : ''}</span>
      `;
      resDiv.appendChild(item);
    });

    resDiv.classList.remove('hidden');
    await refreshStats();

  } catch (err) {
    showToast('❌ Batch request failed.');
    console.error(err);
  } finally {
    btn.textContent = '🔍 Analyze All';
  }
}

// ── Share Result ─────────────────────────────────────────────────────────────
function shareResult() {
  const label = document.getElementById('verdictLabel').textContent;
  const conf  = document.getElementById('gaugePct').textContent;
  const text  = `I just analyzed a news article with TruthLens AI!\n\nVerdict: ${label}\nConfidence: ${conf}\n\nTry it yourself: ${window.location.href}`;
  if (navigator.share) {
    navigator.share({ title: 'TruthLens Result', text }).catch(() => {});
  } else {
    navigator.clipboard.writeText(text).then(() => showToast('📋 Result copied to clipboard!'));
  }
}

function copyResult() {
  const label = document.getElementById('verdictLabel').textContent;
  const conf  = document.getElementById('gaugePct').textContent;
  const model = document.getElementById('modelName').textContent;
  const out   = `TruthLens AI\nVerdict: ${label}\nConfidence: ${conf}\nModel: ${model}`;
  navigator.clipboard.writeText(out)
    .then(() => showToast('📋 Copied to clipboard!'))
    .catch(() => showToast('❌ Could not copy'));
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let _toastTimer = null;
function showToast(msg, duration = 2500) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => toast.classList.remove('show'), duration);
}

// ── Particles ─────────────────────────────────────────────────────────────────
function initParticles() {
  const canvas = document.getElementById('particleCanvas');
  if (!canvas) return;
  const ctx    = canvas.getContext('2d');
  const COUNT  = 60;
  let W, H, particles;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function createParticles() {
    particles = Array.from({ length: COUNT }, () => ({
      x:   Math.random() * W,
      y:   Math.random() * H,
      vx:  (Math.random() - 0.5) * 0.4,
      vy:  (Math.random() - 0.5) * 0.4,
      r:   Math.random() * 2 + 0.5,
      alpha: Math.random() * 0.5 + 0.1,
      color: Math.random() > 0.5 ? '79,142,247' : '139,92,246',
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Draw connections
    for (let i = 0; i < COUNT; i++) {
      for (let j = i + 1; j < COUNT; j++) {
        const dx   = particles[i].x - particles[j].x;
        const dy   = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(79,142,247,${0.15 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.6;
          ctx.stroke();
        }
      }
    }

    // Draw dots
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W;
      if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H;
      if (p.y > H) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color},${p.alpha})`;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  resize();
  createParticles();
  draw();

  window.addEventListener('resize', () => { resize(); createParticles(); }, { passive: true });
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
