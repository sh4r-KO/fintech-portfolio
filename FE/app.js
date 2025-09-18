// ---------- tiny fetch helper ----------
async function fetchJSON(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const msg = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status} ${res.statusText} — ${msg}`);
  }
  return res.json();
}

// ---------- card template ----------
function renderCard(p) {
  let link = '';
  // map project titles/slugs to their page
  if (p.slug === 'compound') link = 'compound.html';
  else if (p.slug === 'stocks') link = 'stocks.html';
  else if (p.slug === 'forex') link = 'forex.html';
  else if (p.slug === 'backTradeTest') link = 'backTradeTest.html';
  else link = 'project.html?slug=' + encodeURIComponent(p.slug);

  return `
    <a class="card" href="${link}">
      <div class="card-media" style="background-image:url('${p.cover_image || ''}')"></div>
      <div class="card-body">
        <h3>${p.title}</h3>
        <p class="muted">${p.summary || ''}</p>
      </div>
    </a>
  `;
}


// ---------- home page renderer ----------
async function renderHome() {
  const grid = document.getElementById('projects');
  if (!grid) return; // not on index.html

  const tagBar = document.getElementById('tags');
  const searchInput = document.getElementById('search');

  // state
  let all = [];
  let activeTag = '';
  let query = '';

  // fetch projects
  try {
    const payload = await fetchJSON(`${window.API_BASE}/api/projects`);
    // support either {projects:[...]} or [...]
    all = Array.isArray(payload) ? payload : (payload.projects || []);
  } catch (err) {
    console.error('Failed to load projects:', err);
    grid.innerHTML = `<div class="muted">Couldn’t load projects. Check API_BASE or CORS.</div>`;
    return;
  }

  // build tag list
  const tagSet = new Set();
  all.forEach(p => {
    (p.tags || []).forEach(t => tagSet.add(t));
    (p.tech || []).forEach(t => tagSet.add(t));
  });

  tagBar.innerHTML = ['All', ...[...tagSet].sort()]
    .map((t, i) => `<button type="button" class="chip ${i === 0 ? 'chip-active' : ''}">${t}</button>`)
    .join('');
  activeTag = 'All';

  function apply() {
    query = (searchInput?.value || '').toLowerCase().trim();

    const filtered = all.filter(p => {
      const hay = [
        p.title || '',
        p.summary || '',
        ...(p.tags || []),
        ...(p.tech || []),
      ].join(' ').toLowerCase();

      const matchesQuery = !query || hay.includes(query);
      const matchesTag = activeTag === 'All' || (p.tags || []).includes(activeTag) || (p.tech || []).includes(activeTag);
      return matchesQuery && matchesTag;
    });

    grid.innerHTML = filtered.length
      ? filtered.map(renderCard).join('')
      : `<div class="muted">No projects match your search.</div>`;
  }

  // events
  tagBar.addEventListener('click', (e) => {
    if (!(e.target instanceof HTMLElement)) return;
    if (!e.target.classList.contains('chip')) return;
    [...tagBar.children].forEach(c => c.classList.remove('chip-active'));
    e.target.classList.add('chip-active');
    activeTag = e.target.textContent || 'All';
    apply();
  });

  if (searchInput) {
    searchInput.addEventListener('input', apply);
  }


  
  // first paint
  apply();


}

// ---------- contact form ----------
(function initContact() {
  const form = document.getElementById('contactForm');
  if (!form) return;
  const status = document.getElementById('contactStatus');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    status.textContent = 'Sending...';
    const data = Object.fromEntries(new FormData(form).entries());
    try {
      const res = await fetch(`${window.API_BASE}/api/contact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      if (!res.ok) throw new Error('Failed');
      status.textContent = 'Thanks! I\'ll get back to you.';
      form.reset();
    } catch (err) {
      status.textContent = 'Error sending message. Please try again later.';
    }
  });
})();

// ---------- project detail page ----------
async function renderProject() {
  const mount = document.getElementById('project');
  if (!mount) return; // not on project.html
  const params = new URLSearchParams(location.search);
  const slug = params.get('slug');
  if (!slug) { mount.textContent = 'Missing project slug.'; return; }
  try {
    const p = await fetchJSON(`${window.API_BASE}/api/projects/${encodeURIComponent(slug)}`);
    mount.innerHTML = `
<article class="article">
  <h1>${p.title}</h1>
  <p class="muted">${p.summary || ''}</p>
  ${p.cover_image ? `<img class='banner' src='${p.cover_image}' alt='${p.title} cover'/>` : ''}
  <div class="chips">
    ${[...(p.tags || []), ...(p.tech || [])].map(t => `<span class='chip chip-small'>${t}</span>`).join('')}
  </div>
  ${p.problem ? `<h2>Problem</h2><p>${p.problem}</p>` : ''}
  ${p.approach ? `<h2>Approach</h2><p>${p.approach}</p>` : ''}
  ${p.results ? `<h2>Results</h2><p>${p.results}</p>` : ''}
  ${p.metrics ? `<h3>Key Metrics</h3>
    <ul class='metrics'>
      ${p.metrics.rows_processed ? `<li><strong>Rows:</strong> ${Number(p.metrics.rows_processed).toLocaleString()}</li>` : ''}
      ${p.metrics.latency_ms ? `<li><strong>P95 Latency:</strong> ${p.metrics.latency_ms} ms</li>` : ''}
      ${p.metrics.cost_savings_pct ? `<li><strong>Cost Savings:</strong> ${p.metrics.cost_savings_pct}%</li>` : ''}
    </ul>` : ''}
  ${(p.links && (p.links.github || p.links.demo || p.links.paper)) ? `<p class='links'>
      ${p.links.github ? `<a target='_blank' rel='noopener' href='${p.links.github}'>GitHub</a>` : ''}
      ${p.links.demo ? `<a target='_blank' rel='noopener' href='${p.links.demo}'>Live demo</a>` : ''}
      ${p.links.paper ? `<a target='_blank' rel='noopener' href='${p.links.paper}'>Paper</a>` : ''}
    </p>` : ''}
</article>`;
  } catch (err) {
    console.error(err);
    mount.textContent = 'Project not found.';
  }
}

// ---------- boot ----------
renderHome();
renderProject();



// app.js
function api(base, path) {
  const trimmed = (base || "").replace(/\/+$/, "");
  return `${trimmed}${path}`;
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("backtestForm");
  const out  = document.getElementById("out");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = {
      // Match exactly what your backend expects:
      symbol: form.symbol.value.trim(),
      initial_cash: Number(form.cash.value),
      // add other params needed by your endpoint…
    };

    try {
      const res = await fetch(api(window.API_BASE, "/api/backtest"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      // If CORS is misconfigured, you’ll see a network error before this line.
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} – ${text || "backtest failed"}`);
      }

      const data = await res.json();
      out.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      out.textContent = `Request error: ${err.message}`;
      console.error(err);
    }
  });
});
