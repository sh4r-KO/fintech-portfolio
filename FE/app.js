// Adjust to your backend host if deploying separately
if (e.target.classList.contains('chip')) {
    [...tagBar.children].forEach(c => c.classList.remove('chip-active'));
    e.target.classList.add('chip-active');
    activeTag = e.target.textContent;
    apply();
}
//});


searchInput.addEventListener('input', apply);
apply();


// contact form
const form = document.getElementById('contactForm');
if (form) {
    const status = document.getElementById('contactStatus');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        status.textContent = 'Sending...';
        const data = Object.fromEntries(new FormData(form).entries());
        try {
            const res = await fetch(`${API_BASE}/api/contact`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
            if (!res.ok) throw new Error('Failed');
            status.textContent = 'Thanks! I\'ll get back to you.';
            form.reset();
        } catch (err) {
            status.textContent = 'Error sending message. Please try again later.';
        }
    });
}
//}


async function renderProject() {
    const mount = document.getElementById('project');
    if (!mount) return;
    const params = new URLSearchParams(location.search);
    const slug = params.get('slug');
    if (!slug) { mount.textContent = 'Missing project slug.'; return; }
    try {
        const p = await fetchJSON(`${API_BASE}/api/projects/${encodeURIComponent(slug)}`);
        mount.innerHTML = `
<article class="article">
<h1>${p.title}</h1>
<p class="muted">${p.summary}</p>
${p.cover_image ? `<img class='banner' src='${p.cover_image}' alt='${p.title} cover'/>` : ''}
<div class="chips">${[...(p.tags || []), ...(p.tech || [])].map(t => `<span class='chip chip-small'>${t}</span>`).join('')}</div>
${p.problem ? `<h2>Problem</h2><p>${p.problem}</p>` : ''}
${p.approach ? `<h2>Approach</h2><p>${p.approach}</p>` : ''}
${p.results ? `<h2>Results</h2><p>${p.results}</p>` : ''}
${p.metrics ? `<h3>Key Metrics</h3>
<ul class='metrics'>
${p.metrics.rows_processed ? `<li><strong>Rows:</strong> ${p.metrics.rows_processed.toLocaleString()}</li>` : ''}
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
        mount.textContent = 'Project not found.';
    }
}


renderHome();
renderProject();