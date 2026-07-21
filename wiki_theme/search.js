// Recherche sémantique du wiki : réutilise l'index Chroma déjà calculé par
// l'agent (voir wiki_api.py -> chroma_store.py) via une petite API JSON,
// plutôt que de dupliquer un index de recherche côté site statique.
(function () {
  function debounce(fn, delay) {
    let timer;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), delay);
    };
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s == null ? "" : String(s);
    return div.innerHTML;
  }

  function renderResults(container, data) {
    const stories = data.stories || [];
    const moments = data.moments || [];
    if (!stories.length && !moments.length) {
      container.innerHTML = "<p><em>Aucun résultat.</em></p>";
      return;
    }
    let html = "";
    if (stories.length) {
      html += "<h3>Histoires</h3><ul>";
      for (const s of stories) {
        html += `<li><a href="${escapeHtml(s.url)}">${escapeHtml(s.title)}</a>`;
        if (s.summary) html += ` — ${escapeHtml(s.summary)}`;
        html += "</li>";
      }
      html += "</ul>";
    }
    if (moments.length) {
      html += "<h3>Passages précis</h3><ul>";
      for (const m of moments) {
        html += `<li><a href="${escapeHtml(m.url)}">${escapeHtml(m.title)}</a> — ${escapeHtml(m.excerpt)}</li>`;
      }
      html += "</ul>";
    }
    container.innerHTML = html;
  }

  function runSearch(input, results) {
    const q = input.value.trim();
    if (!q) {
      results.innerHTML = "";
      return;
    }
    results.innerHTML = "<p><em>Recherche…</em></p>";
    fetch("/api/wiki/search?q=" + encodeURIComponent(q))
      .then((r) => r.json())
      .then((data) => renderResults(results, data))
      .catch(() => {
        results.innerHTML = "<p><em>Recherche indisponible pour le moment.</em></p>";
      });
  }

  document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("wiki-search-input");
    const results = document.getElementById("wiki-search-results");
    if (!input || !results) return;
    input.addEventListener("input", debounce(() => runSearch(input, results), 300));
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        runSearch(input, results);
      }
    });
  });
})();
