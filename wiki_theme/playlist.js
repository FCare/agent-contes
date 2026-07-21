// Lecteur de playlist recueil : <audio> ne sait pas lire un .m3u directement,
// ce script le parse et enchaîne les pistes dans un unique lecteur natif (voir
// reference/build_wiki.py::_render_recueil_page pour le balisage attendu).
(function () {
  function parseM3u(text) {
    const lines = text.split(/\r?\n/);
    const entries = [];
    let pendingLabel = null;
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("#EXTINF:")) {
        const comma = trimmed.indexOf(",");
        pendingLabel = comma >= 0 ? trimmed.slice(comma + 1).trim() : null;
      } else if (trimmed && !trimmed.startsWith("#")) {
        entries.push({ label: pendingLabel || trimmed, url: trimmed });
        pendingLabel = null;
      }
    }
    return entries;
  }

  function setupPlayer(container) {
    const src = container.getAttribute("data-playlist");
    const audio = container.querySelector("audio");
    const nowPlaying = container.querySelector(".wiki-playlist-now");
    const prevBtn = container.querySelector(".wiki-playlist-prev");
    const nextBtn = container.querySelector(".wiki-playlist-next");
    if (!src || !audio) return;

    fetch(src)
      .then((r) => r.text())
      .then((text) => {
        const entries = parseM3u(text);
        if (!entries.length) {
          if (nowPlaying) nowPlaying.textContent = "Playlist vide.";
          return;
        }
        let index = 0;

        function load(i, autoplay) {
          if (i < 0 || i >= entries.length) return;
          index = i;
          audio.src = entries[i].url;
          if (nowPlaying) {
            nowPlaying.textContent = (i + 1) + "/" + entries.length + " — " + entries[i].label;
          }
          if (autoplay) audio.play().catch(() => {});
        }

        audio.addEventListener("ended", () => load(index + 1, true));
        if (prevBtn) prevBtn.addEventListener("click", () => load(index - 1, true));
        if (nextBtn) nextBtn.addEventListener("click", () => load(index + 1, true));

        load(0, false);
      })
      .catch(() => {
        if (nowPlaying) nowPlaying.textContent = "Playlist indisponible.";
      });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".wiki-playlist").forEach(setupPlayer);
  });
})();
