/* GraphDefect-KG — batch / folder analysis gallery */
(function () {
  const $ = (id) => document.getElementById(id);

  // Class colours (aligned with the defect legend).
  const CLASS_COLORS = {
    "Crazing": "#e53e3e", "Inclusion": "#38b2ac", "Patches": "#3182ce",
    "Pitted Surface": "#48bb78", "Rolled-in Scale": "#d69e2e", "Scratches": "#b83280",
  };
  const IMG_RE = /\.(jpg|jpeg|png|bmp)$/i;

  const fileInput = $("fileInput"), folderInput = $("folderInput");
  $("pickFiles").addEventListener("click", () => fileInput.click());
  $("pickFolder").addEventListener("click", () => folderInput.click());
  fileInput.addEventListener("change", (e) => run(e.target.files));
  folderInput.addEventListener("change", (e) => run(e.target.files));

  // model status notice
  API.health().then((h) => {
    if (!h.hybrid_trained) {
      const n = $("modelNotice"); n.style.display = "block";
      n.textContent = h.knn_baseline_available
        ? "Predictions come from the fitted MobileNetV2+KNN baseline (deep hybrid untrained)."
        : "No trained model available — outputs are NOT valid. See README.";
    }
  }).catch(() => {});

  async function run(fileList) {
    hideError();
    const files = Array.from(fileList).filter((f) => IMG_RE.test(f.name));
    if (!files.length) return showError("No JPG/PNG/BMP images found in your selection.");
    if (files.length > 200) return showError("Please select 200 images or fewer per batch.");

    $("gallery").innerHTML = "";
    $("summaryBar").classList.remove("show");
    const prog = $("progress"); prog.classList.add("show");
    $("progressText").textContent = `Analysing ${files.length} image(s) on CPU… this may take a moment.`;
    $("barFill").style.width = "12%";
    let t = 12;
    const tick = setInterval(() => { t = Math.min(90, t + 4); $("barFill").style.width = t + "%"; }, 500);

    try {
      const data = await API.predictBatch(files);
      clearInterval(tick); $("barFill").style.width = "100%";
      setTimeout(() => prog.classList.remove("show"), 400);
      renderSummary(data);
      renderGallery(data.results);
    } catch (err) {
      clearInterval(tick); prog.classList.remove("show");
      showError("Batch analysis failed: " + err.message);
    }
  }

  function renderSummary(data) {
    const bar = $("summaryBar");
    const dist = Object.entries(data.class_distribution)
      .filter(([, n]) => n > 0)
      .sort((a, b) => b[1] - a[1])
      .map(([c, n]) => `<span class="dist-chip" style="background:${CLASS_COLORS[c] || "#718096"}">${c}: ${n}</span>`)
      .join("");
    bar.innerHTML =
      `<span class="stat"><b>${data.succeeded}</b>/${data.count} analysed</span>` +
      (data.failed ? `<span class="stat">· <b>${data.failed}</b> failed</span>` : "") +
      `<span class="stat">· ${(data.elapsed_ms / 1000).toFixed(1)}s</span>` +
      `<span style="flex:1"></span>` + dist;
    bar.classList.add("show");
  }

  function renderGallery(results) {
    const gallery = $("gallery");
    results.forEach((r) => {
      const card = document.createElement("div");
      card.className = "gcard" + (r.error ? " err" : "");
      if (r.error) {
        card.innerHTML = `<div class="gbody"><b>${r.filename}</b><br>${r.error}</div>`;
        gallery.appendChild(card);
        return;
      }
      const color = CLASS_COLORS[r.predicted_class] || "#2b6cb0";
      card.style.setProperty("--cls", color);
      const props = (r.visual_properties || []).slice(0, 3)
        .map((p) => `<span class="p">${p}</span>`).join("");
      card.innerHTML = `
        <canvas width="230" height="230"></canvas>
        <div class="gbody">
          <div class="gclass">
            <span class="name">${r.predicted_class}</span>
            <span class="conf">${(r.confidence * 100).toFixed(0)}%</span>
          </div>
          <div class="gprops">${props}</div>
          <div class="greason">${r.reason}</div>
          <div class="gfname">${r.filename}</div>
        </div>`;
      gallery.appendChild(card);
      drawImage(card.querySelector("canvas"), r, color);
    });
  }

  function drawImage(canvas, r, color) {
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      const S = canvas.width;                 // square canvas matches square preprocessing
      ctx.drawImage(img, 0, 0, S, S);
      const size = r.image_size || 224;
      (r.important_regions || []).forEach((rg, i) => {
        const b = rg.bbox;
        const x = (b.x0 / size) * S, y = (b.y0 / size) * S;
        const w = ((b.x1 - b.x0) / size) * S, h = ((b.y1 - b.y0) / size) * S;
        ctx.strokeStyle = color;
        ctx.lineWidth = i === 0 ? 3 : 1.6;
        ctx.strokeRect(x, y, w, h);
        // label chip
        const label = `${(rg.importance * 100).toFixed(0)}%`;
        ctx.font = "10px sans-serif";
        const tw = ctx.measureText(label).width + 6;
        ctx.fillStyle = color;
        ctx.fillRect(x, Math.max(0, y - 13), tw, 13);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x + 3, Math.max(9, y - 3));
      });
    };
    img.onerror = () => { ctx.fillStyle = "#333"; ctx.fillRect(0, 0, canvas.width, canvas.height); };
    img.src = r.image_url;
  }

  function showError(m) { const e = $("errorBox"); e.textContent = m; e.classList.add("show"); }
  function hideError() { $("errorBox").classList.remove("show"); }
})();
