/* GraphDefect-KG — landing page logic: upload, preview, analyse */
(function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const previewWrap = document.getElementById("previewWrap");
  const previewImg = document.getElementById("previewImg");
  const previewMeta = document.getElementById("previewMeta");
  const analyseBtn = document.getElementById("analyseBtn");
  const loading = document.getElementById("loading");
  const errorBox = document.getElementById("errorBox");
  const classGrid = document.getElementById("classGrid");
  const modelNotice = document.getElementById("modelNotice");

  let currentFile = null;

  // --- populate class list + model status ---
  API.classes()
    .then((d) => {
      classGrid.innerHTML = d.classes
        .map(
          (c) => `<div class="class-pill"><b>${c.name}</b>
            <span>${(c.visual_properties || []).slice(0, 2).join(", ")}</span></div>`
        )
        .join("");
    })
    .catch(() => { classGrid.innerHTML = "<span class='small'>Could not load classes.</span>"; });

  API.health()
    .then((h) => {
      if (!h.hybrid_trained) {
        modelNotice.style.display = "block";
        modelNotice.textContent =
          h.knn_baseline_available
            ? "Note: the deep hybrid model is untrained; predictions come from the fitted MobileNetV2+KNN baseline. Train the hybrid model for research-grade results (see README)."
            : "Note: no trained models are available yet. Outputs are from an untrained network and are NOT valid results. See README to fit the baseline / train models.";
      }
    })
    .catch(() => {});

  // --- upload interactions ---
  dropzone.addEventListener("click", () => fileInput.click());
  ["dragover", "dragenter"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
  );
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  fileInput.addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) handleFile(f);
  });

  function handleFile(file) {
    hideError();
    const okTypes = ["image/jpeg", "image/png", "image/bmp"];
    if (!okTypes.includes(file.type) && !/\.(jpg|jpeg|png|bmp)$/i.test(file.name)) {
      showError("Unsupported file type. Please upload a JPG, PNG or BMP image.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError("File is larger than 10 MB.");
      return;
    }
    currentFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewMeta.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
    previewWrap.classList.add("show");
    analyseBtn.disabled = false;
  }

  analyseBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    hideError();
    analyseBtn.disabled = true;
    loading.classList.add("show");
    try {
      const result = await API.predict(currentFile);
      Store.save(result);
      window.location.href = `/results.html?id=${encodeURIComponent(result.prediction_id)}`;
    } catch (err) {
      loading.classList.remove("show");
      analyseBtn.disabled = false;
      showError("Analysis failed: " + err.message);
    }
  });

  function showError(msg) { errorBox.textContent = msg; errorBox.classList.add("show"); }
  function hideError() { errorBox.classList.remove("show"); }
})();
