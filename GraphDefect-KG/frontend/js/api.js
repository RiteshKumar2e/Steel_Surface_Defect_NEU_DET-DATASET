/* GraphDefect-KG — thin API client (vanilla JS, no framework) */
const API = (() => {
  const base = "";

  async function _json(url, opts) {
    const res = await fetch(base + url, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  }

  return {
    health: () => _json("/api/health"),
    classes: () => _json("/api/classes"),
    modelInfo: () => _json("/api/model-info"),
    predict: (file) => {
      const fd = new FormData();
      fd.append("file", file);
      return _json("/api/predict", { method: "POST", body: fd });
    },
    predictBatch: (files) => {
      const fd = new FormData();
      Array.from(files).forEach((f) => fd.append("files", f));
      return _json("/api/predict-batch", { method: "POST", body: fd });
    },
    getGraph: (id) => _json(`/api/graph/${id}`),
    getPrediction: (id) => _json(`/api/prediction/${id}`),
    downloadReportUrl: (id) => `/api/download-report/${id}`,
  };
})();

/* Small shared storage helper so the landing page can hand a result to the
   results page without re-uploading. */
const Store = {
  save(result) {
    try { sessionStorage.setItem("gdkg:last", JSON.stringify(result)); } catch (_) {}
  },
  load() {
    try { return JSON.parse(sessionStorage.getItem("gdkg:last")); } catch (_) { return null; }
  },
};
