/* GraphDefect-KG — results page controller */
(function () {
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const id = params.get("id");

  const GATE_COLORS = {
    cnn_global: "#2b6cb0", cnn_patch: "#4299e1", handcrafted: "#d69e2e",
    gcn: "#805ad5", gnn: "#dd6b20", kg: "#00897b",
  };
  const GATE_LABELS = {
    cnn_global: "CNN global", cnn_patch: "CNN patch", handcrafted: "Handcrafted",
    gcn: "GCN", gnn: "GNN", kg: "Knowledge",
  };

  async function boot() {
    let result = Store.load();
    if (!result || (id && result.prediction_id !== id)) {
      if (!id) return fail("No prediction id provided.");
      try { result = await API.getPrediction(id); } catch (e) { return fail(e.message); }
    }
    render(result);
    try {
      const graph = await API.getGraph(result.prediction_id);
      renderGraph(graph);
    } catch (e) {
      console.warn("graph load failed", e);
    }
    $("loading").classList.remove("show");
    $("resultRoot").style.display = "block";
  }

  function fail(msg) {
    $("loading").classList.remove("show");
    const box = $("errorBox");
    box.textContent = "Could not load results: " + msg;
    box.classList.add("show");
  }

  function render(r) {
    $("predClass").textContent = r.predicted_class;
    $("predConf").textContent = (r.confidence * 100).toFixed(1) + "% confidence";
    $("resultImg").src = r.image_url;
    $("imgCaption").textContent = r.filename;
    $("sourceTag").innerHTML =
      `Prediction source: <code>${r.prediction_source}</code> · graph: ` +
      `${r.graph_summary.nodes} nodes / ${r.graph_summary.edges} edges`;

    if (r.untrained_notice) {
      const n = $("modelNotice");
      n.style.display = "block";
      n.textContent = r.model_trained
        ? "The deep hybrid graph model is untrained; the class shown is from the fitted MobileNetV2+KNN baseline. The GCN/GNN outputs below are illustrative until trained (see README)."
        : "WARNING: no trained model available — this prediction is from an untrained network and is NOT a valid result.";
    }

    // probability bars
    const probs = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]);
    $("probBars").innerHTML = probs.map(([name, p]) => `
      <div class="prob-row ${name === r.predicted_class ? "top" : ""}">
        <span class="name">${name}</span>
        <span class="prob-track"><span class="prob-fill" style="width:${(p * 100).toFixed(1)}%"></span></span>
        <span class="pct">${(p * 100).toFixed(1)}%</span>
      </div>`).join("");

    $("explanation").textContent = r.explanation;

    // evidence chips
    const sup = (r.reasoning.supporting_evidence || []).map(
      (e) => `<span class="chip support">✓ ${e.property} (${e.activation})</span>`);
    const con = (r.reasoning.contradicting_evidence || []).map(
      (e) => `<span class="chip contra">✗ ${e.property} (${e.activation})</span>`);
    $("evidenceChips").innerHTML = sup.concat(con).join("") || "<span class='chip'>None</span>";

    // knowledge graph path
    $("kgPath").innerHTML = (r.reasoning.knowledge_graph_path || []).map((t) => `
      <li><span class="obj">${t.subject}</span>
        <span class="rel">${t.relation.replace(/_/g, " ")}</span>
        <span class="obj">${t.object}</span>
        ${t.support != null ? `<span class="support-badge">support ${t.support}</span>` : ""}
      </li>`).join("");

    // comparison table
    $("compareBody").innerHTML = (r.model_comparison || []).map((m) => `
      <tr>
        <td>${m.model}</td>
        <td>${m.predicted_class}</td>
        <td>${(m.confidence * 100).toFixed(0)}%</td>
        <td>${m.trained
          ? '<span class="tag-trained">trained</span>'
          : '<span class="tag-untrained">untrained</span>'}</td>
      </tr>`).join("");

    // fusion gates
    renderGates(r.component_gates || {});

    // important regions
    $("regionsList").innerHTML = (r.important_regions || []).map((rg) => `
      <div class="detail-row">
        <span class="k">Patch ${rg.patch} · ${rg.dominant_property}</span>
        <span class="v">${(rg.importance * 100).toFixed(0)}%</span>
      </div>`).join("");

    // download
    $("downloadBtn").addEventListener("click", () => {
      window.location.href = API.downloadReportUrl(r.prediction_id);
    });
  }

  function renderGates(gates) {
    const entries = Object.entries(gates);
    const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
    $("gateBar").innerHTML = entries.map(([k, v]) =>
      `<span class="gate-seg" style="width:${(v / total * 100).toFixed(1)}%;background:${GATE_COLORS[k] || "#ccc"}" title="${GATE_LABELS[k]}: ${v}"></span>`).join("");
    $("gateLegend").innerHTML = entries.map(([k, v]) =>
      `<span style="--dot:${GATE_COLORS[k] || "#ccc"}">${GATE_LABELS[k] || k} ${(v * 100).toFixed(0)}%</span>`).join("");
  }

  // ---- graph + side panel ----
  function renderGraph(graph) {
    $("graphCounts").textContent = `${graph.counts.nodes} nodes · ${graph.counts.edges} edges`;
    const container = $("cy");
    GraphView.init(container, graph, showDetails);
    GraphView.fit();
    renderCypher(GraphView.cypher());
    buildLegend(graph.legend, graph.node_types, graph.edge_types);
    buildFilters(graph.node_types, graph.edge_types);
    wireToolbar();
  }

  // Neo4j-style Cypher caption with light syntax highlighting.
  function renderCypher(text) {
    const bar = $("cypherBar");
    if (!bar || !text) return;
    // escape, then highlight relationship/property groups before bare labels
    const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const html = esc
      .replace(/(\[[^\]]*\])/g, '<span class="rel">$1</span>')
      .replace(/(\{[^}]*\})/g, '<span class="prop">$1</span>')
      .replace(/(^|[\s(])(:[A-Z][A-Za-z_]+)/g, '$1<span class="nd">$2</span>')
      .replace(/\b(MATCH|RETURN|LIMIT|WHERE|AND)\b/g, '<span class="kw">$1</span>')
      .replace(/(\b\d+\b)(?![^<]*<\/span>)/g, '<span class="num">$1</span>');
    bar.innerHTML = html;
  }

  function showDetails(kind, data) {
    const body = $("sideBody");
    $("sideTitle").textContent = kind === "node" ? "Node details" : "Edge details";
    if (kind === "node") {
      body.innerHTML = renderNodeDetail(data);
    } else {
      body.innerHTML = renderEdgeDetail(data);
    }
  }

  function row(k, v, mono) {
    return `<div class="detail-row"><span class="k">${k}</span>
      <span class="v ${mono ? "mono" : ""}">${v}</span></div>`;
  }

  function renderNodeDetail(d) {
    const meta = d.meta || {};
    let html = `<span class="detail-type" style="background:${d.color}">${d.ntype.replace(/_/g, " ")}</span>`;
    html += row("ID", d.id, true);
    html += row("Label", d.label);
    html += row("Importance", (d.importance * 100).toFixed(1) + "%");
    html += row("Supports prediction",
      d.supports ? '<span class="support-yes">yes</span>' : '<span class="support-no">no</span>');
    if (meta.semantic) html += `<div style="margin:10px 0;color:var(--text-muted)">${meta.semantic}</div>`;
    if (meta.confidence != null) html += row("Confidence", (meta.confidence * 100).toFixed(1) + "%");
    if (meta.probability != null) html += row("Probability", (meta.probability * 100).toFixed(1) + "%");
    if (meta.activation != null) html += row("Activation", meta.activation);
    if (meta.dominant_property) html += row("Dominant cue", meta.dominant_property);
    if (meta.bbox) html += row("Region bbox",
      `[${meta.bbox.x0},${meta.bbox.y0},${meta.bbox.x1},${meta.bbox.y1}]`, true);
    if (meta.features) {
      html += `<div class="section-title" style="margin:12px 0 4px">Feature values</div><ul class="detail-list">`;
      html += Object.entries(meta.features).map(([k, v]) => `<li>${k}: ${v}</li>`).join("");
      html += `</ul>`;
    }
    // connected nodes
    const cy = GraphView.cy;
    if (cy) {
      const n = cy.getElementById(d.id);
      const neigh = n.connectedEdges().map((e) => {
        const other = e.source().id() === d.id ? e.target() : e.source();
        return `${other.data("label")} (${e.data("relation").replace(/_/g, " ")})`;
      });
      if (neigh.length) {
        html += `<div class="section-title" style="margin:12px 0 4px">Connections (${neigh.length})</div><ul class="detail-list">`;
        html += neigh.slice(0, 12).map((x) => `<li>${x}</li>`).join("");
        html += `</ul>`;
      }
    }
    return html;
  }

  function renderEdgeDetail(d) {
    const meta = d.meta || {};
    let html = `<span class="detail-type" style="background:${d.color}">${d.relation.replace(/_/g, " ")}</span>`;
    html += row("Source", d.source, true);
    html += row("Target", d.target, true);
    html += row("Relation", d.relation.replace(/_/g, " "));
    html += row("Weight", d.weight);
    if (meta.similarity != null) html += row("Similarity", meta.similarity);
    if (meta.attention != null) html += row("Attention", meta.attention);
    if (meta.contribution != null) html += row("Contribution", meta.contribution);
    return html;
  }

  function buildLegend(legend, nodeTypes, edgeTypes) {
    const nl = $("nodeLegend"), el = $("edgeLegend");
    nl.innerHTML = legend.nodes
      .filter((n) => nodeTypes.includes(n.type))
      .map((n) => `<div class="legend-item"><span class="legend-swatch" style="background:${n.color}"></span>${n.label}</div>`)
      .join("");
    el.innerHTML = legend.edges
      .filter((e) => edgeTypes.includes(e.relation))
      .map((e) => `<div class="legend-item"><span class="legend-line" style="border-top-color:${e.color};border-top-style:${e.style}"></span>${e.label}</div>`)
      .join("");
  }

  function buildFilters(nodeTypes, edgeTypes) {
    const nf = $("nodeTypeFilter"), ef = $("edgeTypeFilter");
    nodeTypes.forEach((t) => nf.insertAdjacentHTML("beforeend",
      `<option value="${t}">${t.replace(/_/g, " ")}</option>`));
    edgeTypes.forEach((t) => ef.insertAdjacentHTML("beforeend",
      `<option value="${t}">${t.replace(/_/g, " ")}</option>`));
    nf.addEventListener("change", () => GraphView.filterNodeType(nf.value));
    ef.addEventListener("change", () => GraphView.filterEdgeType(ef.value));
  }

  function wireToolbar() {
    document.querySelectorAll(".graph-toolbar .tool").forEach((btn) => {
      btn.addEventListener("click", () => {
        const act = btn.dataset.act;
        if (act === "fit") GraphView.fit();
        else if (act === "reset") GraphView.reset();
        else if (act === "path") GraphView.highlightPath();
        else if (act === "important") GraphView.highlightImportant();
        else if (act === "expand") GraphView.expandNeighbours();
      });
    });
    let timer;
    $("nodeSearch").addEventListener("input", (e) => {
      clearTimeout(timer);
      timer = setTimeout(() => GraphView.search(e.target.value.trim()), 220);
    });
    $("sideClose").addEventListener("click", () => {
      $("sideBody").innerHTML = '<div class="side-empty">Click any node or edge to inspect its contribution.</div>';
      $("sideTitle").textContent = "Details";
      GraphView.clearHighlight();
    });
  }

  boot();
})();
