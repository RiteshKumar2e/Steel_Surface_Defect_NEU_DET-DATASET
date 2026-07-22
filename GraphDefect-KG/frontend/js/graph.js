/* GraphDefect-KG — interactive Cytoscape graph view */
const GraphView = (() => {
  let cy = null;
  let payload = null;
  let onSelect = null;

  // ---- Neo4j-style helpers ----
  // Mix a hex colour toward white to produce a lighter "ring" (Neo4j look).
  function lighten(hex, amt) {
    hex = (hex || "#718096").replace("#", "");
    if (hex.length === 3) hex = hex.split("").map((c) => c + c).join("");
    const r = parseInt(hex.slice(0, 2), 16), g = parseInt(hex.slice(2, 4), 16),
      b = parseInt(hex.slice(4, 6), 16);
    const mix = (c) => Math.round(c + (255 - c) * amt);
    return `rgb(${mix(r)},${mix(g)},${mix(b)})`;
  }

  // Node diameter — bigger so labels sit inside the circle (Neo4j caption style).
  function nodeSize(ele) {
    const imp = ele.data("importance") || 0;
    const type = ele.data("ntype");
    if (type === "defect_class") return 34 + imp * 44;        // sized by probability
    const base = { image: 74, prediction: 78, patch: 42 }[type] || 52;
    return base + imp * 16;
  }

  function styles() {
    return [
      {
        selector: "node",
        style: {
          // Neo4j: every node is a coloured circle, distinguished by colour + label.
          "background-color": "data(color)",
          "background-opacity": 0.95,
          "shape": "ellipse",
          "label": "data(label)",
          "font-size": (ele) => (ele.data("ntype") === "prediction" ? 12 : 9.5),
          "font-weight": 600,
          "min-zoomed-font-size": 5,
          "color": "#ffffff",
          "text-valign": "center",
          "text-halign": "center",
          "text-wrap": "wrap",
          "text-max-width": (ele) => nodeSize(ele) - 8,
          "text-outline-color": "data(color)",
          "text-outline-width": 1.4,
          "text-outline-opacity": 0.85,
          "width": nodeSize,
          "height": nodeSize,
          // lighter same-hue ring, like a Neo4j graph node
          "border-width": 3,
          "border-color": (ele) => lighten(ele.data("color"), 0.5),
          "border-opacity": 1,
          "overlay-opacity": 0,
        },
      },
      // The inspected / prediction node gets an extra-bright ring (Neo4j "target").
      {
        selector: 'node[ntype="prediction"]',
        style: { "border-width": 5, "border-color": (ele) => lighten(ele.data("color"), 0.6) },
      },
      {
        selector: "node[?supports]",
        style: { "border-color": (ele) => lighten(ele.data("color"), 0.55) },
      },
      {
        selector: "edge",
        style: {
          "width": (ele) => 1.2 + (ele.data("weight") || 0.3) * 3,
          "line-color": "data(color)",
          "line-style": (ele) => ele.data("line_style") || "solid",
          "target-arrow-color": "data(color)",
          "target-arrow-shape": "triangle",
          "arrow-scale": 0.8,
          "curve-style": "bezier",
          "opacity": 0.55,
          // Neo4j-style relationship caption on each edge
          "label": (ele) => (ele.data("relation") || "").replace(/_/g, " ").toUpperCase(),
          "font-size": 7.5,
          "color": "#64748b",
          "text-rotation": "autorotate",
          "text-background-color": "#eef2f8",
          "text-background-opacity": 0.9,
          "text-background-padding": 1,
          "min-zoomed-font-size": 7,
        },
      },
      { selector: ".faded", style: { "opacity": 0.1, "text-opacity": 0.08 } },
      {
        selector: ".highlight",
        style: { "opacity": 1, "text-opacity": 1, "border-color": "#dd6b20",
                 "border-width": 4, "line-color": "#dd6b20",
                 "target-arrow-color": "#dd6b20", "z-index": 999 },
      },
      { selector: ":selected", style: { "border-color": "#1d4ed8", "border-width": 5 } },
    ];
  }

  // Build a Neo4j-style Cypher caption for the current prediction graph.
  function cypherFor(data) {
    const nodes = data.elements.filter((e) => e.group === "nodes");
    const pred = nodes.find((n) => n.data.ntype === "prediction");
    const clsName = pred ? (pred.data.label || "Defect").replace("Prediction: ", "") : "Defect";
    const n = nodes.length;
    return `MATCH (d:Defect {class:'${clsName}'})-[r:HAS_PROPERTY|ASSOCIATED_WITH|PREDICTS*1..2]-(n) RETURN d,r,n LIMIT ${n}`;
  }

  function init(container, data, selectHandler) {
    payload = data;
    onSelect = selectHandler;
    cy = cytoscape({
      container,
      elements: data.elements,
      style: styles(),
      layout: layoutConfig(),
      wheelSensitivity: 0.25,
      minZoom: 0.2,
      maxZoom: 3,
    });

    cy.on("tap", "node", (e) => onSelect && onSelect("node", e.target.data(), e.target));
    cy.on("tap", "edge", (e) => onSelect && onSelect("edge", e.target.data(), e.target));
    cy.on("tap", (e) => { if (e.target === cy) clearHighlight(); });
    return cy;
  }

  function layoutConfig() {
    return {
      name: "cose",
      animate: false,
      nodeRepulsion: 22000,     // more spread -> Neo4j-like radial layout
      idealEdgeLength: 130,
      edgeElasticity: 0.35,
      gravity: 0.5,
      numIter: 1500,
      nodeOverlap: 24,
      padding: 40,
    };
  }

  // --- controls ---
  const fit = () => cy && cy.fit(null, 40);
  const reset = () => { if (cy) { clearHighlight(); cy.layout(layoutConfig()).run(); cy.fit(null, 40); } };

  function clearHighlight() {
    if (!cy) return;
    cy.elements().removeClass("faded highlight");
  }

  function highlightImportant() {
    if (!cy) return;
    clearHighlight();
    const nodes = cy.nodes().filter((n) => (n.data("importance") || 0) >= 0.55
      || ["prediction", "defect_evidence"].includes(n.data("ntype")));
    const nb = nodes.closedNeighborhood();
    cy.elements().addClass("faded");
    nb.removeClass("faded").addClass("highlight");
  }

  function highlightPath() {
    if (!cy) return;
    clearHighlight();
    // prediction path: supporting evidence/patches -> prediction -> predicted class -> knowledge
    const pred = cy.getElementById("prediction");
    if (pred.empty()) return;
    let path = pred.closedNeighborhood();
    // include predicted class node + its knowledge/cause neighbours
    const predClass = cy.nodes(`[ntype="defect_class"][?meta]`).filter(
      (n) => n.data("meta") && n.data("meta").is_prediction);
    path = path.union(predClass.closedNeighborhood());
    // supporting evidence
    const support = cy.nodes(`[ntype="defect_evidence"][?supports]`);
    path = path.union(support.closedNeighborhood());
    cy.elements().addClass("faded");
    path.removeClass("faded").addClass("highlight");
    cy.fit(path, 50);
  }

  function expandNeighbours() {
    if (!cy) return;
    const sel = cy.$(":selected");
    if (sel.empty()) return;
    clearHighlight();
    const nb = sel.closedNeighborhood();
    cy.elements().addClass("faded");
    nb.removeClass("faded").addClass("highlight");
  }

  function search(term) {
    if (!cy) return;
    clearHighlight();
    if (!term) return;
    const t = term.toLowerCase();
    const matched = cy.nodes().filter((n) => n.data("label").toLowerCase().includes(t));
    if (matched.empty()) return;
    cy.elements().addClass("faded");
    matched.closedNeighborhood().removeClass("faded").addClass("highlight");
    cy.fit(matched, 80);
  }

  function filterNodeType(type) {
    if (!cy) return;
    clearHighlight();
    if (!type) { cy.elements().removeClass("faded"); return; }
    cy.elements().addClass("faded");
    const nodes = cy.nodes(`[ntype="${type}"]`);
    nodes.removeClass("faded");
    nodes.connectedEdges().removeClass("faded");
  }

  function filterEdgeType(rel) {
    if (!cy) return;
    clearHighlight();
    if (!rel) { cy.elements().removeClass("faded"); return; }
    cy.elements().addClass("faded");
    const edges = cy.edges(`[relation="${rel}"]`);
    edges.removeClass("faded");
    edges.connectedNodes().removeClass("faded");
  }

  function focusNode(id) {
    if (!cy) return;
    const n = cy.getElementById(id);
    if (n.nonempty()) { n.select(); cy.animate({ center: { eles: n }, zoom: 1.4 }, { duration: 300 }); }
  }

  return {
    init, fit, reset, highlightImportant, highlightPath, expandNeighbours,
    search, filterNodeType, filterEdgeType, focusNode, clearHighlight,
    cypher: () => (payload ? cypherFor(payload) : ""),
    get cy() { return cy; },
  };
})();
