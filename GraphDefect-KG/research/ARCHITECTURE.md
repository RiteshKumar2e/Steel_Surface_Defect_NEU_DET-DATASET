# Architecture — GraphDefect-KG

## 1. Component-level architecture
```mermaid
flowchart TD
    A[Input Steel Image] --> B[Preprocessing]
    B --> C[MobileNetV2 Feature Extractor]
    B --> D[Handcrafted Feature Extractor]
    C --> E[Patch Embeddings]
    D --> F[Texture / Shape / Edge Features]
    E --> G[KNN Graph Builder]
    F --> G
    G --> H[GCN]
    G --> I[GNN / GAT]
    H --> J[Gated Feature Fusion]
    I --> J
    C --> J
    D --> J
    KG[Knowledge Graph Affinity] --> J
    J --> K[Six-Class Prediction]
    K --> L[Knowledge Graph Reasoner]
    G --> L
    L --> M[LLM-Inspired Explanation]
    M --> N[FastAPI Response]
    N --> O[Interactive Web Graph]
```

## 2. Model pipeline (data shapes)
```mermaid
flowchart LR
    IMG[224x224x3] --> CNN[MobileNetV2\n1280-d global]
    IMG --> PATCH[16 patches\n16x1280]
    PATCH --> PROJ[project\n16x128]
    IMG --> HAND[handcrafted\n16x32 patch, 32 global]
    PROJ --> NODE[node feats\n16x160]
    HAND --> NODE
    NODE --> KNN[KNN graph\nedge_index, edge_weight]
    KNN --> GCN[GCN\n256-d]
    KNN --> GAT[GAT\n256-d]
    CNN --> FUSE[Gated Fusion\n6x128]
    GCN --> FUSE
    GAT --> FUSE
    HAND --> FUSE
    KGV[KG affinity 6-d] --> FUSE
    FUSE --> OUT[6-class logits]
```

## 3. API flow
```mermaid
sequenceDiagram
    participant U as Browser
    participant F as FastAPI
    participant P as PredictionService
    participant M as Models + KG
    U->>F: POST /api/predict (image)
    F->>P: validate + decode
    P->>M: preprocess → features → KNN graph → GCN/GAT/hybrid
    M-->>P: logits, attention, gates
    P->>M: KG affinity + reasoning + explanation
    P->>P: build prediction graph → Cytoscape JSON
    P-->>F: result (+ persisted graph_{id}.json)
    F-->>U: JSON result
    U->>F: GET /api/graph/{id}
    F-->>U: Cytoscape elements
    U->>U: render interactive graph
```

## 4. Frontend ↔ backend flow
```mermaid
flowchart LR
    IDX[index.html\napp.js] -->|POST /api/predict| API[(FastAPI)]
    API -->|result JSON| SS[sessionStorage]
    SS --> RES[results.html\nprediction.js]
    RES -->|GET /api/graph/id| API
    RES --> CY[graph.js + Cytoscape.js]
    RES -->|GET /api/download-report/id| API
```

## 5. Graph creation flow
```mermaid
flowchart TD
    P[Patches] --> NF[Fused node features 160-d]
    NF --> KNN[KNN adjacency + weights]
    KNN --> RG[RegionGraph]
    RG --> PG[Prediction Graph builder]
    GF[Global features] --> PG
    PR[Prediction + probs] --> PG
    KGN[Knowledge Graph] --> PG
    ATT[GAT attention] --> PG
    PG --> CY[Cytoscape elements + legend]
```

## 6. Knowledge reasoning flow
```mermaid
flowchart LR
    FEAT[Interpretable features] --> ACT[Property activations]
    ACT --> AFF[Per-class affinity vector]
    PRED[Predicted class] --> PATH[KG reasoning path\ndefect→property→cause]
    ACT --> PATH
    PATH --> EXP[Template explanation\n(optional local LLM)]
    AFF --> EXP
```

## 7. Data schema
### Node schema
| field | type | meaning |
|-------|------|---------|
| id | str | unique node id |
| type | enum(NodeType) | image / patch / region / cnn_feature / texture_feature / shape_feature / edge_feature / defect_evidence / defect_class / prediction / knowledge_concept / cause_concept / explanation |
| label | str | display label |
| shape, color | str | rendering hints |
| importance / activation / probability | float | signal used for sizing |
| supports | bool | supports vs contradicts prediction |
| meta | dict | features, bbox, semantic text, connections |

### Edge schema
| field | type | meaning |
|-------|------|---------|
| id | str | `source__relation__target` |
| source, target | str | node ids |
| relation | enum(EdgeType) | knn_of / visually_similar_to / spatially_adjacent_to / contains / extracted_from / supports / contradicts / associated_with / predicts / belongs_to / has_pattern / has_property / has_texture / has_shape / has_orientation / has_confidence / semantically_related_to |
| weight | float | edge strength |
| color, line_style | str | rendering hints |
| meta | dict | similarity, attention, contribution |

### Result schema (API)
See `backend/api/schemas.py` (`PredictionResponse`): `predicted_class`, `confidence`,
`probabilities`, `prediction_source`, `model_trained`, `untrained_notice`, `reasoning`,
`explanation`, `visual_features`, `kg_affinity`, `component_gates`, `important_regions`,
`model_comparison`, `graph_summary`, `timings_ms`.
