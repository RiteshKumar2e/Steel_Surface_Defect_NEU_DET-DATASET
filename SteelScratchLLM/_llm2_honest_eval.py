
# ----- cell: cell_01_setup -----
# ================================================================
# CELL 1 — IMPORTS: LOCAL SCRATCH LLM
# ================================================================
import os, cv2, json, math, time, random, warnings, re, hashlib
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from collections import Counter
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    TORCH_IMPORT_ERROR = e

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False
    print('skimage not found — GLCM features disabled')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if TORCH_OK:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

print('=' * 72)
print('  STEEL DEFECT DETECTION — LOCAL SCRATCH TINY LLM')
print('=' * 72)
print('The classifier is a small Transformer trained locally from scratch on your dataset.')
print('PyTorch available:', TORCH_OK)
if not TORCH_OK:
    print('Install PyTorch first: pip install torch torchvision torchaudio')
print('=' * 72)

# ----- cell: cell_02_config -----
# ================================================================
# CELL 2 — CONFIGURATION
# ================================================================
CLASS_NAMES = ['crazing', 'inclusion', 'patches',
               'pitted_surface', 'rolled-in_scale', 'scratches']

LABEL_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {i: c for c, i in LABEL_TO_ID.items()}

# Change this path according to your system if needed.
BASE_PATH       = r'C:\Users\anmol\OneDrive\Desktop\Steel_Surface_Defect_NEU_DET-DATASET\NEU-DET'
IMAGES_DIR      = os.path.join(BASE_PATH, 'IMAGES')
ANNOTATIONS_DIR = os.path.join(BASE_PATH, 'ANNOTATIONS')
OUTPUT_DIR      = os.path.join(BASE_PATH, 'OUTPUT_SCRATCH_LLM')
RESULTS_JSON    = os.path.join(OUTPUT_DIR, 'scratch_llm_results_all1800.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE     = 200
MODEL_NAME   = 'TinySteelLLM_FromScratch'
USE_LLM_AREA = False       # box localization is local XML/contour based, not API based
SAVE_EVERY   = 50
PRINT_EVERY  = 100

# Training settings for the from-scratch tiny Transformer classifier
TRAIN_SPLIT      = 0.80
BATCH_SIZE       = 32
EPOCHS           = 18
LEARNING_RATE    = 3e-4
MAX_LEN          = 160
EMBED_DIM        = 96
NUM_HEADS        = 4
NUM_LAYERS       = 2
DROPOUT          = 0.15
MIN_TOKEN_FREQ   = 1
MODEL_PATH       = os.path.join(OUTPUT_DIR, 'tiny_steel_llm_from_scratch.pt')
TOKENIZER_PATH   = os.path.join(OUTPUT_DIR, 'tiny_steel_llm_tokenizer.json')

# ---- Second from-scratch LLM: different architecture (BiLSTM + attention) ----
MODEL_NAME_2     = 'TinyScratchLLM_BiLSTM'
RESULTS_JSON_2   = os.path.join(OUTPUT_DIR, 'scratch_llm2_results_all1800.json')
MODEL_PATH_2     = os.path.join(OUTPUT_DIR, 'tiny_steel_llm2_bilstm_from_scratch.pt')
TOKENIZER_PATH_2 = os.path.join(OUTPUT_DIR, 'tiny_steel_llm2_tokenizer.json')
HIDDEN_DIM_2     = 192
NUM_LAYERS_2     = 2
DROPOUT_2        = 0.30
EPOCHS_2         = 30
LEARNING_RATE_2  = 8e-4
LABEL_SMOOTH_2   = 0.06
AUGMENT_LLM2     = 2      # extra noise-jittered prompt variants per training image
JITTER_SCALE_2   = 0.04   # relative magnitude of the augmentation jitter
ENSEMBLE_SIZE_2  = 5      # snapshot ensemble: top-K val checkpoints, probabilities averaged

DEFECT_COLORS = {
    'crazing':        '#FF6B6B',
    'inclusion':      '#4ECDC4',
    'patches':        '#45B7D1',
    'pitted_surface': '#96CEB4',
    'rolled-in_scale':'#FFEAA7',
    'scratches':      '#DDA0DD'
}
SOURCE_COLORS = {
    'xml_annotation':      '#00FF88',
    'scratch_tiny_llm':    '#4FC3F7',
    'contour_detection':   '#FFA726',
    'full_image_fallback': '#EF5350',
    'scratch_lstm_llm':          '#BA68C8',
    'scratch_lstm_llm_ensemble': '#AB47BC'
}

def collect_image_tasks():
    tasks = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(IMAGES_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f'Missing dir: {cls_dir}')
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                tasks.append({'cls': cls, 'fname': fname, 'path': os.path.join(cls_dir, fname)})
    return tasks

print(f'IMAGES_DIR  : {IMAGES_DIR}')
print(f'ANNOTATIONS : {ANNOTATIONS_DIR}')
print(f'OUTPUT      : {OUTPUT_DIR}')

# ----- cell: cell_03_features -----
# ================================================================
# CELL 3 — RICH FEATURE EXTRACTION
# ================================================================

def compute_glcm(gray_u8):
    if not SKIMAGE_OK:
        return {}
    try:
        g    = (gray_u8 // 32).astype(np.uint8)
        glcm = graycomatrix(g, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=8, symmetric=True, normed=True)
        return {
            'glcm_contrast':    round(float(np.mean(graycoprops(glcm,'contrast'))),    3),
            'glcm_homogeneity': round(float(np.mean(graycoprops(glcm,'homogeneity'))), 3),
            'glcm_energy':      round(float(np.mean(graycoprops(glcm,'energy'))),       3),
            'glcm_correlation': round(float(np.mean(graycoprops(glcm,'correlation'))),  3),
        }
    except Exception:
        return {}


def extract_rich_features(img_path, img_size=200):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (img_size, img_size))
    gray    = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_u8 = gray.astype(np.uint8)
    H, W    = gray.shape
    f = {}

    # Basic stats
    f['mean_intensity'] = round(float(np.mean(gray)), 2)
    f['std_intensity']  = round(float(np.std(gray)),  2)
    f['contrast']       = round(float(np.std(gray)/(np.mean(gray)+1e-5)), 3)
    p = np.histogram(gray_u8, bins=256, range=(0,256))[0].astype(float)
    p /= p.sum() + 1e-9
    p = p[p > 0]
    f['entropy'] = round(float(-np.sum(p*np.log2(p))), 3)

    # Edges
    edges = cv2.Canny(gray_u8, 50, 150)
    # Fix for OpenCV 4.8.0 crash with float32 input
    gx = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    f['edge_density']    = round(float(np.sum(edges>0)/edges.size*100), 2)
    f['sobel_magnitude'] = round(float(np.mean(mag)), 3)
    f['laplacian_var']   = round(float(np.var(cv2.Laplacian(gray_u8,cv2.CV_64F))), 2)
    f['angle_variance']  = round(float(np.var(np.arctan2(gy, gx))), 3)
    h_e = np.sum(np.abs(gy)); v_e = np.sum(np.abs(gx)); tot = h_e+v_e+1e-5
    f['horizontal_edge_ratio'] = round(float(h_e/tot), 3)
    f['vertical_edge_ratio']   = round(float(v_e/tot), 3)

    # Morphological regions
    _, binary   = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_OTSU)
    inv         = cv2.bitwise_not(binary)
    morphed     = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas, circs, aspects, bboxes_raw = [], [], [], []
    for c in contours:
        a = cv2.contourArea(c)
        if a < 30: continue
        areas.append(a)
        p_len = cv2.arcLength(c, True)
        circs.append((4*np.pi*a)/(p_len**2) if p_len>0 else 0)
        x,y,bw,bh = cv2.boundingRect(c)
        bboxes_raw.append((x,y,bw,bh))
        aspects.append(max(bw,bh)/(min(bw,bh)+1e-5))
    n = len(areas)
    f['num_regions']      = n
    f['avg_area']         = round(float(np.mean(areas))  if n>0 else 0, 2)
    f['max_area']         = round(float(np.max(areas))   if n>0 else 0, 2)
    f['total_defect_pct'] = round(float(np.sum(areas)/(H*W)*100) if n>0 else 0, 2)
    f['avg_circularity']  = round(float(np.mean(circs))  if n>0 else 0, 3)
    f['min_circularity']  = round(float(np.min(circs))   if n>0 else 0, 3)
    f['avg_aspect_ratio'] = round(float(np.mean(aspects)) if n>0 else 1, 2)
    f['max_aspect_ratio'] = round(float(np.max(aspects))  if n>0 else 1, 2)
    bins = [0,0,0,0]
    for a in areas:
        if   a < 100:  bins[0]+=1
        elif a < 500:  bins[1]+=1
        elif a < 3000: bins[2]+=1
        else:          bins[3]+=1
    f['regions_tiny']=bins[0]; f['regions_small']=bins[1]
    f['regions_medium']=bins[2]; f['regions_large']=bins[3]

    # Quadrant
    q1,q2=gray[:H//2,:W//2],gray[:H//2,W//2:]
    q3,q4=gray[H//2:,:W//2],gray[H//2:,W//2:]
    qs={'top_left':np.std(q1),'top_right':np.std(q2),'bottom_left':np.std(q3),'bottom_right':np.std(q4)}
    qa=H*W/4
    qe={'top_left':np.sum(edges[:H//2,:W//2]>0)/qa*100,'top_right':np.sum(edges[:H//2,W//2:]>0)/qa*100,
        'bottom_left':np.sum(edges[H//2:,:W//2]>0)/qa*100,'bottom_right':np.sum(edges[H//2:,W//2:]>0)/qa*100}
    f['quadrant_std']={k:round(float(v),2) for k,v in qs.items()}
    f['quadrant_edge_density']={k:round(float(v),2) for k,v in qe.items()}
    f['most_active_quadrant']=max(qs,key=qs.get)
    f['edge_row_peak_pct']=round(float(np.argmax(np.sum(edges,axis=1)))/H*100,1)
    f['edge_col_peak_pct']=round(float(np.argmax(np.sum(edges,axis=0)))/W*100,1)

    # Brightness
    f['dark_pixel_pct']   = round(float(np.sum(gray<80)/gray.size*100),2)
    f['bright_pixel_pct'] = round(float(np.sum(gray>180)/gray.size*100),2)
    f['mid_pixel_pct']    = round(100-f['dark_pixel_pct']-f['bright_pixel_pct'],2)

    # FFT
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    cy,cx   = H//2,W//2
    f['fft_center_energy']=round(float(np.sum(fft_mag[cy-20:cy+20,cx-20:cx+20])/(fft_mag.sum()+1e-9)),4)

    f.update(compute_glcm(gray_u8))
    f['_raw_bboxes'] = bboxes_raw
    return f, img_res

print('Cell 3 OK — extract_rich_features() defined')

# ----- cell: cell_04_prompt -----
# ================================================================
# CELL 4 — FEATURE PROMPT BUILDER FOR LOCAL SCRATCH LLM
# ================================================================

def normalize_label(text):
    if text is None:
        return None
    raw = str(text).strip().lower().replace('-', '_').replace(' ', '_')
    raw = re.sub(r'[^a-z0-9_]', '', raw)
    if raw in LABEL_TO_ID:
        return raw
    if raw == 'rolledinscale' or raw == 'rolled_in_scale':
        return 'rolled-in_scale'
    if raw == 'pittedsurface' or raw == 'pitted_surface':
        return 'pitted_surface'
    return None


def bin_value(value, bins):
    for name, lo, hi in bins:
        if lo <= value < hi:
            return name
    return bins[-1][0]


def features_to_prompt(f):
    """Convert image features into a structured text prompt.
    This prompt is fed to the local Transformer classifier.
    """
    entropy_bin = bin_value(float(f.get('entropy', 0)), [
        ('entropy_very_low', 0, 4.0), ('entropy_low', 4.0, 5.5),
        ('entropy_medium', 5.5, 6.5), ('entropy_high', 6.5, 10.0)
    ])
    area_bin = bin_value(float(f.get('avg_area', 0)), [
        ('avg_area_tiny', 0, 100), ('avg_area_small', 100, 500),
        ('avg_area_medium', 500, 2500), ('avg_area_large', 2500, 1e9)
    ])
    aspect_bin = bin_value(float(f.get('max_aspect_ratio', 1)), [
        ('aspect_low', 0, 1.8), ('aspect_medium', 1.8, 4.0),
        ('aspect_high', 4.0, 8.0), ('aspect_very_high', 8.0, 1e9)
    ])
    coverage_bin = bin_value(float(f.get('total_defect_pct', 0)), [
        ('coverage_low', 0, 5), ('coverage_medium', 5, 15),
        ('coverage_high', 15, 35), ('coverage_very_high', 35, 100)
    ])
    dark_bin = bin_value(float(f.get('dark_pixel_pct', 0)), [
        ('dark_low', 0, 8), ('dark_medium', 8, 20),
        ('dark_high', 20, 40), ('dark_very_high', 40, 100)
    ])
    edge_bin = bin_value(float(f.get('edge_density', 0)), [
        ('edge_low', 0, 4), ('edge_medium', 4, 10),
        ('edge_high', 10, 20), ('edge_very_high', 20, 100)
    ])

    tokens = [
        'steel', 'surface', 'defect', 'classification',
        f"regions_{int(f.get('num_regions', 0))}",
        f"tiny_{int(f.get('regions_tiny', 0))}",
        f"small_{int(f.get('regions_small', 0))}",
        f"medium_{int(f.get('regions_medium', 0))}",
        f"large_{int(f.get('regions_large', 0))}",
        entropy_bin, area_bin, aspect_bin, coverage_bin, dark_bin, edge_bin,
        f"active_{f.get('most_active_quadrant', 'center')}",
        f"h_edge_{round(float(f.get('horizontal_edge_ratio', 0)), 1)}",
        f"v_edge_{round(float(f.get('vertical_edge_ratio', 0)), 1)}",
        f"circularity_{round(float(f.get('avg_circularity', 0)), 1)}",
        f"glcm_contrast_{round(float(f.get('glcm_contrast', 0)), 1)}",
        f"glcm_energy_{round(float(f.get('glcm_energy', 0)), 1)}"
    ]
    return ' '.join(tokens)

print('Cell 4 OK — features_to_prompt() defined')

# ----- cell: cell_05_classifiers -----
# ================================================================
# CELL 5 — FROM-SCRATCH TINY LLM / TRANSFORMER CLASSIFIER
# ================================================================

if not TORCH_OK:
    raise ImportError('PyTorch is required for the scratch tiny LLM. Install: pip install torch torchvision torchaudio')

SPECIAL_TOKENS = ['<pad>', '<unk>', '<cls>']
PAD_ID, UNK_ID, CLS_ID = 0, 1, 2

class SimpleTokenizer:
    def __init__(self, token_to_id=None, max_len=160):
        self.token_to_id = token_to_id or {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.max_len = max_len

    def tokenize(self, text):
        return re.findall(r'[a-zA-Z0-9_\.\-]+', str(text).lower())

    def build(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))
        for tok, freq in sorted(counter.items()):
            if freq >= min_freq and tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)
        return self

    def encode(self, text):
        ids = [CLS_ID] + [self.token_to_id.get(tok, UNK_ID) for tok in self.tokenize(text)]
        ids = ids[:self.max_len]
        attn = [1] * len(ids)
        while len(ids) < self.max_len:
            ids.append(PAD_ID)
            attn.append(0)
        return ids, attn

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'token_to_id': self.token_to_id, 'max_len': self.max_len}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        return cls(obj['token_to_id'], obj.get('max_len', MAX_LEN))


class SteelPromptDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, label = self.rows[idx]
        ids, attn = self.tokenizer.encode(text)
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.long)
        }


class TinySteelLLM(nn.Module):
    """Small Transformer classifier trained from scratch.
    It is LLM-style because it learns token embeddings + self-attention over a feature prompt.
    No pretrained weights or external API are used.
    """
    def __init__(self, vocab_size, num_classes, max_len=160, embed_dim=96, heads=4, layers=2, dropout=0.15):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        key_padding_mask = ~attention_mask.bool()
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        cls_vec = self.norm(x[:, 0, :])
        return self.classifier(cls_vec)


SCRATCH_LLM_MODEL = None
SCRATCH_TOKENIZER = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def stratified_split(rows, train_ratio=0.80):
    by_label = {i: [] for i in range(len(CLASS_NAMES))}
    for row in rows:
        by_label[row[1]].append(row)
    train_rows, val_rows = [], []
    rnd = random.Random(SEED)
    for label, items in by_label.items():
        rnd.shuffle(items)
        cut = max(1, int(len(items) * train_ratio))
        train_rows.extend(items[:cut])
        val_rows.extend(items[cut:])
    rnd.shuffle(train_rows)
    rnd.shuffle(val_rows)
    return train_rows, val_rows


def build_training_rows(tasks, img_size=200):
    rows, skipped = [], 0
    for idx, task in enumerate(tasks, 1):
        f, _ = extract_rich_features(task['path'], img_size)
        if f is None:
            skipped += 1
            continue
        rows.append((features_to_prompt(f), LABEL_TO_ID[task['cls']]))
        if idx % 200 == 0:
            print(f'Feature prompts prepared: {idx}/{len(tasks)}')
    print(f'Training rows: {len(rows)} | skipped: {skipped}')
    return rows


def evaluate_loader(model, loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    per_class = {c: {'correct': 0, 'total': 0} for c in CLASS_NAMES}
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            logits = model(ids, mask)
            loss = F.cross_entropy(logits, y)
            pred = logits.argmax(dim=1)
            loss_sum += float(loss.item()) * y.size(0)
            total += y.size(0)
            correct += int((pred == y).sum().item())
            for yy, pp in zip(y.cpu().tolist(), pred.cpu().tolist()):
                cls = ID_TO_LABEL[yy]
                per_class[cls]['total'] += 1
                if yy == pp:
                    per_class[cls]['correct'] += 1
    return correct / max(total, 1), loss_sum / max(total, 1), per_class


def train_tiny_steel_llm(force_retrain=False):
    global SCRATCH_LLM_MODEL, SCRATCH_TOKENIZER

    tasks = collect_image_tasks()
    if not tasks:
        raise FileNotFoundError('No dataset images found. Check BASE_PATH / IMAGES_DIR.')

    if (not force_retrain) and os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        SCRATCH_TOKENIZER = SimpleTokenizer.load(TOKENIZER_PATH)
        SCRATCH_LLM_MODEL = TinySteelLLM(
            vocab_size=len(SCRATCH_TOKENIZER.token_to_id),
            num_classes=len(CLASS_NAMES),
            max_len=MAX_LEN,
            embed_dim=EMBED_DIM,
            heads=NUM_HEADS,
            layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(DEVICE)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        SCRATCH_LLM_MODEL.load_state_dict(ckpt['model_state'])
        SCRATCH_LLM_MODEL.eval()
        print(f'Loaded trained scratch tiny LLM from: {MODEL_PATH}')
        return SCRATCH_LLM_MODEL, SCRATCH_TOKENIZER

    rows = build_training_rows(tasks, IMG_SIZE)
    train_rows, val_rows = stratified_split(rows, TRAIN_SPLIT)
    print(f'Train: {len(train_rows)} | Validation: {len(val_rows)} | Device: {DEVICE}')

    SCRATCH_TOKENIZER = SimpleTokenizer(max_len=MAX_LEN).build([r[0] for r in train_rows], MIN_TOKEN_FREQ)
    train_ds = SteelPromptDataset(train_rows, SCRATCH_TOKENIZER)
    val_ds = SteelPromptDataset(val_rows, SCRATCH_TOKENIZER)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    SCRATCH_LLM_MODEL = TinySteelLLM(
        vocab_size=len(SCRATCH_TOKENIZER.token_to_id),
        num_classes=len(CLASS_NAMES),
        max_len=MAX_LEN,
        embed_dim=EMBED_DIM,
        heads=NUM_HEADS,
        layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(SCRATCH_LLM_MODEL.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_acc, best_state = -1.0, None

    for epoch in range(1, EPOCHS + 1):
        SCRATCH_LLM_MODEL.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = SCRATCH_LLM_MODEL(ids, mask)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(SCRATCH_LLM_MODEL.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * y.size(0)
            seen += y.size(0)

        val_acc, val_loss, _ = evaluate_loader(SCRATCH_LLM_MODEL, val_loader)
        print(f'Epoch {epoch:02d}/{EPOCHS} | train_loss={total_loss/max(seen,1):.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%')
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in SCRATCH_LLM_MODEL.state_dict().items()}

    if best_state is not None:
        SCRATCH_LLM_MODEL.load_state_dict(best_state)
    SCRATCH_LLM_MODEL.eval()

    SCRATCH_TOKENIZER.save(TOKENIZER_PATH)
    torch.save({
        'model_state': SCRATCH_LLM_MODEL.state_dict(),
        'class_names': CLASS_NAMES,
        'config': {
            'max_len': MAX_LEN, 'embed_dim': EMBED_DIM,
            'heads': NUM_HEADS, 'layers': NUM_LAYERS,
            'dropout': DROPOUT
        },
        'best_val_acc': best_acc
    }, MODEL_PATH)
    print(f'Saved scratch tiny LLM: {MODEL_PATH}')
    print(f'Best validation accuracy: {best_acc*100:.2f}%')
    return SCRATCH_LLM_MODEL, SCRATCH_TOKENIZER


def classify_defect_ensemble(features, image_key=None):
    if SCRATCH_LLM_MODEL is None or SCRATCH_TOKENIZER is None:
        raise RuntimeError('Scratch tiny LLM is not trained/loaded. Run train_tiny_steel_llm() first.')
    prompt = features_to_prompt(features)
    ids, attn = SCRATCH_TOKENIZER.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    m = torch.tensor([attn], dtype=torch.bool, device=DEVICE)
    SCRATCH_LLM_MODEL.eval()
    with torch.no_grad():
        logits = SCRATCH_LLM_MODEL(x, m)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_id = int(np.argmax(probs))
    return ID_TO_LABEL[pred_id], float(probs[pred_id]), 'scratch_tiny_llm'

print('Cell 5 OK — scratch tiny LLM classifier defined')

# ----- cell: cell_05b_llm2_def -----
# ================================================================
# CELL 5B — SECOND FROM-SCRATCH LLM: BiLSTM + RICH PROMPT + ENSEMBLE
# A different architecture family from the Transformer in Cell 5,
# trained independently from random init, producing its own results.
#
# Extra accuracy levers stacked on top of the multi-view-pooling BiLSTM:
#   1) features_to_prompt_llm2()  -- a much richer structured-text prompt
#      (finer bins + GLCM + per-quadrant stats + spectral/edge-peak info)
#      so confusable classes (scratches vs patches vs crazing) leave a
#      more distinctive token signature for the model to learn from.
#   2) build_training_rows_llm2() -- light feature-space jitter creates
#      AUGMENT_LLM2 extra training variants per image (more data for a
#      from-scratch model without touching the original dataset).
#   3) snapshot ensembling -- the top ENSEMBLE_SIZE_2 checkpoints (by val
#      accuracy) are kept and their softmax probabilities are averaged at
#      inference time, which smooths out single-run variance.
# ================================================================

class TinyScratchLLM_BiLSTM(nn.Module):
    """Second from-scratch tiny LLM.
    Token embeddings -> bidirectional LSTM -> three pooled views
    (learned attention pooling, masked max-pooling, masked mean-pooling)
    concatenated -> classification head. No pretrained weights or
    external API; trained from random initialization, with a recurrent
    encoder instead of self-attention.
    """
    def __init__(self, vocab_size, num_classes, hidden_dim=192, embed_dim=96, layers=2, dropout=0.30):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.embed_dropout = nn.Dropout(dropout * 0.5)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        enc_dim = hidden_dim * 2
        self.attn_score = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2),
            nn.Tanh(),
            nn.Linear(enc_dim // 2, 1)
        )
        self.norm = nn.LayerNorm(enc_dim * 3)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(enc_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        mask = attention_mask.bool()
        x = self.embed_dropout(self.token_embedding(input_ids))
        out, _ = self.lstm(x)                                    # (B, T, 2H)

        scores = self.attn_score(out).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_w = torch.softmax(scores, dim=1).unsqueeze(-1)
        attn_pool = torch.sum(out * attn_w, dim=1)

        masked_out = out.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        max_pool, _ = masked_out.max(dim=1)

        mask_f = mask.unsqueeze(-1).float()
        mean_pool = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        pooled = self.norm(torch.cat([attn_pool, max_pool, mean_pool], dim=-1))
        return self.classifier(pooled)


def _fbin(value, edges, names):
    for name, hi in zip(names, edges):
        if value < hi:
            return name
    return names[-1]


def features_to_prompt_llm2(f):
    """Richer structured-text prompt for the second LLM.
    Reuses the same rich feature dict as LLM 1 but keeps far more of it
    (GLCM texture stats, per-quadrant std/edge density, spectral energy,
    edge-peak locations) and bins continuous values more finely, giving
    the BiLSTM more signal to separate visually-similar defect classes.
    """
    b6 = ['_b0', '_b1', '_b2', '_b3', '_b4', '_b5', '_b6']
    entropy_bin  = 'entropy'  + _fbin(float(f.get('entropy', 0)),         [3.5, 4.5, 5.2, 5.8, 6.4, 7.0, 99],   b6)
    area_bin     = 'area'     + _fbin(float(f.get('avg_area', 0)),        [50, 150, 350, 800, 2000, 5000, 1e9], b6)
    aspect_bin   = 'aspect'   + _fbin(float(f.get('max_aspect_ratio', 1)),[1.4, 2.0, 2.8, 4.0, 6.0, 9.0, 1e9],  b6)
    coverage_bin = 'coverage' + _fbin(float(f.get('total_defect_pct', 0)),[2, 5, 9, 15, 24, 35, 100],           b6)
    dark_bin     = 'dark'     + _fbin(float(f.get('dark_pixel_pct', 0)),  [4, 9, 16, 25, 36, 50, 100],          b6)
    bright_bin   = 'bright'   + _fbin(float(f.get('bright_pixel_pct', 0)),[2, 6, 12, 20, 30, 45, 100],          b6)
    edge_bin     = 'edge'     + _fbin(float(f.get('edge_density', 0)),    [2, 4, 7, 11, 16, 24, 100],           b6)
    sobel_bin    = 'sobel'    + _fbin(float(f.get('sobel_magnitude', 0)), [10, 20, 35, 55, 80, 120, 1e9],       b6)
    lap_bin      = 'lap'      + _fbin(float(f.get('laplacian_var', 0)),   [50, 150, 350, 700, 1300, 2500, 1e9], b6)
    fft_bin      = 'fft'      + _fbin(float(f.get('fft_center_energy', 0)),[0.05, 0.1, 0.18, 0.28, 0.4, 0.55, 1.0], b6)

    qstd  = f.get('quadrant_std', {})
    qedge = f.get('quadrant_edge_density', {})
    quad_tokens = []
    for q in ('top_left', 'top_right', 'bottom_left', 'bottom_right'):
        quad_tokens.append(f"qstd_{q}_{int(round(float(qstd.get(q, 0))))}")
        quad_tokens.append(f"qedge_{q}_{int(round(float(qedge.get(q, 0))))}")

    tokens = [
        'steel', 'surface', 'defect', 'classification',
        f"regions_{int(f.get('num_regions', 0))}",
        f"tiny_{int(f.get('regions_tiny', 0))}",
        f"small_{int(f.get('regions_small', 0))}",
        f"medium_{int(f.get('regions_medium', 0))}",
        f"large_{int(f.get('regions_large', 0))}",
        entropy_bin, area_bin, aspect_bin, coverage_bin,
        dark_bin, bright_bin, edge_bin, sobel_bin, lap_bin, fft_bin,
        f"active_{f.get('most_active_quadrant', 'center')}",
        f"h_edge_{round(float(f.get('horizontal_edge_ratio', 0)), 2)}",
        f"v_edge_{round(float(f.get('vertical_edge_ratio', 0)), 2)}",
        f"row_peak_{int(round(float(f.get('edge_row_peak_pct', 0)) / 10) * 10)}",
        f"col_peak_{int(round(float(f.get('edge_col_peak_pct', 0)) / 10) * 10)}",
        f"circularity_{round(float(f.get('avg_circularity', 0)), 2)}",
        f"min_circularity_{round(float(f.get('min_circularity', 0)), 2)}",
        f"aspect_avg_{round(float(f.get('avg_aspect_ratio', 1)), 1)}",
        f"glcm_contrast_{round(float(f.get('glcm_contrast', 0)), 2)}",
        f"glcm_homogeneity_{round(float(f.get('glcm_homogeneity', 0)), 2)}",
        f"glcm_energy_{round(float(f.get('glcm_energy', 0)), 2)}",
        f"glcm_correlation_{round(float(f.get('glcm_correlation', 0)), 2)}",
    ] + quad_tokens
    return ' '.join(tokens)


def _jitter_feature_dict(f, rng, scale=0.04):
    """Light multiplicative-Gaussian jitter on numeric feature values.
    Used only to synthesize extra *training* prompts (data augmentation
    in feature space) -- never used at inference time.
    """
    out = {}
    for k, v in f.items():
        if k == '_raw_bboxes':
            continue
        if isinstance(v, dict):
            out[k] = {kk: float(vv) * (1.0 + rng.gauss(0, scale)) for kk, vv in v.items()}
        elif isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = float(v) * (1.0 + rng.gauss(0, scale))
        else:
            out[k] = v
    return out


def build_training_rows_llm2(tasks, img_size=200, augment=2, jitter_scale=0.04):
    rows, skipped = [], 0
    rng = random.Random(SEED + 7)
    for idx, task in enumerate(tasks, 1):
        f, _ = extract_rich_features(task['path'], img_size)
        if f is None:
            skipped += 1
            continue
        label = LABEL_TO_ID[task['cls']]
        rows.append((features_to_prompt_llm2(f), label))
        for _ in range(augment):
            rows.append((features_to_prompt_llm2(_jitter_feature_dict(f, rng, jitter_scale)), label))
        if idx % 200 == 0:
            print(f'Feature prompts prepared (LLM2, +{augment} jittered each): {idx}/{len(tasks)}')
    print(f'Training rows (LLM2): {len(rows)}  (from {len(tasks) - skipped} images x {augment + 1})  | skipped: {skipped}')
    return rows


SCRATCH_LLM_ENSEMBLE_2 = []     # snapshot ensemble: list of loaded TinyScratchLLM_BiLSTM
SCRATCH_TOKENIZER_2 = None


def _build_llm2_model(tokenizer):
    return TinyScratchLLM_BiLSTM(
        vocab_size=len(tokenizer.token_to_id),
        num_classes=len(CLASS_NAMES),
        hidden_dim=HIDDEN_DIM_2,
        embed_dim=EMBED_DIM,
        layers=NUM_LAYERS_2,
        dropout=DROPOUT_2
    ).to(DEVICE)


def _load_llm2_ensemble(state_dicts, tokenizer):
    models = []
    for state in state_dicts:
        m = _build_llm2_model(tokenizer)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models


def train_tiny_steel_llm2(force_retrain=False):
    global SCRATCH_LLM_ENSEMBLE_2, SCRATCH_TOKENIZER_2

    tasks = collect_image_tasks()
    if not tasks:
        raise FileNotFoundError('No dataset images found. Check BASE_PATH / IMAGES_DIR.')

    if (not force_retrain) and os.path.exists(MODEL_PATH_2) and os.path.exists(TOKENIZER_PATH_2):
        SCRATCH_TOKENIZER_2 = SimpleTokenizer.load(TOKENIZER_PATH_2)
        ckpt = torch.load(MODEL_PATH_2, map_location=DEVICE)
        SCRATCH_LLM_ENSEMBLE_2 = _load_llm2_ensemble(ckpt['model_states'], SCRATCH_TOKENIZER_2)
        print(f'Loaded second-LLM snapshot ensemble ({len(SCRATCH_LLM_ENSEMBLE_2)} models) from: {MODEL_PATH_2}')
        return SCRATCH_LLM_ENSEMBLE_2, SCRATCH_TOKENIZER_2

    rows = build_training_rows_llm2(tasks, IMG_SIZE, augment=AUGMENT_LLM2, jitter_scale=JITTER_SCALE_2)
    train_rows, val_rows = stratified_split(rows, TRAIN_SPLIT)
    print(f'Train: {len(train_rows)} | Validation: {len(val_rows)} | Device: {DEVICE}')

    SCRATCH_TOKENIZER_2 = SimpleTokenizer(max_len=MAX_LEN).build([r[0] for r in train_rows], MIN_TOKEN_FREQ)
    train_ds = SteelPromptDataset(train_rows, SCRATCH_TOKENIZER_2)
    val_ds = SteelPromptDataset(val_rows, SCRATCH_TOKENIZER_2)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = _build_llm2_model(SCRATCH_TOKENIZER_2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE_2, epochs=EPOCHS_2,
        steps_per_epoch=max(len(train_loader), 1), pct_start=0.15
    )

    snapshots = []   # list of (val_acc, state_dict) -- kept sorted, top ENSEMBLE_SIZE_2

    for epoch in range(1, EPOCHS_2 + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTH_2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item()) * y.size(0)
            seen += y.size(0)

        val_acc, val_loss, _ = evaluate_loader(model, val_loader)
        cur_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch:02d}/{EPOCHS_2} | train_loss={total_loss/max(seen,1):.4f} | '
              f'val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}% | lr={cur_lr:.2e}')

        snapshots.append((val_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
        snapshots = sorted(snapshots, key=lambda s: -s[0])[:ENSEMBLE_SIZE_2]

    best_states = [s for _, s in snapshots]
    best_accs = [a for a, _ in snapshots]
    SCRATCH_LLM_ENSEMBLE_2 = _load_llm2_ensemble(best_states, SCRATCH_TOKENIZER_2)

    SCRATCH_TOKENIZER_2.save(TOKENIZER_PATH_2)
    torch.save({
        'model_states': best_states,
        'snapshot_val_accs': best_accs,
        'class_names': CLASS_NAMES,
        'config': {
            'hidden_dim': HIDDEN_DIM_2, 'embed_dim': EMBED_DIM,
            'layers': NUM_LAYERS_2, 'dropout': DROPOUT_2
        },
        'best_val_acc': best_accs[0] if best_accs else -1.0
    }, MODEL_PATH_2)
    print(f'Saved second-LLM snapshot ensemble ({len(best_states)} models): {MODEL_PATH_2}')
    print('Snapshot validation accuracies: ' + ', '.join(f'{a*100:.2f}%' for a in best_accs))
    return SCRATCH_LLM_ENSEMBLE_2, SCRATCH_TOKENIZER_2


def classify_defect_llm2(features, image_key=None):
    if not SCRATCH_LLM_ENSEMBLE_2 or SCRATCH_TOKENIZER_2 is None:
        raise RuntimeError('Second scratch LLM is not trained/loaded. Run train_tiny_steel_llm2() first.')
    prompt = features_to_prompt_llm2(features)
    ids, attn = SCRATCH_TOKENIZER_2.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    m = torch.tensor([attn], dtype=torch.bool, device=DEVICE)
    probs_sum = None
    with torch.no_grad():
        for net in SCRATCH_LLM_ENSEMBLE_2:
            net.eval()
            probs = torch.softmax(net(x, m), dim=1)[0]
            probs_sum = probs.clone() if probs_sum is None else probs_sum + probs
    probs = (probs_sum / len(SCRATCH_LLM_ENSEMBLE_2)).detach().cpu().numpy()
    pred_id = int(np.argmax(probs))
    return ID_TO_LABEL[pred_id], float(probs[pred_id]), 'scratch_lstm_llm_ensemble'

print('Cell 5B OK — second from-scratch LLM (rich-prompt BiLSTM + augmentation + snapshot ensemble) defined')

# ----- cell: 4fdcc097 -----
# ================================================================
# CELL 6 — LOCAL AREA DETECTION: XML FIRST, THEN CONTOUR FALLBACK
# ================================================================

def load_xml_annotations(img_filename, target_size=(200, 200)):
    base = os.path.splitext(img_filename)[0]
    xml = os.path.join(ANNOTATIONS_DIR, base + '.xml')
    if not os.path.exists(xml):
        return None
    try:
        root = ET.parse(xml).getroot()
        size = root.find('size')
        orig_w = int(size.find('width').text)
        orig_h = int(size.find('height').text)
        sx, sy = target_size[1] / orig_w, target_size[0] / orig_h
        regions = []
        for obj in root.findall('object'):
            bb = obj.find('bndbox')
            xmin = max(0, int(float(bb.find('xmin').text) * sx))
            ymin = max(0, int(float(bb.find('ymin').text) * sy))
            xmax = min(target_size[1], int(float(bb.find('xmax').text) * sx))
            ymax = min(target_size[0], int(float(bb.find('ymax').text) * sy))
            w, h = xmax - xmin, ymax - ymin
            label = normalize_label(obj.find('name').text) or obj.find('name').text
            if w > 2 and h > 2:
                regions.append({'bbox': [xmin, ymin, w, h], 'label': label, 'area': w * h, 'source': 'xml_annotation'})
        return regions or None
    except Exception:
        return None

def contour_regions(img_res, pred_class, img_size):
    gray = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(binary, edges)
    ker3 = np.ones((3, 3), np.uint8)
    ker5 = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker5)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, ker3)
    cts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regs = []
    for c in cts:
        area = cv2.contourArea(c)
        if area > 60:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw > 2 and bh > 2:
                regs.append({'bbox': [x, y, bw, bh], 'label': pred_class, 'area': int(area), 'source': 'contour_detection'})
    regs = sorted(regs, key=lambda r: r['area'], reverse=True)[:8]
    return regs or [{'bbox': [0, 0, img_size, img_size], 'label': pred_class, 'area': img_size ** 2, 'source': 'full_image_fallback'}]


def detect_areas(f, img_res, pred_class, img_filename, img_size=200, use_llm=False):
    xml = load_xml_annotations(img_filename, (img_size, img_size))
    if xml:
        # During visualization, show predicted class on XML boxes while keeping XML source.
        for r in xml:
            r['label'] = pred_class
        return xml, 'xml_annotation'
    regs = contour_regions(img_res, pred_class, img_size)
    return regs, regs[0]['source']

print('Cell 6 OK — local XML/contour area detector loaded')


# ----- cell: cell_07b_train_llm2 -----
# ================================================================
# CELL 7B — TRAIN OR LOAD THE SECOND LOCAL SCRATCH LLM (BiLSTM ensemble)
# ================================================================
# NOTE: this version trains with a richer per-image prompt, light
# feature-space augmentation, and keeps a snapshot ensemble (top
# ENSEMBLE_SIZE_2 checkpoints by validation accuracy, averaged at
# inference). The checkpoint format changed, so the first run needs
# force_retrain=False; afterwards you can switch it back to False to
# just reload the saved ensemble.

model2, tokenizer2 = train_tiny_steel_llm2(force_retrain=False)
print('Ready for detection with:', MODEL_NAME_2, f'(ensemble of {len(model2)} models)')

# ----- cell: cell_08b_pipeline_llm2 -----
# ================================================================
# CELL 8B — SINGLE-IMAGE PIPELINE FOR THE SECOND LLM
# ================================================================

def process_image_llm2(img_path, img_filename, img_size=200):
    f, img_res = extract_rich_features(img_path, img_size)
    if f is None:
        return {'error': f'load failed: {img_path}'}
    pred, conf, method = classify_defect_llm2(f, image_key=img_filename)
    regions, area_src = detect_areas(f, img_res, pred, img_filename,
                                     img_size=img_size, use_llm=False)
    return {
        'image':       img_res,
        'filename':    img_filename,
        'pred_label':  pred,
        'confidence':  conf,
        'cls_method':  method,
        'regions':     regions,
        'area_method': area_src,
        'features':    f,
        'prompt':      features_to_prompt_llm2(f)
    }

print('Cell 8B OK — process_image_llm2() defined')

# ----- HONEST SCORING PATCH -----
# detect_areas normally returns XML ground-truth boxes for images that have annotations
# (which is nearly every NEU-DET image), making IoU trivially 1.0 and AP50/AP75
# meaningless (flat across all thresholds). Override to always use the real contour
# detector so we measure genuine localization quality.
def detect_areas(f, img_res, pred_class, img_filename, img_size=200, use_llm=False):
    regs = contour_regions(img_res, pred_class, img_size)
    return regs, regs[0]['source']


# ----- LLM2 inference: classify + contour regions for all images -----
import time as _time
all_tasks = collect_image_tasks()
print(f"Scoring {len(all_tasks)} images with LLM2 + honest contour detector...")
predictions_llm2 = []
_t0 = _time.time()
for idx, task in enumerate(all_tasks, 1):
    try:
        result = process_image_llm2(task['path'], task['fname'], img_size=IMG_SIZE)
        if 'error' not in result:
            predictions_llm2.append(result)
    except Exception as ex:
        pass
    if idx % 300 == 0 or idx == len(all_tasks):
        print(f"  [{idx}/{len(all_tasks)}]  {_time.time()-_t0:.0f}s")
print(f"Done. {len(predictions_llm2)} predictions built.")


# ----- genuine mAP / AP50 / AP75 for LLM2 (contour detector, no GT leakage) -----
import numpy as np

def _build_gt(tasks, img_size=IMG_SIZE):
    gt = {}
    for t in tasks:
        regs = load_xml_annotations(t['fname'], (img_size, img_size))
        if regs:
            gt[t['fname']] = [{'bbox': list(r['bbox']), 'label': r['label']} for r in regs]
        else:
            gt[t['fname']] = [{'bbox': [0, 0, img_size, img_size], 'label': t['cls']}]
    return gt

def _iou(a, b):
    ax1,ay1,aw,ah = a; ax2,ay2 = ax1+aw, ay1+ah
    bx1,by1,bw,bh = b; bx2,by2 = bx1+bw, by1+bh
    iw = max(0.0, min(ax2,bx2)-max(ax1,bx1))
    ih = max(0.0, min(ay2,by2)-max(ay1,by1))
    inter = iw*ih
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0

def _ap(dets, gt_by_img, thr):
    n_gt = sum(len(v) for v in gt_by_img.values())
    if n_gt == 0: return None
    dets = sorted(dets, key=lambda d: -d[0])
    matched = {fn: [False]*len(b) for fn,b in gt_by_img.items()}
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    for i,(score,fn,box) in enumerate(dets):
        best_iou,best_j = 0.0,-1
        for j,g in enumerate(gt_by_img.get(fn,[])):
            v = _iou(box,g)
            if v > best_iou: best_iou,best_j = v,j
        if best_iou >= thr and best_j >= 0 and not matched[fn][best_j]:
            tp[i]=1; matched[fn][best_j]=True
        else:
            fp[i]=1
    tp_c,fp_c = np.cumsum(tp),np.cumsum(fp)
    rec  = tp_c/(n_gt+1e-9)
    prec = tp_c/np.maximum(tp_c+fp_c,1e-9)
    mrec = np.concatenate(([0.0],rec,[1.0]))
    mpre = np.concatenate(([0.0],prec,[0.0]))
    for k in range(len(mpre)-1,0,-1): mpre[k-1]=max(mpre[k-1],mpre[k])
    idx = np.where(mrec[1:]!=mrec[:-1])[0]
    return float(np.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1]))

def _compute_map(predictions, gt, thrs=None):
    if thrs is None: thrs = [round(0.5+0.05*i,2) for i in range(10)]
    det_by_cls = {c:[] for c in CLASS_NAMES}
    for p in predictions:
        c,s,fn = p['pred_label'],float(p.get('confidence',0.0)),p['filename']
        for r in p.get('regions',[]):
            det_by_cls.setdefault(c,[]).append((s,fn,list(r['bbox'])))
    gt_by_cls = {c:{} for c in CLASS_NAMES}
    for fn,boxes in gt.items():
        for b in boxes:
            gt_by_cls.setdefault(b['label'],{}).setdefault(fn,[]).append(b['bbox'])
    ap_per_iou = {}
    for thr in thrs:
        aps = [_ap(det_by_cls.get(c,[]),gt_by_cls.get(c,{}),thr)
               for c in CLASS_NAMES]
        aps = [a for a in aps if a is not None]
        ap_per_iou[f'{thr:.2f}'] = float(np.mean(aps)) if aps else 0.0
    return float(np.mean(list(ap_per_iou.values()))), ap_per_iou

gt = _build_gt(all_tasks)
map_val, ap_per = _compute_map(predictions_llm2, gt)
ap50 = ap_per.get('0.50', 0.0)
ap75 = ap_per.get('0.75', 0.0)

print()
print("="*60)
print("  LLM2 (TinyScratchLLM_BiLSTM) — GENUINE DETECTION SCORES")
print("  Contour detector used for boxes (no GT leakage)")
print("="*60)
print(f"  mAP@[0.50:0.95] = {map_val*100:.2f}%")
print(f"  AP50 (IoU=0.50) = {ap50*100:.2f}%")
print(f"  AP75 (IoU=0.75) = {ap75*100:.2f}%")
print()
print("  Per-IoU breakdown:")
for iou,ap in sorted(ap_per.items(), key=lambda x: float(x[0])):
    print(f"    IoU {iou}: {ap*100:.2f}%")
print("="*60)
print("DONE_LLM2_HONEST")
