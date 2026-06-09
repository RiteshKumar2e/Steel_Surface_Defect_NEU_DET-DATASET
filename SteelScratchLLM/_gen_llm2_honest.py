import json

nb = json.load(open('LLM_STEEL_SCRATCH_LOCAL.ipynb', encoding='utf-8'))
cells_by_id = {c.get('id'): ''.join(c.get('source', [])) for c in nb['cells']}

order = [
    'cell_01_setup',
    'cell_02_config',
    'cell_03_features',
    'cell_04_prompt',
    'cell_05_classifiers',
    'cell_05b_llm2_def',
    '4fdcc097',
    'cell_07b_train_llm2',
    'cell_08b_pipeline_llm2',
]

parts = []
for cid in order:
    src = cells_by_id[cid]
    if cid == 'cell_07b_train_llm2':
        src = src.replace('force_retrain=True', 'force_retrain=False')
    parts.append("\n# ----- cell: " + cid + " -----\n" + src)

# Override detect_areas to ALWAYS use contour detector (never GT XML boxes)
# so AP50/AP75 measures real localization, not GT-leakage.
override = r"""
# ----- HONEST SCORING PATCH -----
# detect_areas normally returns XML ground-truth boxes for images that have annotations
# (which is nearly every NEU-DET image), making IoU trivially 1.0 and AP50/AP75
# meaningless (flat across all thresholds). Override to always use the real contour
# detector so we measure genuine localization quality.
def detect_areas(f, img_res, pred_class, img_filename, img_size=200, use_llm=False):
    regs = contour_regions(img_res, pred_class, img_size)
    return regs, regs[0]['source']
"""

parts.append(override)

# Compact LLM2 inference loop (classification + contour-based regions, no image saving)
inference_loop = r"""
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
"""

parts.append(inference_loop)

# Genuine mAP block
map_block = r"""
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
"""

parts.append(map_block)

script = "\n".join(parts)

import ast
try:
    ast.parse(script)
    print("syntax OK")
except SyntaxError as e:
    print("SYNTAX ERROR:", e)

with open('_llm2_honest_eval.py', 'w', encoding='utf-8') as f:
    f.write(script)
print(f"wrote _llm2_honest_eval.py  ({len(script)} chars)")
