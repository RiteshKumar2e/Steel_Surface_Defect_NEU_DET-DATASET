"""Tabular baselines on the RAW un-discretized descriptor vector.

Reviewer 1 Q4: XGBoost, Random Forest and an MLP under identical splits,
reporting Macro-F1.

The point of these baselines is to isolate what the text-prompt representation
contributes. They consume `Bundle.*.X` -- the same 92 floats the prompt is built
from, before discretization -- so the only difference from SteelSense-BiLSTM is:

    tabular baseline : raw floats            -> tree / MLP
    SteelSense       : floats -> bins -> tokens -> BiLSTM

If a gradient-boosted tree on the raw vector matches or beats the BiLSTM, the
serialize-to-text step is not what produces the accuracy, and the paper's core
claim has to be restated. That is a result either way and it belongs in the
manuscript.

Hyper-parameters are tuned on the VALIDATION split only, by small grid search;
the test split is scored once with the selected setting, exactly as for the
BiLSTM.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from metrics import evaluate

try:
    from xgboost import XGBClassifier

    HAVE_XGB = True
except Exception:  # pragma: no cover
    HAVE_XGB = False


def _grid(name: str) -> List[dict]:
    return {
        "XGBoost": [
            {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.9},
            {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8},
            {"n_estimators": 800, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8},
        ],
        "RandomForest": [
            {"n_estimators": 500, "max_depth": None, "max_features": "sqrt"},
            {"n_estimators": 800, "max_depth": 24, "max_features": "sqrt"},
            {"n_estimators": 800, "max_depth": None, "max_features": 0.3},
        ],
        "MLP": [
            {"hidden_layer_sizes": (256, 128)},
            {"hidden_layer_sizes": (512, 256)},
            {"hidden_layer_sizes": (256, 256, 128)},
        ],
        "LogisticRegression": [{"C": 0.5}, {"C": 2.0}, {"C": 10.0}],
        "SVM-RBF": [{"C": 2.0, "gamma": "scale"}, {"C": 10.0, "gamma": "scale"}],
    }[name]


def _build(name: str, params: dict, seed: int, n_classes: int):
    if name == "XGBoost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
            **params,
        )
    if name == "RandomForest":
        return RandomForestClassifier(
            random_state=seed, n_jobs=-1, class_weight="balanced_subsample", **params
        )
    if name == "MLP":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        activation="relu",
                        alpha=1e-4,
                        batch_size=32,
                        learning_rate_init=1e-3,
                        max_iter=400,
                        early_stopping=True,
                        n_iter_no_change=25,
                        random_state=seed,
                        **params,
                    ),
                ),
            ]
        )
    if name == "LogisticRegression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                           random_state=seed, **params)),
            ]
        )
    if name == "SVM-RBF":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", SVC(probability=True, class_weight="balanced",
                            random_state=seed, **params)),
            ]
        )
    raise ValueError(name)


def available_models() -> List[str]:
    m = ["RandomForest", "MLP", "LogisticRegression", "SVM-RBF"]
    if HAVE_XGB:
        m.insert(0, "XGBoost")
    return m


def run_one(
    name: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    classes: Sequence[str],
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Dict, np.ndarray]:
    best, best_score, best_params = None, -1.0, None
    for params in _grid(name):
        clf = _build(name, params, seed, len(classes))
        clf.fit(Xtr, ytr)
        s = f1_score(yva, clf.predict(Xva), average="macro", zero_division=0)
        if s > best_score:
            best, best_score, best_params = clf, s, params
    t0 = time.perf_counter()
    prob = best.predict_proba(Xte)
    infer_s = time.perf_counter() - t0
    pred = prob.argmax(1)
    m = evaluate(yte, pred, classes, prob)
    m.update(
        {
            "model": name,
            "seed": seed,
            "selected_params": best_params,
            "val_macro_f1": float(best_score),
            "test_infer_seconds_total": round(infer_s, 4),
            "representation": "raw un-discretized descriptor vector",
            "n_features": int(Xtr.shape[1]),
        }
    )
    try:
        est = best.named_steps["clf"] if hasattr(best, "named_steps") else best
        m["n_params"] = _param_count(name, est, Xtr.shape[1], len(classes))
    except Exception:
        pass
    if verbose:
        print(f"  {name:20s} val_macroF1 {best_score:.4f} -> "
              f"test acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}")
    return m, prob


def _param_count(name: str, est, n_feat: int, n_cls: int) -> int:
    if name == "MLP":
        return int(sum(c.size for c in est.coefs_) + sum(b.size for b in est.intercepts_))
    if name == "LogisticRegression":
        return int(est.coef_.size + est.intercept_.size)
    if name == "RandomForest":
        return int(sum(t.tree_.node_count for t in est.estimators_))
    if name == "XGBoost":
        return int(est.get_booster().trees_to_dataframe().shape[0])
    if name == "SVM-RBF":
        return int(est.support_vectors_.size + est.dual_coef_.size)
    return -1


def run_all(
    Xtr, ytr, Xva, yva, Xte, yte, classes, seed: int = 42, models=None, verbose=True
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    models = models or available_models()
    out, probs = {}, {}
    for name in models:
        try:
            m, p = run_one(name, Xtr, ytr, Xva, yva, Xte, yte, classes, seed, verbose)
            out[name], probs[name] = m, p
        except Exception as e:  # a failed baseline is recorded, never scored
            out[name] = {"model": name, "seed": seed, "failed": repr(e)}
            if verbose:
                print(f"  {name:20s} FAILED: {e}")
    return out, probs
