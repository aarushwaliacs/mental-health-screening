"""TreeScreen backend: interpretable CART screening for depression, anxiety and stress (single-file edition).

Run API:   uvicorn server:app --port 8000          (set ADMIN_API_KEY first; docs at /docs)
CLI:       python server.py train --synthetic --activate
           python server.py train --data data.csv --name NAME --provenance "source, date, label definition" --activate
           python server.py validate data.csv
           python server.py evaluate [--run-id ID] [--format table|csv|json]
Screening and decision support only -- not a diagnosis. Answers are never stored or logged.
Sections below: questionnaire, config, security, features, synthetic data, dataset validation, explanations,
training/evaluation, registry/datasets/jobs, screening logic, schemas, API, app factory, CLI.
"""
from __future__ import annotations
from typing import Any
from functools import lru_cache
from pathlib import Path
from fastapi import Request
from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets
import threading
import time
from collections import defaultdict, deque
from fastapi import Depends, HTTPException, Request, status
from typing import Iterable
import numpy as np
import pandas as pd
import io
import re
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import math
import platform
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, make_scorer, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import json
import os
from dataclasses import dataclass
import joblib
import hashlib
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ======================================================================
# core/questionnaire
# ======================================================================
LEVELS: list[str] = ["Low", "Moderate", "High"]

FREQUENCY_SCALE_ID = "frequency_0_3"
SCALES: dict[str, dict[str, Any]] = {
    FREQUENCY_SCALE_ID: {
        "id": FREQUENCY_SCALE_ID,
        "options": [
            {"value": 0, "label": "Not at all"},
            {"value": 1, "label": "Several days"},
            {"value": 2, "label": "More than half the days"},
            {"value": 3, "label": "Nearly every day"},
        ],
    }
}
ITEM_MIN, ITEM_MAX = 0, 3
SELF_HARM_ITEM_ID = "phq9_9"

_PHQ9 = [
    ("phq9_1", "Little interest or pleasure in doing things", "PHQ-9 #1 interest"),
    ("phq9_2", "Feeling down, depressed, or hopeless", "PHQ-9 #2 low mood"),
    ("phq9_3", "Trouble falling or staying asleep, or sleeping too much", "PHQ-9 #3 sleep"),
    ("phq9_4", "Feeling tired or having little energy", "PHQ-9 #4 energy"),
    ("phq9_5", "Poor appetite or overeating", "PHQ-9 #5 appetite"),
    (
        "phq9_6",
        "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
        "PHQ-9 #6 self-worth",
    ),
    (
        "phq9_7",
        "Trouble concentrating on things, such as reading the newspaper or watching television",
        "PHQ-9 #7 concentration",
    ),
    (
        "phq9_8",
        "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or "
        "restless that you have been moving around a lot more than usual",
        "PHQ-9 #8 psychomotor",
    ),
    (
        "phq9_9",
        "Thoughts that you would be better off dead, or of hurting yourself in some way",
        "PHQ-9 #9 self-harm thoughts",
    ),
]

_GAD7 = [
    ("gad7_1", "Feeling nervous, anxious, or on edge", "GAD-7 #1 nervous"),
    ("gad7_2", "Not being able to stop or control worrying", "GAD-7 #2 uncontrolled worry"),
    ("gad7_3", "Worrying too much about different things", "GAD-7 #3 excess worry"),
    ("gad7_4", "Trouble relaxing", "GAD-7 #4 relaxing"),
    ("gad7_5", "Being so restless that it is hard to sit still", "GAD-7 #5 restless"),
    ("gad7_6", "Becoming easily annoyed or irritable", "GAD-7 #6 irritable"),
    ("gad7_7", "Feeling afraid, as if something awful might happen", "GAD-7 #7 afraid"),
]

# PROPOSED supplementary items (configurable; not validated; not supervisor-approved)
_SUPPLEMENTARY = [
    ("sup_sleep_quality", "sleep", "I have been unhappy with the quality of my sleep", "Sleep quality"),
    (
        "sup_sleep_irregular",
        "sleep",
        "My sleep times have been irregular (going to bed or waking up at very different times)",
        "Irregular sleep schedule",
    ),
    (
        "sup_academic_overwhelm",
        "academic",
        "I have felt overwhelmed by my academic workload",
        "Academic overload",
    ),
    (
        "sup_academic_worry",
        "academic",
        "I have worried about my exam results or academic performance",
        "Academic worry",
    ),
    (
        "sup_social_avoidance",
        "behaviour",
        "I have avoided friends, family or social activities I would normally take part in",
        "Social avoidance",
    ),
    (
        "sup_routine_change",
        "behaviour",
        "I have noticed changes in my usual daily routine (for example skipping classes or neglecting tasks)",
        "Routine change",
    ),
    (
        "sup_physical_inactivity",
        "lifestyle",
        "I have been physically inactive (little or no exercise or movement)",
        "Physical inactivity",
    ),
    (
        "sup_unable_cope",
        "stress",
        "I have felt unable to cope with the demands placed on me",
        "Unable to cope",
    ),
]

SECTIONS: list[dict[str, Any]] = [
    {
        "id": "phq9",
        "title": "Mood (PHQ-9)",
        "standardized": True,
        "instrument": "PHQ-9",
        "stem": "Over the last 2 weeks, how often have you been bothered by any of the following problems?",
        "time_period": "Last 2 weeks",
        "provenance": "Standard PHQ-9 wording and 0-3 response options (Kroenke, Spitzer & Williams, 2001).",
        "items": [
            {
                "id": i,
                "text": t,
                "short": s,
                "scale": FREQUENCY_SCALE_ID,
                "domain": "depression",
                "safety_critical": i == SELF_HARM_ITEM_ID,
            }
            for i, t, s in _PHQ9
        ],
    },
    {
        "id": "gad7",
        "title": "Worry and anxiety (GAD-7)",
        "standardized": True,
        "instrument": "GAD-7",
        "stem": "Over the last 2 weeks, how often have you been bothered by the following problems?",
        "time_period": "Last 2 weeks",
        "provenance": "Standard GAD-7 wording and 0-3 response options (Spitzer, Kroenke, Williams & Löwe, 2006).",
        "items": [
            {"id": i, "text": t, "short": s, "scale": FREQUENCY_SCALE_ID, "domain": "anxiety", "safety_critical": False}
            for i, t, s in _GAD7
        ],
    },
    {
        "id": "supplementary",
        "title": "Sleep, study and daily life (supplementary)",
        "standardized": False,
        "instrument": "Supplementary behavioural indicators (proposed, not validated)",
        "stem": "Over the last 2 weeks, how often has each of the following been true for you?",
        "time_period": "Last 2 weeks",
        "provenance": (
            "Proposed by this implementation to cover sleep, academic stress, behavioural change and lifestyle themes "
            "from the Phase-I report. Not a validated instrument and not supervisor-approved; configurable."
        ),
        "items": [
            {
                "id": i,
                "text": t,
                "short": s,
                "scale": FREQUENCY_SCALE_ID,
                "domain": d,
                "safety_critical": False,
            }
            for i, d, t, s in _SUPPLEMENTARY
        ],
    },
]

ITEMS: list[dict[str, Any]] = [item for sec in SECTIONS for item in sec["items"]]
ITEM_IDS: list[str] = [i["id"] for i in ITEMS]
ITEM_BY_ID: dict[str, dict[str, Any]] = {i["id"]: i for i in ITEMS}
SECTION_OF_ITEM: dict[str, str] = {i["id"]: sec["id"] for sec in SECTIONS for i in sec["items"]}

# Derived (deterministic, non-fitted) features: simple sums of raw items.
DERIVED_FEATURES: list[dict[str, Any]] = [
    {
        "id": "phq9_total",
        "label": "PHQ-9 total score",
        "short": "PHQ-9 total",
        "items": [i["id"] for i in SECTIONS[0]["items"]],
        "description": "Sum of the nine PHQ-9 items (0-27), the standard PHQ-9 total score.",
    },
    {
        "id": "gad7_total",
        "label": "GAD-7 total score",
        "short": "GAD-7 total",
        "items": [i["id"] for i in SECTIONS[1]["items"]],
        "description": "Sum of the seven GAD-7 items (0-21), the standard GAD-7 total score.",
    },
    {
        "id": "sup_total",
        "label": "Supplementary indicators total",
        "short": "Supplementary total",
        "items": [i["id"] for i in SECTIONS[2]["items"]],
        "description": "Sum of the eight proposed supplementary items (0-24). Not a validated score.",
    },
]
DERIVED_IDS: list[str] = [d["id"] for d in DERIVED_FEATURES]

FEATURE_INFO: dict[str, dict[str, Any]] = {
    **{
        i["id"]: {
            "id": i["id"],
            "label": f'{i["short"]} — "{i["text"]}"' if len(i["text"]) < 70 else i["short"],
            "short": i["short"],
            "description": i["text"],
            "kind": "item",
            "section": SECTION_OF_ITEM[i["id"]],
        }
        for i in ITEMS
    },
    **{
        d["id"]: {
            "id": d["id"],
            "label": d["label"],
            "short": d["short"],
            "description": d["description"],
            "kind": "derived",
            "section": None,
        }
        for d in DERIVED_FEATURES
    },
}

# ---- prediction targets -----------------------------------------------------
TARGETS: list[dict[str, Any]] = [
    {
        "id": "depression_risk",
        "display_name": "Depression risk",
        "required": False,
        "definition": (
            "Low / Moderate / High depression-risk label supplied in the dataset. The definition of the label "
            "(clinician rating, validated scale cut-offs, etc.) comes from the dataset provider and must be "
            "documented in docs/dataset-schema.md when the real dataset is integrated."
        ),
    },
    {
        "id": "anxiety_risk",
        "display_name": "Anxiety risk",
        "required": False,
        "definition": "Low / Moderate / High anxiety-risk label supplied in the dataset (definition from the dataset provider).",
    },
    {
        "id": "stress_risk",
        "display_name": "Stress risk",
        "required": False,
        "definition": (
            "Low / Moderate / High stress-risk label supplied in the dataset (definition from the dataset provider). "
            "No standard scoring rule for stress exists among the instruments used here, so no questionnaire-based "
            "reference score is shown for stress."
        ),
    },
    {
        "id": "overall_risk",
        "display_name": "Overall risk",
        "required": False,
        "definition": (
            "Optional single Low / Moderate / High overall-risk label supplied in the dataset. Only trained and "
            "shown if the dataset contains this column."
        ),
    },
]
TARGET_IDS: list[str] = [t["id"] for t in TARGETS]
TARGET_BY_ID: dict[str, dict[str, Any]] = {t["id"]: t for t in TARGETS}

LEVEL_MEANING: dict[str, str] = {
    "Low": (
        "Screening indicators are in the range the training labels describe as low risk. This is not a guarantee "
        "of good mental health."
    ),
    "Moderate": (
        "Screening indicators are in the range the training labels describe as moderate risk. A conversation with "
        "a counsellor or health professional may be worthwhile."
    ),
    "High": (
        "Screening indicators are in the range the training labels describe as high risk. Speaking with a "
        "qualified mental-health professional is recommended."
    ),
}

# ---- reference scoring (standard published bands; NOT model output) ---------
PHQ9_BANDS = [
    (0, 4, "Minimal"),
    (5, 9, "Mild"),
    (10, 14, "Moderate"),
    (15, 19, "Moderately severe"),
    (20, 27, "Severe"),
]
GAD7_BANDS = [
    (0, 4, "Minimal"),
    (5, 9, "Mild"),
    (10, 14, "Moderate"),
    (15, 21, "Severe"),
]


def band_for(total: int, bands: list[tuple[int, int, str]]) -> str:
    for lo, hi, name in bands:
        if lo <= total <= hi:
            return name
    raise ValueError(f"total {total} outside defined bands")


def public_questionnaire() -> dict[str, Any]:
    """JSON-serialisable questionnaire definition for the API/frontend."""
    return {
        "title": "24-item screening questionnaire",
        "scales": list(SCALES.values()),
        "sections": SECTIONS,
        "item_count": len(ITEM_IDS),
        "validation_notice": (
            "PHQ-9 and GAD-7 sections use the standard published items. The supplementary section is a proposed, "
            "unvalidated set of items. The combined form has not been clinically validated."
        ),
    }


# ======================================================================
# core/config
# ======================================================================
REPO_ROOT = Path(__file__).resolve().parent
MIN_ADMIN_KEY_LENGTH = 16


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    admin_api_key: str | None = None
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    data_dir: Path = REPO_ROOT / "data"
    crisis_resources_file: Path | None = None

    max_upload_mb: int = 5
    max_upload_rows: int = 200_000
    max_json_body_kb: int = 16

    predict_rate_limit_per_min: int = 30
    public_rate_limit_per_min: int = 120
    admin_rate_limit_per_min: int = 120
    admin_failed_auth_per_min: int = 10

    train_n_jobs: int = 1
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def admin_enabled(self) -> bool:
        return bool(self.admin_api_key) and len(self.admin_api_key or "") >= MIN_ADMIN_KEY_LENGTH


@lru_cache
def get_settings() -> Settings:
    return Settings()


def app_settings(request: Request) -> Settings:
    """Dependency: the Settings instance the running app was created with."""
    return request.app.state.settings


# ======================================================================
# core/security
# ======================================================================
class SlidingWindowLimiter:
    """Small in-memory per-key sliding-window counter (single-process deployments)."""

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, q: deque[float], now: float, window: float) -> None:
        while q and now - q[0] > window:
            q.popleft()

    def hit(self, key: str, limit: int, window: float = 60.0) -> tuple[bool, int]:
        """Record an event. Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            q = self._events[key]
            self._prune(q, now, window)
            if len(q) >= limit:
                return False, max(1, int(window - (now - q[0])) + 1)
            q.append(now)
            return True, 0

    def count(self, key: str, window: float = 60.0) -> int:
        now = time.monotonic()
        with self._lock:
            q = self._events[key]
            self._prune(q, now, window)
            return len(q)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


limiter = SlidingWindowLimiter()


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limit(scope: str, setting_name: str):
    """Dependency factory: limit requests per client IP for a scope."""

    def dep(request: Request, settings: Settings = Depends(app_settings)) -> None:
        limit = getattr(settings, setting_name)
        ok, retry = limiter.hit(f"{scope}:{client_ip(request)}", limit)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please wait a moment and try again.",
                headers={"Retry-After": str(retry)},
            )

    return dep


def require_admin(request: Request, settings: Settings = Depends(app_settings)) -> None:
    """Bearer-token guard for every /api/admin route."""
    ip = client_ip(request)
    ok, retry = limiter.hit(f"admin:{ip}", settings.admin_rate_limit_per_min)
    if not ok:
        raise HTTPException(429, "Too many requests.", headers={"Retry-After": str(retry)})

    if not settings.admin_enabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Administration is disabled: ADMIN_API_KEY is not configured (minimum 16 characters).",
        )

    if limiter.count(f"adminfail:{ip}") >= settings.admin_failed_auth_per_min:
        raise HTTPException(429, "Too many failed attempts. Try again later.", headers={"Retry-After": "60"})

    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    valid = scheme.lower() == "bearer" and secrets.compare_digest(
        token.encode(), (settings.admin_api_key or "").encode()
    )
    if not valid:
        limiter.hit(f"adminfail:{ip}", 10_000)
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or missing admin credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ======================================================================
# ml/features
# ======================================================================
def feature_columns(include_derived_totals: bool = True) -> list[str]:
    return list(ITEM_IDS) + (list(DERIVED_IDS) if include_derived_totals else [])


def compute_derived(items: pd.DataFrame) -> pd.DataFrame:
    """Return derived totals. A total is NaN if any contributing item is missing."""
    out = pd.DataFrame(index=items.index)
    for d in DERIVED_FEATURES:
        cols = d["items"]
        out[d["id"]] = items[cols].sum(axis=1, min_count=len(cols))
    return out


def build_feature_frame(df: pd.DataFrame, include_derived_totals: bool = True) -> pd.DataFrame:
    """Select/compute model features in a fixed column order, as float64."""
    missing = [c for c in ITEM_IDS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing questionnaire item columns: {missing}")
    items = df[ITEM_IDS].apply(pd.to_numeric, errors="coerce").astype("float64")
    if include_derived_totals:
        items = pd.concat([items, compute_derived(items)], axis=1)
    return items[feature_columns(include_derived_totals)]


def answers_to_frame(answers: dict[str, int], include_derived_totals: bool) -> pd.DataFrame:
    """Single respondent -> one-row feature frame (same code path as training)."""
    return build_feature_frame(pd.DataFrame([answers]), include_derived_totals)


def totals(answers: dict[str, int]) -> dict[str, int]:
    frame = compute_derived(pd.DataFrame([answers]).astype("float64"))
    return {k: int(v) for k, v in frame.iloc[0].items()}


def is_integer_valued(values: Iterable[float]) -> bool:
    arr = np.asarray(list(values), dtype=float)
    return bool(np.all(np.isfinite(arr)) and np.all(arr == np.round(arr)))


# ======================================================================
# ml/synthetic
# ======================================================================
# item_id -> (loadings on [depression, anxiety, stress], cutpoints)
_DEFAULT_CUTS = (0.35, 1.05, 1.75)
_LOADINGS: dict[str, tuple[float, float, float]] = {
    "phq9_1": (0.9, 0.1, 0.1),
    "phq9_2": (1.0, 0.2, 0.1),
    "phq9_3": (0.6, 0.2, 0.3),
    "phq9_4": (0.7, 0.2, 0.3),
    "phq9_5": (0.5, 0.1, 0.2),
    "phq9_6": (0.8, 0.2, 0.2),
    "phq9_7": (0.7, 0.2, 0.3),
    "phq9_8": (0.5, 0.2, 0.1),
    "phq9_9": (0.6, 0.1, 0.1),
    "gad7_1": (0.2, 1.0, 0.3),
    "gad7_2": (0.2, 1.0, 0.2),
    "gad7_3": (0.2, 0.9, 0.3),
    "gad7_4": (0.1, 0.8, 0.4),
    "gad7_5": (0.1, 0.7, 0.2),
    "gad7_6": (0.2, 0.6, 0.4),
    "gad7_7": (0.2, 0.8, 0.2),
    "sup_sleep_quality": (0.4, 0.3, 0.4),
    "sup_sleep_irregular": (0.2, 0.2, 0.3),
    "sup_academic_overwhelm": (0.2, 0.3, 0.9),
    "sup_academic_worry": (0.1, 0.4, 0.8),
    "sup_social_avoidance": (0.6, 0.3, 0.2),
    "sup_routine_change": (0.5, 0.2, 0.4),
    "sup_physical_inactivity": (0.3, 0.1, 0.2),
    "sup_unable_cope": (0.3, 0.3, 0.9),
}
_CUTS_OVERRIDE = {"phq9_9": (1.3, 1.9, 2.5), "sup_physical_inactivity": (0.3, 0.9, 1.5)}


def _ordinal(z: np.ndarray, cuts: tuple[float, float, float]) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(cuts), z, side="right"), 0, ITEM_MAX)


def _label(latent: np.ndarray, rng: np.random.Generator, noise_sd: float, q: tuple[float, float]) -> np.ndarray:
    noisy = latent + rng.normal(0, noise_sd, size=latent.shape)
    lo, hi = np.quantile(noisy, q)
    return np.where(noisy <= lo, "Low", np.where(noisy <= hi, "Moderate", "High"))


def generate_synthetic_dataset(
    n: int = 1500,
    seed: int = 42,
    include_overall: bool = False,
) -> pd.DataFrame:
    """Generate a labelled SYNTHETIC dataset (columns: 24 items, targets, is_synthetic)."""
    assert [i for s in SECTIONS for i in [it["id"] for it in s["items"]]] == ITEM_IDS
    rng = np.random.default_rng(seed)

    corr = np.array([[1.0, 0.55, 0.50], [0.55, 1.0, 0.60], [0.50, 0.60, 1.0]])
    latent = rng.multivariate_normal(np.zeros(3), corr, size=n)
    d, a, s = latent[:, 0], latent[:, 1], latent[:, 2]

    data: dict[str, np.ndarray] = {}
    for item_id in ITEM_IDS:
        w = np.array(_LOADINGS[item_id])
        z = latent @ w * 0.7 + rng.normal(0, 0.55, size=n)
        data[item_id] = _ordinal(z, _CUTS_OVERRIDE.get(item_id, _DEFAULT_CUTS))

    df = pd.DataFrame(data)
    q = (0.60, 0.85)  # ~60% Low, 25% Moderate, 15% High: deliberately imbalanced
    df["depression_risk"] = _label(d, rng, 0.3, q)
    df["anxiety_risk"] = _label(a, rng, 0.3, q)
    df["stress_risk"] = _label(s, rng, 0.3, q)
    if include_overall:
        df["overall_risk"] = _label((d + a + s) / 3, rng, 0.35, q)
    df["is_synthetic"] = 1
    return df


# ======================================================================
# ml/data
# ======================================================================
GROUP_COL = "participant_id"
SYNTHETIC_COL = "is_synthetic"
KEEP_OPTIONAL = [GROUP_COL, SYNTHETIC_COL]
MIN_CLASS_COUNT = 10  # per class, per target, to be trainable with 5-fold CV
LEAKAGE_ACC = 0.98
_PII_PATTERN = re.compile(r"(name|e-?mail|phone|mobile|contact|roll|enrol|student_?id|aadhaar|address|dob|birth)", re.I)


class DatasetError(ValueError):
    """Raised for unreadable datasets (not for ordinary validation findings)."""


def read_csv_bytes(raw: bytes, max_rows: int) -> pd.DataFrame:
    """Parse CSV bytes defensively. Only pandas' CSV parser touches the content."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError("File is not valid UTF-8 text. Export the dataset as UTF-8 CSV.") from exc
    try:
        df = pd.read_csv(io.StringIO(text), nrows=max_rows + 1, skipinitialspace=True)
    except Exception as exc:  # pandas raises many parser errors
        raise DatasetError(f"Could not parse CSV: {exc}") from exc
    if len(df) > max_rows:
        raise DatasetError(f"Dataset has more than the allowed {max_rows} rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalise_labels(series: pd.Series) -> pd.Series:
    mapping = {lv.lower(): lv for lv in LEVELS}
    return series.astype("string").str.strip().str.lower().map(mapping)


def sanitise(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only schema columns; normalise target labels. Raises on missing item columns."""
    keep = [c for c in ITEM_IDS if c in df.columns]
    keep += [c for c in TARGET_IDS if c in df.columns]
    keep += [c for c in KEEP_OPTIONAL if c in df.columns]
    out = df[keep].copy()
    for t in TARGET_IDS:
        if t in out.columns:
            out[t] = normalise_labels(out[t])
    return out


def _single_feature_leakage(features: pd.DataFrame, y: pd.Series) -> list[tuple[str, float]]:
    hits: list[tuple[str, float]] = []
    X = features.fillna(features.median(numeric_only=True))
    for col in X.columns:
        if X[col].nunique() < 2:
            continue
        try:
            acc = cross_val_score(
                DecisionTreeClassifier(max_depth=3, random_state=0), X[[col]], y, cv=3, scoring="accuracy"
            ).mean()
        except ValueError:
            continue
        if acc >= LEAKAGE_ACC:
            hits.append((col, float(acc)))
    return hits


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Inspect a raw dataframe. Returns a JSON-serialisable report.

    ``ok`` is True only when no errors were found and at least one target is trainable.
    """
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(map(str, df.columns)),
    }

    pii = [c for c in df.columns if _PII_PATTERN.search(str(c)) and c not in ITEM_IDS + TARGET_IDS + KEEP_OPTIONAL]
    if pii:
        warnings.append(
            f"Columns {pii} look like personal identifiers. They are ignored and will NOT be stored, but remove them "
            "from the file before upload."
        )
    extra = [c for c in df.columns if c not in ITEM_IDS + TARGET_IDS + KEEP_OPTIONAL and c not in pii]
    if extra:
        warnings.append(f"Columns not in the schema will be ignored and not stored: {extra}")

    missing_items = [c for c in ITEM_IDS if c not in df.columns]
    if missing_items:
        errors.append(f"Missing required item columns ({len(missing_items)}): {missing_items}")
    present_targets = [t for t in TARGET_IDS if t in df.columns]
    if not present_targets:
        errors.append(f"No target column found. Provide at least one of: {TARGET_IDS}")
    report["targets_present"] = present_targets
    report["conditions_covered"] = [t.replace("_risk", "") for t in present_targets if t != "overall_risk"]
    missing_conditions = [t for t in TARGET_IDS[:3] if t not in df.columns]
    if present_targets and missing_conditions:
        warnings.append(
            f"Targets not in the dataset: {missing_conditions}. The system will not predict these conditions."
        )

    if len(df) == 0:
        errors.append("Dataset has no rows.")

    if errors:
        report.update(ok=False, errors=errors, warnings=warnings, trainable_targets=[], blocked_targets={})
        return report

    # --- item checks
    items = df[ITEM_IDS].apply(pd.to_numeric, errors="coerce")
    non_numeric = [c for c in ITEM_IDS if df[c].notna().sum() != items[c].notna().sum()]
    if non_numeric:
        errors.append(f"Non-numeric values in item columns: {non_numeric}")
    out_of_range = [c for c in ITEM_IDS if ((items[c] < ITEM_MIN) | (items[c] > ITEM_MAX)).any()]
    if out_of_range:
        errors.append(f"Item values outside {ITEM_MIN}-{ITEM_MAX} in: {out_of_range}")
    non_int = [c for c in ITEM_IDS if (items[c].dropna() != items[c].dropna().round()).any()]
    if non_int:
        errors.append(f"Non-integer item values in: {non_int}")

    miss = df.isna().sum()
    report["missing_values"] = {str(k): int(v) for k, v in miss.items() if v > 0}
    miss_pct = items.isna().mean()
    high_missing = [c for c in ITEM_IDS if miss_pct[c] > 0.2]
    if high_missing:
        warnings.append(f"Item columns with >20% missing values: {high_missing}")
    elif report["missing_values"]:
        warnings.append("Some values are missing; they are median-imputed inside the pipeline (fit on training data only).")

    dups = int(df.duplicated().sum())
    report["duplicate_rows"] = dups
    if dups:
        warnings.append(
            f"{dups} rows are exact duplicates of another row. With ordinal items these can be genuinely identical "
            "answer patterns from different respondents, or accidental repeats. If they are accidental, remove them "
            "to avoid train/test overlap; add a pseudonymous participant_id to make the split participant-safe."
        )
    if GROUP_COL in df.columns:
        n_part = int(df[GROUP_COL].nunique())
        report["n_participants"] = n_part
        if n_part < len(df):
            warnings.append(
                f"{GROUP_COL} repeats ({n_part} participants for {len(df)} rows): participant-grouped splits will be used."
            )

    # --- target checks
    trainable: list[str] = []
    blocked: dict[str, str] = {}
    class_dist: dict[str, dict[str, int]] = {}
    for t in present_targets:
        raw = df[t]
        norm = normalise_labels(raw)
        bad = int(raw.notna().sum() - norm.notna().sum())
        if bad:
            errors.append(f"{t}: {bad} values are not one of {LEVELS} (case-insensitive).")
            blocked[t] = "invalid labels"
            continue
        n_missing_lab = int(norm.isna().sum())
        if n_missing_lab:
            warnings.append(f"{t}: {n_missing_lab} rows have no label and will be excluded from that target.")
        counts = norm.value_counts().reindex(LEVELS).fillna(0).astype(int)
        class_dist[t] = {lv: int(counts[lv]) for lv in LEVELS}
        present = [lv for lv in LEVELS if counts[lv] > 0]
        if len(present) < 2:
            blocked[t] = f"only one class present ({present}); cannot train a classifier"
            continue
        small = {lv: int(counts[lv]) for lv in present if counts[lv] < MIN_CLASS_COUNT}
        if small:
            blocked[t] = f"classes with fewer than {MIN_CLASS_COUNT} samples: {small}"
            continue
        trainable.append(t)
        if len(present) < 3:
            warnings.append(f"{t}: only classes {present} present; predictions will be limited to those.")
        if counts.min() / counts.max() < 0.2:
            warnings.append(
                f"{t}: strong class imbalance ({dict(counts)}). Class weighting is evaluated during training; "
                "review per-class recall, not accuracy alone."
            )
    report["class_distribution"] = class_dist
    report["trainable_targets"] = trainable
    report["blocked_targets"] = blocked
    if not trainable:
        errors.append("No target is trainable. " + "; ".join(f"{k}: {v}" for k, v in blocked.items()))

    # --- leakage heuristics (only when the data are otherwise valid)
    leak_msgs: list[str] = []
    if not errors:
        feats = build_feature_frame(df, include_derived_totals=True)
        for t in trainable:
            y = normalise_labels(df[t])
            mask = y.notna()
            hits = _single_feature_leakage(feats[mask], y[mask].astype(str))
            for col, acc in hits:
                leak_msgs.append(
                    f"{t}: feature '{col}' alone reaches {acc:.1%} cross-validated accuracy. The label may be defined "
                    f"from this feature (label leakage)."
                    + (" Train with include_derived_totals=false." if col in DERIVED_IDS else "")
                )
        tcols = [t for t in present_targets if t in trainable]
        for i, a in enumerate(tcols):
            for b in tcols[i + 1 :]:
                if (normalise_labels(df[a]) == normalise_labels(df[b])).mean() > 0.99:
                    leak_msgs.append(f"Targets {a} and {b} are identical in >99% of rows; check the label construction.")
    report["leakage_warnings"] = leak_msgs
    warnings.extend(leak_msgs)

    # --- synthetic flag
    if SYNTHETIC_COL in df.columns:
        flag = pd.to_numeric(df[SYNTHETIC_COL], errors="coerce").fillna(0)
        report["is_synthetic"] = bool((flag == 1).all())
        if 0 < (flag == 1).sum() < len(df):
            warnings.append("Dataset mixes synthetic and non-synthetic rows; it will be treated as non-synthetic.")
    else:
        report["is_synthetic"] = None

    report.update(ok=not errors, errors=errors, warnings=warnings)
    return report


def target_matrix(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Rows with a label for ``target`` -> (frame, string labels, ordered classes present)."""
    y = normalise_labels(df[target])
    mask = y.notna().to_numpy()
    classes = [lv for lv in LEVELS if (y[mask] == lv).any()]
    return df.loc[mask], y[mask].astype(str).reset_index(drop=True), classes


def encode(y: pd.Series, classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    return y.map(idx).to_numpy(dtype=int)


# ======================================================================
# ml/explain
# ======================================================================
def feature_short(name: str) -> str:
    return FEATURE_INFO.get(name, {}).get("short", name)


def condition_text(feature: str, threshold: float, go_left: bool) -> str:
    """Human-readable condition. Integer-valued features show as <= k / >= k+1."""
    label = feature_short(feature)
    frac = threshold - math.floor(threshold)
    if abs(frac - 0.5) < 1e-9:
        k = math.floor(threshold)
        return f"{label} ≤ {k}" if go_left else f"{label} ≥ {k + 1}"
    return f"{label} ≤ {threshold:.2f}" if go_left else f"{label} > {threshold:.2f}"


def node_class_counts(clf, X_train_transformed: np.ndarray, y_train: np.ndarray, n_classes: int) -> np.ndarray:
    """Raw (unweighted) training samples per node and class: shape (n_nodes, n_classes)."""
    indicator = clf.decision_path(X_train_transformed)  # csr: samples x nodes
    onehot = np.eye(n_classes, dtype=np.int64)[y_train]
    return np.asarray(indicator.T @ onehot, dtype=np.int64)


def node_prediction(clf, node_id: int, classes: list[str]) -> str:
    """Class the estimator predicts for samples ending in this node (argmax of tree_.value)."""
    idx = int(np.argmax(clf.tree_.value[node_id][0]))
    return classes[int(clf.classes_[idx])]


def _counts_dict(counts_row: np.ndarray, classes: list[str]) -> dict[str, int]:
    return {c: int(counts_row[k]) for k, c in enumerate(classes)}


def tree_to_dict(clf, feature_names: list[str], classes: list[str], counts: np.ndarray) -> dict[str, Any]:
    t = clf.tree_

    def rec(i: int, depth: int) -> dict[str, Any]:
        is_leaf = t.children_left[i] == -1
        node: dict[str, Any] = {
            "id": int(i),
            "depth": depth,
            "n_samples": int(counts[i].sum()),
            "class_counts": _counts_dict(counts[i], classes),
            "prediction": node_prediction(clf, i, classes),
            "is_leaf": bool(is_leaf),
        }
        if not is_leaf:
            f = feature_names[int(t.feature[i])]
            thr = float(t.threshold[i])
            node.update(
                feature=f,
                feature_label=feature_short(f),
                threshold=thr,
                left=rec(int(t.children_left[i]), depth + 1),
                right=rec(int(t.children_right[i]), depth + 1),
            )
        return node

    return rec(0, 0)


def extract_rules(clf, feature_names: list[str], classes: list[str], counts: np.ndarray) -> list[dict[str, Any]]:
    """One if-then rule per leaf (conjunction of conditions from the root)."""
    t = clf.tree_
    rules: list[dict[str, Any]] = []

    def rec(i: int, conds: list[dict[str, Any]]) -> None:
        if t.children_left[i] == -1:
            n = int(counts[i].sum())
            rules.append(
                {
                    "leaf_id": int(i),
                    "conditions": conds,
                    "text": "IF " + " AND ".join(c["text"] for c in conds) if conds else "ALWAYS",
                    "prediction": node_prediction(clf, i, classes),
                    "n_samples": n,
                    "class_counts": _counts_dict(counts[i], classes),
                    "class_share": {c: (float(counts[i][k]) / n if n else 0.0) for k, c in enumerate(classes)},
                }
            )
            return
        f = feature_names[int(t.feature[i])]
        thr = float(t.threshold[i])
        for go_left, child in ((True, int(t.children_left[i])), (False, int(t.children_right[i]))):
            rec(
                child,
                conds
                + [
                    {
                        "feature": f,
                        "operator": "<=" if go_left else ">",
                        "threshold": thr,
                        "text": condition_text(f, thr, go_left),
                    }
                ],
            )

    rec(0, [])
    return rules


def decision_path_steps(
    clf,
    x_transformed: np.ndarray,
    feature_names: list[str],
    classes: list[str],
    counts: np.ndarray,
) -> dict[str, Any]:
    """Trace one sample through the fitted tree using clf.decision_path.

    ``x_transformed`` is the (1, n_features) array the estimator itself receives.
    """
    t = clf.tree_
    x = np.asarray(x_transformed, dtype=np.float32).reshape(1, -1)
    node_ids = [int(n) for n in clf.decision_path(x).indices]
    steps: list[dict[str, Any]] = []
    for n in node_ids:
        if t.children_left[n] == -1:
            continue
        f_idx = int(t.feature[n])
        f = feature_names[f_idx]
        thr = float(t.threshold[n])
        value = float(x[0, f_idx])
        go_left = bool(value <= thr)
        steps.append(
            {
                "node_id": n,
                "feature": f,
                "feature_label": feature_short(f),
                "value": value,
                "threshold": thr,
                "direction": "left" if go_left else "right",
                "satisfied_condition": condition_text(f, thr, go_left),
                "text": f"{condition_text(f, thr, go_left)} (your value: {value:g})",
            }
        )
    leaf = node_ids[-1]
    n = int(counts[leaf].sum())
    return {
        "node_ids": node_ids,
        "steps": steps,
        "leaf_id": leaf,
        "prediction": node_prediction(clf, leaf, classes),
        "leaf_evidence": {
            "n_train_samples": n,
            "class_counts": _counts_dict(counts[leaf], classes),
            "class_share": {c: (float(counts[leaf][k]) / n if n else 0.0) for k, c in enumerate(classes)},
            "description": (
                "Raw (unweighted) counts of training-set respondents who ended in this leaf, by their dataset label. "
                "This is descriptive evidence about the training data, not a calibrated probability for you."
            ),
        },
    }


# ======================================================================
# ml/training
# ======================================================================
SELECTION_METRICS = ("f1_macro", "recall_macro", "balanced_accuracy")
SCORING = {
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "balanced_accuracy": "balanced_accuracy",
}
MODEL_ORDER = ["cart", "random_forest", "xgboost", "lightgbm", "logistic_regression"]
MODEL_DISPLAY = {
    "cart": "CART Decision Tree (pruned)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "logistic_regression": "Logistic Regression",
}
CCP_ALPHAS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
SCHEMA_VERSION = 1

METRIC_DEFINITIONS = {
    "accuracy": "Fraction of held-out test respondents classified correctly.",
    "balanced_accuracy": "Mean of per-class recall (robust to class imbalance).",
    "precision_macro": "Unweighted mean of per-class precision (each class counts equally).",
    "recall_macro": "Unweighted mean of per-class recall (each class counts equally).",
    "f1_macro": "Unweighted mean of per-class F1.",
    "roc_auc_ovr_macro": (
        "One-vs-rest ROC-AUC averaged over classes using predict_proba (binary targets: standard ROC-AUC). "
        "Null when the test split lacks a class. Tree/ensemble probabilities are not calibrated."
    ),
    "per_class": "Precision, recall, F1 and support per class (class order as listed in `classes`).",
    "confusion_matrix": "Rows = true class, columns = predicted class, on the held-out test split.",
}


@dataclass
class TrainOptions:
    test_size: float = 0.2
    seed: int = 42
    selection_metric: str = "f1_macro"
    include_derived_totals: bool = True
    n_jobs: int = 1
    cv_folds: int = 5
    models: list[str] = field(default_factory=lambda: list(MODEL_ORDER))
    targets: list[str] | None = None

    def validate(self) -> None:
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
        if not 0.1 <= self.test_size <= 0.5:
            raise ValueError("test_size must be between 0.1 and 0.5")
        if self.cv_folds < 2 or self.cv_folds > 10:
            raise ValueError("cv_folds must be between 2 and 10")
        unknown = [m for m in self.models if m not in MODEL_ORDER]
        if unknown:
            raise ValueError(f"unknown models: {unknown}")
        if "cart" not in self.models:
            raise ValueError("the primary 'cart' model must be included")


@dataclass
class TrainResult:
    report: dict[str, Any]
    bundle: dict[str, Any]
    explain: dict[str, Any]


def _py(obj: Any) -> Any:
    """Recursively convert numpy types to plain Python for JSON."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _py(obj.tolist())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _make_pipe(clf, scale: bool = False) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def _model_specs(seed: int) -> dict[str, dict[str, Any]]:
    """Estimator + hyper-parameter grid per model. Missing optional libs -> skipped spec."""
    specs: dict[str, dict[str, Any]] = {}
    specs["cart"] = {
        "pipe": _make_pipe(DecisionTreeClassifier(random_state=seed)),
        "grid": {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [3, 4, 5, 6],
            "clf__min_samples_leaf": [1, 5, 10, 20],
            "clf__class_weight": [None, "balanced"],
            "clf__ccp_alpha": CCP_ALPHAS,
        },
    }
    specs["random_forest"] = {
        "pipe": _make_pipe(RandomForestClassifier(random_state=seed, n_jobs=1)),
        "grid": {
            "clf__n_estimators": [200],
            "clf__max_depth": [None, 8],
            "clf__min_samples_leaf": [1, 5],
            "clf__class_weight": [None, "balanced"],
        },
    }
    try:
        from xgboost import XGBClassifier

        specs["xgboost"] = {
            "pipe": _make_pipe(
                XGBClassifier(tree_method="hist", eval_metric="mlogloss", random_state=seed, n_jobs=1, verbosity=0)
            ),
            "grid": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.1],
            },
            "balanced_sample_weight": True,
        }
    except Exception as exc:  # ImportError or XGBoostError (missing native lib)
        specs["xgboost"] = {"skipped": f"xgboost unavailable: {type(exc).__name__}: {exc}"}
    try:
        from lightgbm import LGBMClassifier

        specs["lightgbm"] = {
            "pipe": _make_pipe(
                LGBMClassifier(class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1)
            ),
            "grid": {
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [15, 31],
                "clf__learning_rate": [0.1],
            },
        }
    except Exception as exc:
        specs["lightgbm"] = {"skipped": f"lightgbm unavailable: {type(exc).__name__}: {exc}"}
    specs["logistic_regression"] = {
        "pipe": _make_pipe(LogisticRegression(max_iter=2000, random_state=seed), scale=True),
        "grid": {"clf__C": [0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
    }
    return specs


def _split(
    df: pd.DataFrame, targets: list[str], opts: TrainOptions, warnings: list[str]
) -> tuple[np.ndarray, np.ndarray, str]:
    n = len(df)
    labels = pd.DataFrame(
        {t: normalise_labels(df[t]).astype(object).where(normalise_labels(df[t]).notna(), "NA") for t in targets}
    )
    combo = labels.astype(str).agg("|".join, axis=1)
    stratify_on = "+".join(targets)
    if combo.value_counts().min() < 2:
        combo = labels[targets[0]].astype(str)
        stratify_on = targets[0]
        warnings.append(
            f"Rare joint label combinations; test split stratified on '{targets[0]}' only."
        )
    codes = pd.factorize(combo)[0]
    groups = df[GROUP_COL] if GROUP_COL in df.columns else None
    if groups is not None and groups.nunique() < n:
        n_splits = max(2, int(round(1 / opts.test_size)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=opts.seed)
        train_idx, test_idx = next(sgkf.split(np.zeros(n), codes, groups.to_numpy()))
        strategy = (
            f"participant-grouped: first fold of StratifiedGroupKFold(n_splits={n_splits}, shuffle, "
            f"seed={opts.seed}); stratified on {stratify_on}"
        )
    else:
        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=opts.test_size, stratify=codes, random_state=opts.seed
        )
        strategy = f"stratified train_test_split(test_size={opts.test_size}, seed={opts.seed}); stratified on {stratify_on}"
    return np.sort(train_idx), np.sort(test_idx), strategy


def _metrics(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, classes: list[str], warnings: list[str], tag: str) -> dict:
    y_pred = pipe.predict(X)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y, y_pred, labels=labels, zero_division=0)
    present = np.array([(y == k).any() for k in labels])
    auc: float | None
    try:
        proba = pipe.predict_proba(X)
        if len(classes) == 2:
            auc = float(roc_auc_score(y, proba[:, 1]))
        else:
            auc = float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels))
    except ValueError as exc:
        auc = None
        warnings.append(f"{tag}: ROC-AUC not computed ({exc}).")
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "precision_macro": float(p[present].mean()),
        "recall_macro": float(r[present].mean()),
        "f1_macro": float(f[present].mean()),
        "roc_auc_ovr_macro": auc,
        "per_class": {
            c: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k])}
            for k, c in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "n_test": int(len(y)),
    }


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k.replace("clf__", ""): _py(v) for k, v in params.items()}


def _dist(y: pd.Series | np.ndarray, classes: list[str], encoded: bool = False) -> dict[str, int]:
    arr = np.asarray(y)
    if encoded:
        return {c: int((arr == k).sum()) for k, c in enumerate(classes)}
    return {c: int((arr == c).sum()) for c in classes}


def train_and_evaluate(
    df: pd.DataFrame,
    dataset_meta: dict[str, Any],
    opts: TrainOptions | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> TrainResult:
    """Run the full protocol on a sanitised dataframe and return report, bundle and explanations."""
    opts = opts or TrainOptions()
    opts.validate()
    progress = progress or (lambda f, m: None)
    started = time.perf_counter()
    warnings: list[str] = []

    validation = validate_dataframe(df)
    if not validation["ok"]:
        raise ValueError("Dataset failed validation: " + " | ".join(validation["errors"]))
    trainable = validation["trainable_targets"]
    targets = [t for t in (opts.targets or TARGET_IDS) if t in trainable]
    if not targets:
        raise ValueError("None of the requested targets is trainable in this dataset.")
    skipped_targets = {t: validation["blocked_targets"].get(t, "not present in dataset") for t in TARGET_IDS if t not in targets}
    warnings.extend(validation["warnings"])

    df = df.reset_index(drop=True)
    X_all = build_feature_frame(df, opts.include_derived_totals)
    cols = feature_columns(opts.include_derived_totals)
    train_idx, test_idx, strategy = _split(df, targets, opts, warnings)
    specs = _model_specs(opts.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:4]

    is_synth = bool(dataset_meta.get("is_synthetic"))
    if is_synth:
        warnings.append(
            "SYNTHETIC DATA: every metric in this report describes software behaviour on generated data and says "
            "nothing about real students or real predictive performance."
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            **{k: dataset_meta.get(k) for k in ("id", "name", "provenance", "sha256")},
            "is_synthetic": is_synth,
            "n_rows": int(len(df)),
            "n_participants": validation.get("n_participants"),
            "duplicate_rows": validation.get("duplicate_rows"),
            "missing_values": validation.get("missing_values", {}),
            "conditions_covered": validation.get("conditions_covered", []),
        },
        "options": asdict(opts),
        "feature_schema": {
            "columns": cols,
            "item_range": [0, 3],
            "derived_totals_included": opts.include_derived_totals,
            "feature_info": {c: FEATURE_INFO[c] for c in cols},
        },
        "preprocessing": [
            "Questionnaire items coerced to numeric (0-3).",
            "Derived totals (PHQ-9, GAD-7, supplementary) = deterministic sums of raw items"
            if opts.include_derived_totals
            else "Derived totals NOT used (raw items only).",
            "Missing values: median imputation, fitted on the training split only (inside the pipeline).",
            "Standard scaling for Logistic Regression only (inside its pipeline).",
            "Class imbalance: class_weight='balanced' evaluated as a hyper-parameter (balanced sample weights for "
            "XGBoost). No resampling; the test split is untouched.",
        ],
        "split": {
            "strategy": strategy,
            "test_size": opts.test_size,
            "seed": opts.seed,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "participant_grouped": bool(GROUP_COL in df.columns and df[GROUP_COL].nunique() < len(df)),
        },
        "tuning": {
            "method": "GridSearchCV on the training split only",
            "cv_folds_requested": opts.cv_folds,
            "selection_metric": opts.selection_metric,
            "recorded_cv_metrics": list(SCORING),
            "ccp_alpha_grid": CCP_ALPHAS,
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "software_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "targets": {},
        "skipped_targets": skipped_targets,
        "warnings": warnings,
    }
    for lib in ("xgboost", "lightgbm"):
        try:
            report["software_versions"][lib] = __import__(lib).__version__
        except Exception:
            report["software_versions"][lib] = None

    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "feature_columns": cols,
        "include_derived_totals": opts.include_derived_totals,
        "is_synthetic": is_synth,
        "targets": {},
    }
    explain: dict[str, Any] = {"run_id": run_id, "is_synthetic": is_synth, "targets": {}}

    total_steps = len(targets) * len(opts.models)
    done = 0
    for t in targets:
        t_df, y_str, classes = target_matrix(df, t)
        row_pos = np.flatnonzero(normalise_labels(df[t]).notna().to_numpy())  # positions in df with a label
        tr_rows = np.intersect1d(row_pos, train_idx)
        te_rows = np.intersect1d(row_pos, test_idx)
        y_full = pd.Series(normalise_labels(df[t]).astype("string"))
        y_tr = encode(y_full.iloc[tr_rows].astype(str).reset_index(drop=True), classes)
        y_te = encode(y_full.iloc[te_rows].astype(str).reset_index(drop=True), classes)
        X_tr, X_te = X_all.iloc[tr_rows], X_all.iloc[te_rows]
        groups_tr = df[GROUP_COL].iloc[tr_rows].to_numpy() if GROUP_COL in df.columns else None

        min_count = int(np.bincount(y_tr, minlength=len(classes)).min())
        n_splits = min(opts.cv_folds, min_count)
        if n_splits < 2:
            raise ValueError(f"{t}: too few samples in the smallest class of the training split for cross-validation.")
        if groups_tr is not None and len(np.unique(groups_tr)) < len(groups_tr):
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=opts.seed)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=opts.seed)
            groups_tr = None

        t_report: dict[str, Any] = {
            "display_name": TARGET_BY_ID[t]["display_name"],
            "target_definition": TARGET_BY_ID[t]["definition"],
            "classes": classes,
            "class_distribution": {
                "all": _dist(y_str, classes),
                "train": _dist(y_tr, classes, encoded=True),
                "test": _dist(y_te, classes, encoded=True),
            },
            "cv_folds": n_splits,
            "primary_model": "cart",
            "models": {},
        }
        report["targets"][t] = t_report

        for key in MODEL_ORDER:
            if key not in opts.models:
                continue
            spec = specs[key]
            done += 1
            progress((done - 1) / total_steps, f"{TARGET_BY_ID[t]['display_name']}: {MODEL_DISPLAY[key]}")
            entry: dict[str, Any] = {"display_name": MODEL_DISPLAY[key]}
            if "skipped" in spec:
                entry.update(status="skipped", reason=spec["skipped"])
                warnings.append(f"{MODEL_DISPLAY[key]} skipped: {spec['skipped']}")
                t_report["models"][key] = entry
                continue
            try:
                t0 = time.perf_counter()
                gs = GridSearchCV(
                    spec["pipe"],
                    spec["grid"],
                    scoring=SCORING,
                    refit=opts.selection_metric,
                    cv=cv,
                    n_jobs=opts.n_jobs,
                    error_score="raise",
                )
                fit_kwargs: dict[str, Any] = {}
                if groups_tr is not None:
                    fit_kwargs["groups"] = groups_tr
                if spec.get("balanced_sample_weight"):
                    fit_kwargs["clf__sample_weight"] = compute_sample_weight("balanced", y_tr)
                gs.fit(X_tr, y_tr, **fit_kwargs)
                fit_s = time.perf_counter() - t0
                best = gs.best_index_
                cvres = gs.cv_results_
                entry.update(
                    status="ok",
                    selected_params=_clean_params(gs.best_params_),
                    n_candidates=int(len(cvres["params"])),
                    cv={
                        m: {
                            "mean": float(cvres[f"mean_test_{m}"][best]),
                            "std": float(cvres[f"std_test_{m}"][best]),
                        }
                        for m in SCORING
                    },
                    metrics=_metrics(gs.best_estimator_, X_te, y_te, classes, warnings, f"{t}/{key}"),
                    fit_seconds=round(fit_s, 2),
                )
                if key == "cart":
                    final_pipe = gs.best_estimator_
                    clf = final_pipe.named_steps["clf"]
                    Xt_tr = final_pipe[:-1].transform(X_tr)
                    counts = node_class_counts(clf, Xt_tr, y_tr, len(classes))
                    entry["tree"] = {
                        "n_leaves": int(clf.get_n_leaves()),
                        "depth": int(clf.get_depth()),
                        "n_nodes": int(clf.tree_.node_count),
                    }
                    pi = permutation_importance(
                        final_pipe, X_te, y_te, scoring=SCORING["f1_macro"], n_repeats=10, random_state=opts.seed, n_jobs=1
                    )
                    importance = [
                        {
                            "feature": c,
                            "label": FEATURE_INFO[c]["short"],
                            "impurity": float(clf.feature_importances_[i]),
                            "permutation_mean": float(pi.importances_mean[i]),
                            "permutation_std": float(pi.importances_std[i]),
                        }
                        for i, c in enumerate(cols)
                    ]
                    importance.sort(key=lambda r: (-r["impurity"], r["feature"]))
                    bundle["targets"][t] = {
                        "classes": classes,
                        "pipeline": final_pipe,
                        "node_counts": counts,
                        "n_train": int(len(y_tr)),
                    }
                    explain["targets"][t] = {
                        "display_name": TARGET_BY_ID[t]["display_name"],
                        "classes": classes,
                        "n_leaves": entry["tree"]["n_leaves"],
                        "depth": entry["tree"]["depth"],
                        "selected_params": entry["selected_params"],
                        "tree": tree_to_dict(clf, cols, classes, counts),
                        "rules": extract_rules(clf, cols, classes, counts),
                        "feature_importance": importance,
                        "notes": (
                            "Impurity importance is computed on training data and favours high-cardinality features; "
                            "permutation importance is the drop in macro-F1 on the held-out test split when a feature "
                            "is shuffled. Both describe predictive association, not causation."
                        ),
                    }
            except Exception as exc:
                if key == "cart":
                    raise
                entry.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                warnings.append(f"{MODEL_DISPLAY[key]} failed for {t}: {exc}")
            t_report["models"][key] = entry

    progress(1.0, "Finished")
    report["duration_seconds"] = round(time.perf_counter() - started, 1)
    report["warnings"] = list(dict.fromkeys(warnings))
    return TrainResult(report=_py(report), bundle=bundle, explain=_py(explain))


def report_to_csv(report: dict[str, Any]) -> str:
    """Flatten a report to CSV: one row per target x model."""
    rows = []
    for t, tr in report.get("targets", {}).items():
        for key, m in tr["models"].items():
            row: dict[str, Any] = {
                "run_id": report["run_id"],
                "dataset_id": report["dataset"].get("id"),
                "synthetic_data": report["dataset"].get("is_synthetic"),
                "target": t,
                "model": m["display_name"],
                "status": m["status"],
                "reason": m.get("reason", ""),
            }
            if m["status"] == "ok":
                mt = m["metrics"]
                for k in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"):
                    row[k] = mt[k]
                for lv in LEVELS:
                    pc = mt["per_class"].get(lv)
                    row[f"precision_{lv}"] = pc["precision"] if pc else ""
                    row[f"recall_{lv}"] = pc["recall"] if pc else ""
                    row[f"f1_{lv}"] = pc["f1"] if pc else ""
                row["n_test"] = mt["n_test"]
                row["cv_selection_score"] = m["cv"][report["tuning"]["selection_metric"]]["mean"]
                row["fit_seconds"] = m["fit_seconds"]
                row["selected_params"] = str(m["selected_params"])
            rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False)


# ======================================================================
# services/registry
# ======================================================================
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{4}$")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class ActiveModel:
    run_id: str
    bundle: dict[str, Any]
    report: dict[str, Any]
    explain: dict[str, Any]


class RunNotFound(KeyError):
    pass


class Registry:
    def __init__(self, artifacts_dir: Path) -> None:
        self.root = Path(artifacts_dir)
        self.runs_dir = self.root / "runs"
        self.active_file = self.root / "active_run.json"
        self._lock = threading.Lock()
        self._cache: ActiveModel | None = None

    # -- helpers
    def _run_dir(self, run_id: str) -> Path:
        if not RUN_ID_RE.match(run_id):
            raise RunNotFound(run_id)
        d = self.runs_dir / run_id
        if not d.is_dir():
            raise RunNotFound(run_id)
        return d

    # -- write
    def save_run(self, result: TrainResult, activate: bool | None = None) -> dict[str, Any]:
        run_id = result.report["run_id"]
        with self._lock:
            d = self.runs_dir / run_id
            if d.exists():
                raise FileExistsError(f"run {run_id} already exists")
            d.mkdir(parents=True)
            _atomic_write_text(d / "report.json", json.dumps(result.report, indent=2))
            _atomic_write_text(d / "explain.json", json.dumps(result.explain))
            joblib.dump(result.bundle, d / "bundle.joblib")
            should_activate = (self.active_run_id() is None) if activate is None else activate
            if should_activate:
                self._set_active(run_id)
        return {"run_id": run_id, "activated": should_activate}

    def _set_active(self, run_id: str) -> None:
        _atomic_write_text(self.active_file, json.dumps({"run_id": run_id}))
        self._cache = None

    def activate(self, run_id: str) -> None:
        self._run_dir(run_id)
        with self._lock:
            self._set_active(run_id)

    # -- read
    def active_run_id(self) -> str | None:
        if not self.active_file.exists():
            return None
        try:
            rid = json.loads(self.active_file.read_text(encoding="utf-8")).get("run_id")
            return rid if rid and RUN_ID_RE.match(rid) and (self.runs_dir / rid).is_dir() else None
        except (json.JSONDecodeError, OSError):
            return None

    def get_report(self, run_id: str) -> dict[str, Any]:
        return json.loads((self._run_dir(run_id) / "report.json").read_text(encoding="utf-8"))

    def get_explain(self, run_id: str) -> dict[str, Any]:
        return json.loads((self._run_dir(run_id) / "explain.json").read_text(encoding="utf-8"))

    def list_runs(self) -> list[dict[str, Any]]:
        out = []
        if not self.runs_dir.is_dir():
            return out
        active = self.active_run_id()
        for d in sorted(self.runs_dir.iterdir(), reverse=True):
            if not RUN_ID_RE.match(d.name) or not (d / "report.json").exists():
                continue
            try:
                r = json.loads((d / "report.json").read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            primary = {}
            for t, tr in r.get("targets", {}).items():
                m = tr["models"].get("cart", {})
                if m.get("status") == "ok":
                    primary[t] = {
                        k: m["metrics"][k] for k in ("accuracy", "recall_macro", "f1_macro", "balanced_accuracy")
                    }
            out.append(
                {
                    "run_id": r["run_id"],
                    "created_at": r["created_at"],
                    "active": r["run_id"] == active,
                    "dataset": {k: r["dataset"].get(k) for k in ("id", "name", "is_synthetic", "n_rows")},
                    "targets": list(r.get("targets", {})),
                    "selection_metric": r["tuning"]["selection_metric"],
                    "cart_metrics": primary,
                    "duration_seconds": r.get("duration_seconds"),
                }
            )
        return out

    def load_active(self) -> ActiveModel | None:
        with self._lock:
            rid = self.active_run_id()
            if rid is None:
                self._cache = None
                return None
            if self._cache and self._cache.run_id == rid:
                return self._cache
            d = self.runs_dir / rid
            try:
                self._cache = ActiveModel(
                    run_id=rid,
                    bundle=joblib.load(d / "bundle.joblib"),
                    report=json.loads((d / "report.json").read_text(encoding="utf-8")),
                    explain=json.loads((d / "explain.json").read_text(encoding="utf-8")),
                )
            except Exception:
                self._cache = None
                return None
            return self._cache


# ======================================================================
# services/datasets
# ======================================================================
DATASET_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class DatasetNotFound(KeyError):
    pass


class DatasetStore:
    def __init__(self, data_dir: Path) -> None:
        self.dir = Path(data_dir) / "uploads"

    def _paths(self, dataset_id: str) -> tuple[Path, Path]:
        if not DATASET_ID_RE.match(dataset_id):
            raise DatasetNotFound(dataset_id)
        return self.dir / f"{dataset_id}.csv", self.dir / f"{dataset_id}.meta.json"

    def save(
        self,
        df: pd.DataFrame,
        name: str,
        provenance: str,
        is_synthetic: bool,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        clean = sanitise(df)
        csv_bytes = clean.to_csv(index=False).encode("utf-8")
        sha = hashlib.sha256(csv_bytes).hexdigest()
        dataset_id = sha[:12]
        csv_path, meta_path = self._paths(dataset_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = csv_path.with_suffix(".csv.tmp")
        tmp.write_bytes(csv_bytes)
        os.replace(tmp, csv_path)
        meta = {
            "id": dataset_id,
            "name": name[:120],
            "provenance": provenance[:500],
            "is_synthetic": bool(is_synthetic),
            "sha256": sha,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": int(len(clean)),
            "targets_present": validation.get("targets_present", []),
            "trainable_targets": validation.get("trainable_targets", []),
            "n_warnings": len(validation.get("warnings", [])),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def list(self) -> list[dict[str, Any]]:
        if not self.dir.is_dir():
            return []
        metas = []
        for p in self.dir.glob("*.meta.json"):
            try:
                metas.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(metas, key=lambda m: m["uploaded_at"], reverse=True)

    def load(self, dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        csv_path, meta_path = self._paths(dataset_id)
        if not csv_path.exists() or not meta_path.exists():
            raise DatasetNotFound(dataset_id)
        return pd.read_csv(csv_path), json.loads(meta_path.read_text(encoding="utf-8"))


# ======================================================================
# services/jobs
# ======================================================================
class JobBusy(RuntimeError):
    pass


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def _running(self) -> bool:
        return any(j["status"] in ("queued", "running") for j in self._jobs.values())

    def start(self, kind: str, fn: Callable[[Callable[[float, str], None]], dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            if self._running():
                raise JobBusy("A job is already running. Wait for it to finish.")
            job_id = uuid.uuid4().hex[:12]
            job = {
                "id": job_id,
                "kind": kind,
                "status": "queued",
                "progress": 0.0,
                "message": "Queued",
                "result": None,
                "error": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
            }
            self._jobs[job_id] = job

        def progress(frac: float, msg: str) -> None:
            job["progress"] = round(float(frac), 3)
            job["message"] = msg

        def runner() -> None:
            job["status"] = "running"
            try:
                job["result"] = fn(progress)
                job["status"] = "succeeded"
                job["progress"] = 1.0
                job["message"] = "Finished"
            except Exception as exc:  # surfaced to the admin who started the job
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
                job["message"] = "Failed"
            finally:
                job["finished_at"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=runner, name=f"job-{job_id}", daemon=True).start()
        return dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return dict(job) if job else None

    def latest(self) -> dict[str, Any] | None:
        if not self._jobs:
            return None
        return dict(max(self._jobs.values(), key=lambda j: j["started_at"]))


# ======================================================================
# services/screening
# ======================================================================
DISCLAIMER = (
    "This is a preliminary screening estimate produced by a decision-support tool. It is not a medical diagnosis. "
    "False positives and false negatives are possible. A Low result does not guarantee good mental health, and a "
    "higher result does not mean you have a condition. Please speak with a qualified mental-health professional if "
    "you are worried about how you feel."
)
LIMITATIONS = [
    "Screening estimates can be wrong in both directions.",
    "The model reflects patterns in its training data and the definition of that data's labels.",
    "Decision paths describe associations in the training data, not causes.",
    "Do not use these results for admissions, employment, discipline or other consequential decisions.",
]
DEFAULT_CRISIS = {
  "region": "IN",
  "verification_notice": "Helpline details were entered from public information when this software was written and can change. The deploying institution MUST verify every number and replace/extend this list (set CRISIS_RESOURCES_FILE) before real use.",
  "emergency": [
    {
      "name": "National emergency number (India)",
      "contact": "112",
      "note": "If you or someone else is in immediate danger."
    }
  ],
  "helplines": [
    {
      "name": "Tele-MANAS (Government of India, 24x7 mental-health support)",
      "contact": "14416 or 1800-891-4416",
      "note": "Free; available in multiple languages."
    },
    {
      "name": "KIRAN Mental Health Rehabilitation Helpline (Government of India)",
      "contact": "1800-599-0019",
      "note": "Free, 24x7."
    }
  ],
  "directory": [
    {
      "name": "Find a Helpline (international directory)",
      "contact": "https://findahelpline.com",
      "note": "Lists free, confidential helplines by country."
    }
  ],
  "human_support": [
    "Your institution's student counselling or wellbeing service",
    "A trusted friend, family member, mentor or faculty member",
    "A qualified mental-health professional or your doctor"
  ]
}


@lru_cache
def load_crisis_resources(path: str | None = None) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else DEFAULT_CRISIS


def safety_assessment(answers: dict[str, int], resources: dict[str, Any]) -> dict[str, Any]:
    """Independent of the model: any non-zero answer to PHQ-9 item 9 triggers support guidance."""
    v = int(answers[SELF_HARM_ITEM_ID])
    triggered = v > 0
    level = "none" if not triggered else ("urgent" if v >= 2 else "support")
    out: dict[str, Any] = {
        "triggered": triggered,
        "level": level,
        "trigger": f"Your answer to the item about thoughts of being better off dead or of hurting yourself ({SELF_HARM_ITEM_ID})"
        if triggered
        else None,
        "message": None,
        "resources": None,
        "note": "This check uses only that one answer and does not depend on the model's result.",
    }
    if triggered:
        out["message"] = (
            "Thank you for answering honestly. You said you have had thoughts of being better off dead or of hurting "
            "yourself. You deserve support, and you do not have to handle this alone. Please reach out to someone "
            "today — a person you trust, your institution's counselling service, or one of the helplines below. "
            + (
                "If you feel you might act on these thoughts or are in immediate danger, call your local emergency "
                "number now or go to the nearest emergency department."
                if level == "urgent"
                else "If these thoughts become stronger or you feel unsafe, call your local emergency number or a helpline straight away."
            )
        )
        out["resources"] = resources
    return out


def reference_scores(answers: dict[str, int]) -> dict[str, Any]:
    t = totals(answers)
    return {
        "note": (
            "Standard questionnaire scoring, shown separately from the model. These bands describe symptom severity "
            "over the last two weeks as published for PHQ-9 and GAD-7; they are not diagnoses."
        ),
        "phq9": {"total": t["phq9_total"], "max": 27, "band": band_for(t["phq9_total"], PHQ9_BANDS)},
        "gad7": {"total": t["gad7_total"], "max": 21, "band": band_for(t["gad7_total"], GAD7_BANDS)},
        "supplementary": {
            "total": t["sup_total"],
            "max": 24,
            "band": None,
            "note": "Proposed items; no validated bands exist.",
        },
        "stress": {"available": False, "note": "No standard scoring rule for stress exists in the instruments used."},
    }


def highest_level(levels: list[str]) -> str | None:
    return max(levels, key=LEVELS.index) if levels else None


def recommendations(level: str | None, safety_triggered: bool) -> list[str]:
    recs: list[str] = []
    if safety_triggered:
        recs.append("Please contact one of the support options shown above today. Reaching out is a sign of strength.")
    if level == "High":
        recs += [
            "We recommend speaking with a qualified mental-health professional or your institution's counselling service soon.",
            "Tell someone you trust how you have been feeling so they can support you.",
        ]
    elif level == "Moderate":
        recs += [
            "Consider talking to your institution's counselling service, a doctor, or someone you trust about how you have been feeling.",
            "If these feelings continue or get worse over the next couple of weeks, please seek professional advice.",
        ]
    elif level == "Low":
        recs += [
            "Keep up routines that support your wellbeing, such as regular sleep, movement and time with people you trust.",
            "If your mood or stress changes, or you are worried at any time, you can reach out for support whatever this result says.",
        ]
    else:
        recs.append("If you are worried about how you feel, talk to a counsellor, doctor or someone you trust.")
    return recs


def _row_vector(pipe, frame) -> np.ndarray:
    return np.asarray(pipe[:-1].transform(frame), dtype=float)


def predict_conditions(answers: dict[str, int], active: ActiveModel) -> dict[str, Any]:
    bundle = active.bundle
    frame = answers_to_frame(answers, bundle["include_derived_totals"])
    cols = bundle["feature_columns"]
    features_used = {c: float(frame.iloc[0][c]) for c in cols}
    conditions = []
    for target, tb in bundle["targets"].items():
        pipe, classes, counts = tb["pipeline"], tb["classes"], tb["node_counts"]
        clf = pipe.named_steps["clf"]
        xt = _row_vector(pipe, frame)
        predicted = classes[int(pipe.predict(frame)[0])]
        path = decision_path_steps(clf, xt, cols, classes, counts)
        if path["prediction"] != predicted:  # internal consistency guard
            raise RuntimeError(f"decision path disagrees with model prediction for {target}")
        seen: set[str] = set()
        factors = []
        for s in path["steps"]:
            if s["feature"] in seen:
                continue
            seen.add(s["feature"])
            info = FEATURE_INFO[s["feature"]]
            factors.append(
                {
                    "feature": s["feature"],
                    "label": info["short"],
                    "description": info["description"],
                    "value": s["value"],
                }
            )
        conditions.append(
            {
                "target": target,
                "display_name": TARGET_BY_ID[target]["display_name"],
                "risk_level": predicted,
                "meaning": LEVEL_MEANING[predicted],
                "classes": classes,
                "decision_path": path,
                "key_factors": factors,
            }
        )
    return {"features_used": features_used, "conditions": conditions}


def screen(
    answers: dict[str, int], active: ActiveModel | None, crisis_resources: dict[str, Any]
) -> dict[str, Any]:
    safety = safety_assessment(answers, crisis_resources)
    refs = reference_scores(answers)
    base: dict[str, Any] = {
        "safety": safety,
        "reference_scores": refs,
        "disclaimer": DISCLAIMER,
        "limitations": LIMITATIONS,
    }
    if active is None:
        return {
            **base,
            "status": "model_unavailable",
            "model": {
                "available": False,
                "run_id": None,
                "is_synthetic": None,
                "reason": "No trained model is active. An administrator must train and activate a model from a dataset first.",
            },
            "conditions": [],
            "summary": None,
            "features_used": None,
            "recommendations": recommendations(None, safety["triggered"]),
        }
    pred = predict_conditions(answers, active)
    levels = [c["risk_level"] for c in pred["conditions"]]
    top = highest_level(levels)
    summary = None
    if top:
        summary = {
            "highest_level": top,
            "derived": True,
            "note": (
                "This is simply the highest of the condition-specific levels above. It is not a separate model output "
                "and not a diagnosis."
            ),
        }
    return {
        **base,
        "status": "ok",
        "model": {
            "available": True,
            "run_id": active.run_id,
            "is_synthetic": bool(active.bundle.get("is_synthetic")),
            "reason": None,
        },
        "conditions": pred["conditions"],
        "summary": summary,
        "features_used": pred["features_used"],
        "recommendations": recommendations(top, safety["triggered"]),
    }


# ======================================================================
# schemas/screening
# ======================================================================
_fields: dict[str, Any] = {
    item_id: (
        int,
        Field(..., ge=ITEM_MIN, le=ITEM_MAX, strict=True, description=ITEM_BY_ID[item_id]["text"]),
    )
    for item_id in ITEM_IDS
}

ScreeningAnswers = create_model(  # type: ignore[call-overload]
    "ScreeningAnswers",
    __config__=ConfigDict(extra="forbid"),
    **_fields,
)


class ScreeningRequest(BaseModel):
    """All 24 items are required integers 0-3. Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")
    answers: ScreeningAnswers  # type: ignore[valid-type]


class ModelInfo(BaseModel):
    available: bool
    run_id: str | None
    is_synthetic: bool | None
    reason: str | None


class ScreeningResponse(BaseModel):
    status: str = Field(description="'ok' or 'model_unavailable'")
    model: ModelInfo
    safety: dict[str, Any]
    reference_scores: dict[str, Any]
    conditions: list[dict[str, Any]]
    summary: dict[str, Any] | None
    features_used: dict[str, float] | None
    recommendations: list[str]
    limitations: list[str]
    disclaimer: str


# ======================================================================
# schemas/admin
# ======================================================================
class TrainRequest(BaseModel):
    dataset_id: str = Field(pattern=r"^[0-9a-f]{12}$")
    selection_metric: str = Field(default="f1_macro")
    include_derived_totals: bool = True
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    cv_folds: int = Field(default=5, ge=2, le=10)
    models: list[str] = Field(default_factory=lambda: list(MODEL_ORDER))
    activate: bool | None = Field(
        default=None,
        description="Activate the new run for predictions. Default: only if no model is active yet.",
    )


class SyntheticRequest(BaseModel):
    n_rows: int = Field(default=1500, ge=300, le=20000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    include_overall: bool = False


__all__ = ["TrainRequest", "SyntheticRequest", "SELECTION_METRICS"]


# ======================================================================
# core/project_info
# ======================================================================
PROJECT_TITLE = (
    "An Interpretable, Multi-Condition Decision Tree Model for Early Detection and Risk Stratification of "
    "Depression, Anxiety, and Stress"
)

REPORT_SOURCE_STATUS = (
    "The Phase-I PDF was not available in the build environment. The objectives, scope and gaps below follow the "
    "summary in the project brief and should be checked against the report. The report's own reference list has not "
    "been copied here; add it before submission."
)

PROJECT_INFO = {
    "title": PROJECT_TITLE,
    "short_title": "Interpretable Decision-Tree Mental-Health Screening",
    "report_source_status": REPORT_SOURCE_STATUS,
    "research_problem": (
        "Early signs of depression, anxiety and stress in students are often missed. Accurate black-box models are "
        "hard to explain to students and counsellors, and screening data are typically imbalanced, so a model that is "
        "accurate overall can still miss the students who most need support. This project builds a pruned CART "
        "decision tree whose every prediction can be traced as a sequence of if-then conditions, and compares it "
        "fairly with less interpretable models."
    ),
    "objectives": [
        {
            "id": "O1",
            "text": "Build an interpretable, pruned CART decision-tree classifier.",
            "implemented_by": "backend/app/ml/training.py (cost-complexity pruning + depth/leaf constraints tuned by cross-validation)",
        },
        {
            "id": "O2",
            "text": "Screen depression, anxiety and stress together using multiple indicators (about 24 self-report items from PHQ-9, GAD-7 and supplementary behavioural indicators).",
            "implemented_by": "backend/app/core/questionnaire.py; one tree per condition target present in the dataset",
        },
        {
            "id": "O3",
            "text": "Include sleep quality, academic stress and behavioural-change indicators.",
            "implemented_by": "Supplementary section (proposed items; configurable; not validated)",
        },
        {
            "id": "O4",
            "text": "Stratify risk as Low, Moderate or High.",
            "implemented_by": "Targets use the dataset's Low/Moderate/High labels; definitions documented in docs/dataset-schema.md",
        },
        {
            "id": "O5",
            "text": "Give transparent if-then decision paths for individual predictions.",
            "implemented_by": "backend/app/ml/explain.py via clf.decision_path; Results and Explainable AI pages",
        },
        {
            "id": "O6",
            "text": "Handle class imbalance and emphasise recall so at-risk students are less likely to be missed.",
            "implemented_by": "Balanced class weights tuned in CV; macro-recall, per-class recall and balanced accuracy reported",
        },
        {
            "id": "O7",
            "text": "Benchmark fairly against Random Forest, XGBoost, LightGBM and Logistic Regression.",
            "implemented_by": "Same split, same CV protocol and same inputs for all models; skipped models reported with a reason",
        },
        {
            "id": "O8",
            "text": "Evaluate with accuracy, precision, recall, F1 and ROC-AUC.",
            "implemented_by": "Evaluation report JSON/CSV and Model Performance page",
        },
        {
            "id": "O9",
            "text": "Be deployable in educational institutions without expensive infrastructure.",
            "implemented_by": "Single-process FastAPI + static React build; no GPU, no external services, no database required",
        },
        {
            "id": "O10",
            "text": "Protect privacy and state clearly that this is a screening/decision-support tool, not a diagnosis.",
            "implemented_by": "No answer storage or logging, no identifiers collected, disclaimers on every result, tested safety flow",
        },
    ],
    "scope": [
        "Preliminary screening and decision support for students in educational institutions.",
        "Not a diagnostic tool; not for admissions, employment, discipline or other consequential decisions.",
        "Predictions are limited to the conditions for which the training dataset provides labels.",
    ],
    "methodology": [
        "Dataset inspection and schema validation (ranges, missingness, duplicates, class balance, leakage heuristics).",
        "Held-out stratified test split; all tuning with cross-validation on the training split only.",
        "Median imputation fitted on training data only; deterministic derived totals; no resampling of the test set.",
        "Pruned CART tuned over depth, leaf size, class weights, criterion and cost-complexity alpha.",
        "Benchmarks: Random Forest, XGBoost, LightGBM, Logistic Regression under the same protocol.",
        "Explanations taken from the fitted tree (structure, rules, decision path) and held-out permutation importance.",
    ],
    "research_gaps": [
        "Interpretability of high-accuracy models for student mental-health screening.",
        "Joint screening of depression, anxiety and stress rather than a single condition.",
        "Explicit handling of class imbalance with attention to recall.",
        "Fair accuracy-versus-interpretability comparison on the same data and protocol.",
    ],
    "technology_stack": {
        "frontend": ["React", "Vite", "TypeScript", "Tailwind CSS", "Recharts", "React Router", "React Hook Form", "Zod", "Lucide"],
        "backend": ["Python", "FastAPI", "Pydantic", "scikit-learn", "pandas", "NumPy", "joblib", "XGBoost", "LightGBM"],
        "testing": ["pytest", "Vitest", "Testing Library"],
    },
    "limitations": [
        "Until the supervisor-provided dataset is integrated, any model in this repository is trained on SYNTHETIC data and its metrics have no scientific meaning.",
        "The supplementary questionnaire items are proposed, unvalidated and not supervisor-approved.",
        "Labels are only as good as their definition in the training data; screening estimates can be wrong in both directions (false positives and false negatives).",
        "A single decision tree is a coarse model; leaf class shares are descriptive training counts, not calibrated probabilities.",
        "Explanations show predictive association in the training data, not causes.",
        "Not clinically validated and not approved for clinical use.",
    ],
    "ethics": [
        "No names, contact details or student IDs are requested; answers are processed in memory and not stored or logged.",
        "Results are screening estimates, not a diagnosis. A Low result does not guarantee good mental health.",
        "A safety check based on the PHQ-9 self-harm item runs independently of the model and always shows human-support information.",
        "Do not use results for admissions, employment, discipline or other consequential decisions about students.",
        "Deploying institutions must verify helplines, obtain ethical approval and arrange counselling follow-up before real-world use.",
    ],
    "references": [],
    "references_status": "Pending: copy the reference list from the Phase-I report here (docs/methodology.md).",
    "implementation_references": [
        "Kroenke K, Spitzer RL, Williams JBW. The PHQ-9: validity of a brief depression severity measure. J Gen Intern Med. 2001;16(9):606-613.",
        "Spitzer RL, Kroenke K, Williams JBW, Löwe B. A brief measure for assessing generalized anxiety disorder: the GAD-7. Arch Intern Med. 2006;166(10):1092-1097.",
        "Breiman L, Friedman JH, Olshen RA, Stone CJ. Classification and Regression Trees. Wadsworth; 1984.",
    ],
}


# ======================================================================
# api/public
# ======================================================================
public_router = APIRouter(prefix="/api")
public_limit = Depends(rate_limit("public", "public_rate_limit_per_min"))
predict_limit = Depends(rate_limit("predict", "predict_rate_limit_per_min"))


def get_registry(request: Request) -> Registry:
    return request.app.state.registry


@public_router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@public_router.get("/project-info", tags=["project"], dependencies=[public_limit])
def project_info() -> dict[str, Any]:
    return PROJECT_INFO


@public_router.get("/questionnaire", tags=["screening"], dependencies=[public_limit])
def questionnaire() -> dict[str, Any]:
    return public_questionnaire()


@public_router.get("/model/status", tags=["model"], dependencies=[public_limit])
def model_status(registry: Registry = Depends(get_registry)) -> dict[str, Any]:
    active = registry.load_active()
    if active is None:
        return {
            "trained": False,
            "message": "No trained model is active. Evaluation results and predictions are pending a dataset and an administrator-run training.",
            "targets": [],
        }
    r = active.report
    return {
        "trained": True,
        "run_id": active.run_id,
        "trained_at": r["created_at"],
        "is_synthetic": bool(r["dataset"].get("is_synthetic")),
        "dataset": {k: r["dataset"].get(k) for k in ("id", "name", "n_rows", "is_synthetic", "conditions_covered")},
        "targets": [
            {"id": t, "display_name": tr["display_name"], "classes": tr["classes"]} for t, tr in r["targets"].items()
        ],
        "skipped_targets": r.get("skipped_targets", {}),
        "message": (
            "SYNTHETIC DATA: this model exists only to demonstrate the software."
            if r["dataset"].get("is_synthetic")
            else "Model trained on the supplied dataset. Not clinically validated."
        ),
    }


@public_router.get("/model/features", tags=["model"], dependencies=[public_limit])
def model_features(registry: Registry = Depends(get_registry)) -> dict[str, Any]:
    active = registry.load_active()
    derived_in = active.bundle["include_derived_totals"] if active else True
    cols = active.bundle["feature_columns"] if active else feature_columns(True)
    return {
        "trained": active is not None,
        "run_id": active.run_id if active else None,
        "derived_totals_included": derived_in,
        "item_ids": ITEM_IDS,
        "derived": [d for d in DERIVED_FEATURES],
        "columns": cols,
        "feature_info": {c: FEATURE_INFO[c] for c in cols},
        "targets": [{"id": t["id"], "display_name": t["display_name"], "definition": t["definition"]} for t in TARGETS],
        "levels": LEVELS,
    }


@public_router.get("/model/metrics", tags=["model"], dependencies=[public_limit])
def model_metrics(registry: Registry = Depends(get_registry)) -> dict[str, Any]:
    active = registry.load_active()
    if active is None:
        return {
            "available": False,
            "message": "Evaluation has not been run. Metrics appear here after an administrator trains models on a dataset.",
        }
    return {"available": True, "report": active.report}


@public_router.get("/model/tree", tags=["model"], dependencies=[public_limit])
def model_tree(
    target: str | None = Query(default=None, description="Target id, e.g. depression_risk"),
    registry: Registry = Depends(get_registry),
) -> dict[str, Any]:
    active = registry.load_active()
    if active is None:
        raise HTTPException(503, "No trained model is active; there is no tree to show.")
    targets = active.explain["targets"]
    chosen = target or next(iter(targets))
    if chosen not in targets:
        raise HTTPException(404, f"Unknown target '{chosen}'. Available: {list(targets)}")
    return {
        "run_id": active.run_id,
        "is_synthetic": bool(active.explain.get("is_synthetic")),
        "target": chosen,
        "available_targets": [{"id": t, "display_name": v["display_name"]} for t, v in targets.items()],
        "feature_info": FEATURE_INFO,
        **targets[chosen],
    }


@public_router.post("/screening/predict", response_model=ScreeningResponse, tags=["screening"], dependencies=[predict_limit])
def predict(
    body: ScreeningRequest,
    request: Request,
    registry: Registry = Depends(get_registry),
    settings: Settings = Depends(app_settings),
) -> dict[str, Any]:
    """Return the actual fitted-model output with its decision path.

    Answers are processed in memory only: not stored and not logged. If no model has
    been trained the response has ``status='model_unavailable'`` but still carries the
    model-independent safety guidance and reference scores.
    """
    answers = body.answers.model_dump()
    resources = load_crisis_resources(str(settings.crisis_resources_file) if settings.crisis_resources_file else None)
    return screen(answers, registry.load_active(), resources)


# ======================================================================
# api/admin
# ======================================================================
admin_router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def registry_dep(request: Request) -> Registry:
    return request.app.state.registry


def datasets_dep(request: Request) -> DatasetStore:
    return request.app.state.datasets


def jobs_dep(request: Request) -> JobManager:
    return request.app.state.jobs


async def _read_upload(file: UploadFile, settings: Settings) -> bytes:
    name = (file.filename or "").lower()
    if not name.endswith(".csv"):
        raise HTTPException(415, "Only .csv files are accepted.")
    limit = settings.max_upload_mb * 1024 * 1024
    raw = await file.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(413, f"File exceeds the {settings.max_upload_mb} MB limit.")
    if not raw:
        raise HTTPException(400, "File is empty.")
    return raw


@admin_router.post("/dataset/validate")
async def validate_dataset(file: UploadFile = File(...), settings: Settings = Depends(app_settings)) -> dict[str, Any]:
    """Inspect an uploaded CSV without storing it."""
    raw = await _read_upload(file, settings)
    try:
        df = read_csv_bytes(raw, settings.max_upload_rows)
    except DatasetError as exc:
        raise HTTPException(422, str(exc)) from exc
    return validate_dataframe(df)


@admin_router.post("/dataset/upload", status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(..., min_length=1, max_length=120),
    provenance: str = Form(..., min_length=3, max_length=500),
    is_synthetic: bool = Form(False),
    settings: Settings = Depends(app_settings),
    store: DatasetStore = Depends(datasets_dep),
) -> dict[str, Any]:
    """Validate and store a dataset. Only schema columns are persisted; other columns are dropped."""
    raw = await _read_upload(file, settings)
    try:
        df = read_csv_bytes(raw, settings.max_upload_rows)
    except DatasetError as exc:
        raise HTTPException(422, str(exc)) from exc
    report = validate_dataframe(df)
    if not report["ok"]:
        raise HTTPException(422, {"message": "Dataset failed validation.", "validation": report})
    synthetic = bool(is_synthetic or report.get("is_synthetic"))
    meta = store.save(df, name, provenance, synthetic, report)
    return {"dataset": meta, "validation": report}


@admin_router.post("/dataset/synthetic", status_code=201)
def create_synthetic(
    body: SyntheticRequest, store: DatasetStore = Depends(datasets_dep)
) -> dict[str, Any]:
    """Generate and register a SYNTHETIC demonstration dataset (software testing only)."""
    df = sanitise(generate_synthetic_dataset(body.n_rows, body.seed, body.include_overall))
    report = validate_dataframe(df)
    meta = store.save(
        df,
        f"SYNTHETIC demo (n={body.n_rows}, seed={body.seed})",
        "Generated by app.ml.synthetic for software testing. Not real data.",
        True,
        report,
    )
    return {"dataset": meta, "validation": report}


@admin_router.get("/datasets")
def list_datasets(store: DatasetStore = Depends(datasets_dep)) -> list[dict[str, Any]]:
    return store.list()


@admin_router.post("/model/train", status_code=202)
def train(
    body: TrainRequest,
    settings: Settings = Depends(app_settings),
    store: DatasetStore = Depends(datasets_dep),
    registry: Registry = Depends(registry_dep),
    jobs: JobManager = Depends(jobs_dep),
) -> dict[str, Any]:
    """Start training in the background. Poll GET /api/admin/jobs/{id}. Previous runs are never overwritten."""
    try:
        df, meta = store.load(body.dataset_id)
    except DatasetNotFound:
        raise HTTPException(404, "Dataset not found.")
    opts = TrainOptions(
        test_size=body.test_size,
        seed=body.seed,
        selection_metric=body.selection_metric,
        include_derived_totals=body.include_derived_totals,
        n_jobs=settings.train_n_jobs,
        cv_folds=body.cv_folds,
        models=body.models,
    )
    try:
        opts.validate()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    def work(progress) -> dict[str, Any]:
        result = train_and_evaluate(df, meta, opts, progress)
        saved = registry.save_run(result, body.activate)
        return saved

    try:
        return jobs.start("train", work)
    except JobBusy as exc:
        raise HTTPException(409, str(exc)) from exc


@admin_router.get("/jobs/{job_id}")
def job_status(job_id: str, jobs: JobManager = Depends(jobs_dep)) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job


@admin_router.get("/jobs")
def latest_job(jobs: JobManager = Depends(jobs_dep)) -> dict[str, Any] | None:
    return jobs.latest()


@admin_router.get("/evaluations")
def list_evaluations(registry: Registry = Depends(registry_dep)) -> list[dict[str, Any]]:
    return registry.list_runs()


@admin_router.get("/evaluations/{run_id}")
def get_evaluation(run_id: str, registry: Registry = Depends(registry_dep)) -> dict[str, Any]:
    try:
        return registry.get_report(run_id)
    except RunNotFound:
        raise HTTPException(404, "Run not found.")


@admin_router.get("/evaluations/{run_id}/export")
def export_evaluation(run_id: str, format: str = "json", registry: Registry = Depends(registry_dep)) -> Response:
    if format not in ("json", "csv"):
        raise HTTPException(422, "format must be 'json' or 'csv'")
    try:
        report = registry.get_report(run_id)
    except RunNotFound:
        raise HTTPException(404, "Run not found.")
    if format == "json":
        body, media = json.dumps(report, indent=2), "application/json"
    else:
        body, media = report_to_csv(report), "text/csv"
    return Response(
        body,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="evaluation-{run_id}.{format}"'},
    )


@admin_router.post("/model/activate/{run_id}")
def activate(run_id: str, registry: Registry = Depends(registry_dep)) -> dict[str, str]:
    try:
        registry.activate(run_id)
    except RunNotFound:
        raise HTTPException(404, "Run not found.")
    return {"active_run_id": run_id}


# ======================================================================
# main
# ======================================================================
log = logging.getLogger("app.access")

DESCRIPTION = (
    "Screening and decision-support API for an interpretable CART decision-tree model of depression, anxiety and "
    "stress risk. **Not a diagnostic tool.** Questionnaire answers are processed in memory and are never stored or logged."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.admin_enabled:
            logging.getLogger("app").warning(
                "ADMIN_API_KEY missing or shorter than 16 characters: /api/admin/* is disabled."
            )
        yield

    app = FastAPI(
        title="Interpretable Decision Tree Mental-Health Screening API",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = Registry(settings.artifacts_dir)
    app.state.datasets = DatasetStore(settings.data_dir)
    app.state.jobs = JobManager()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600,
    )

    @app.middleware("http")
    async def guard_and_log(request: Request, call_next):
        # Reject oversized JSON bodies early (uploads have their own limit).
        if request.method == "POST" and not request.url.path.startswith("/api/admin/dataset/"):
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > settings.max_json_body_kb * 1024:
                return JSONResponse({"detail": "Request body too large."}, status_code=413)
        start = time.perf_counter()
        response = await call_next(request)
        # Privacy: log method, path and status only. Never query strings, bodies or answers.
        log.info(
            "%s %s -> %s (%.0f ms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    app.include_router(public_router)
    app.include_router(admin_router)
    return app


app = create_app()


# ======================================================================
# CLI
# ======================================================================
def _cli() -> None:
    import argparse, sys

    ap = argparse.ArgumentParser(description="TreeScreen CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train + evaluate all models and register the run")
    src = t.add_mutually_exclusive_group(required=True)
    src.add_argument("--data"); src.add_argument("--synthetic", action="store_true")
    t.add_argument("--name"); t.add_argument("--provenance")
    t.add_argument("--n-rows", type=int, default=1500); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--selection-metric", choices=SELECTION_METRICS, default="f1_macro")
    t.add_argument("--no-derived-totals", action="store_true")
    t.add_argument("--cv-folds", type=int, default=5); t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--models", nargs="+", default=list(MODEL_ORDER), choices=MODEL_ORDER)
    t.add_argument("--n-jobs", type=int, default=-1); t.add_argument("--activate", action="store_true")
    v = sub.add_parser("validate", help="validate a dataset CSV"); v.add_argument("csv"); v.add_argument("--json", action="store_true")
    e = sub.add_parser("evaluate", help="print/export a saved run")
    e.add_argument("--run-id"); e.add_argument("--format", choices=["table", "csv", "json"], default="table"); e.add_argument("--out")
    a = ap.parse_args()
    settings = get_settings()

    if a.cmd == "validate":
        try:
            df = read_csv_bytes(open(a.csv, "rb").read(), settings.max_upload_rows)
        except (OSError, DatasetError) as exc:
            sys.exit(f"Cannot read dataset: {exc}")
        r = validate_dataframe(df)
        if a.json:
            print(json.dumps(r, indent=2))
        else:
            print(f"rows={r['n_rows']} ok={r['ok']} trainable={r.get('trainable_targets')} blocked={r.get('blocked_targets')}")
            for x in r["errors"]: print("ERROR  ", x)
            for x in r["warnings"]: print("WARNING", x)
        sys.exit(0 if r["ok"] else 1)

    if a.cmd == "train":
        if a.synthetic:
            df = generate_synthetic_dataset(a.n_rows, a.seed)
            name = a.name or f"SYNTHETIC demo (n={a.n_rows}, seed={a.seed})"
            prov = a.provenance or "Generated by this program for software testing. Not real data."
            synth = True
        else:
            try:
                df = read_csv_bytes(open(a.data, "rb").read(), settings.max_upload_rows)
            except (OSError, DatasetError) as exc:
                sys.exit(f"Cannot read dataset: {exc}")
            if not (a.name and a.provenance):
                sys.exit("--name and --provenance are required for real datasets.")
            name, prov, synth = a.name, a.provenance, False
        rep = validate_dataframe(df)
        if not rep["ok"]:
            sys.exit("Dataset invalid:\n  " + "\n  ".join(rep["errors"]))
        for w in rep["warnings"]: print("WARNING:", w)
        meta = DatasetStore(settings.data_dir).save(df, name, prov, synth or bool(rep.get("is_synthetic")), rep)
        clean, meta = DatasetStore(settings.data_dir).load(meta["id"])
        opts = TrainOptions(test_size=a.test_size, seed=a.seed, selection_metric=a.selection_metric,
                            include_derived_totals=not a.no_derived_totals, n_jobs=a.n_jobs, cv_folds=a.cv_folds, models=a.models)
        res = train_and_evaluate(clean, meta, opts, progress=lambda f, m: print(f"[{f:5.0%}] {m}", flush=True))
        saved = Registry(settings.artifacts_dir).save_run(res, activate=True if a.activate else None)
        print(f"run {saved['run_id']} saved (active={saved['activated']}) in {res.report['duration_seconds']}s")
        return

    reg = Registry(settings.artifacts_dir)
    run_id = a.run_id or reg.active_run_id()
    if not run_id: sys.exit("No run found. Train first.")
    try:
        rep = reg.get_report(run_id)
    except RunNotFound:
        sys.exit(f"Run {run_id} not found.")
    if a.format == "json": text = json.dumps(rep, indent=2)
    elif a.format == "csv": text = report_to_csv(rep)
    else:
        L = ["*** SYNTHETIC DATA: metrics demonstrate software behaviour only ***" if rep["dataset"]["is_synthetic"] else ""]
        L.append(f"run {rep['run_id']} test n={rep['split']['n_test']} selection={rep['tuning']['selection_metric']}")
        for tg, tr in rep["targets"].items():
            L.append(f"\n{tr['display_name']}\n  {'model':32s} {'acc':>6s} {'bal':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'auc':>6s}")
            for m in tr["models"].values():
                if m["status"] != "ok": L.append(f"  {m['display_name']:32s} {m['status']}: {m.get('reason', '')}"); continue
                x = m["metrics"]; auc = "   n/a" if x["roc_auc_ovr_macro"] is None else f"{x['roc_auc_ovr_macro']:6.3f}"
                L.append(f"  {m['display_name']:32s} {x['accuracy']:6.3f} {x['balanced_accuracy']:6.3f} {x['precision_macro']:6.3f} {x['recall_macro']:6.3f} {x['f1_macro']:6.3f} {auc}")
        text = "\n".join(L)
    if a.out: open(a.out, "w", encoding="utf-8").write(text); print("written", a.out)
    else: print(text)


if __name__ == "__main__":
    _cli()
