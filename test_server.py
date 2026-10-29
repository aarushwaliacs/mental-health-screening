"""Backend test suite (pytest). Run: pytest -q   (about 3 minutes: it trains real models on SYNTHETIC data)."""
from __future__ import annotations
import random
import pytest
from fastapi.testclient import TestClient
from server import Settings
from server import ITEM_IDS
from server import limiter
from server import create_app
from server import sanitise
from server import generate_synthetic_dataset
from server import TrainOptions, train_and_evaluate
import pandas as pd
from server import DERIVED_FEATURES, ITEM_IDS, SECTIONS
from server import build_feature_frame, feature_columns, totals
import io
import numpy as np
from server import DatasetError, read_csv_bytes, sanitise, validate_dataframe
import json
import server as tr
from server import normalise_labels
from server import build_feature_frame
from server import answers_to_frame
import logging
import time


# --- conftest ---
ADMIN_KEY = "test-admin-key-0123456789abcdef"


@pytest.fixture(autouse=True)
def _reset_limiter():
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture(scope="session")
def synthetic_df():
    return sanitise(generate_synthetic_dataset(600, seed=11))


@pytest.fixture(scope="session")
def trained(synthetic_df):
    """One real training run (all 5 models, 3 targets) shared by tests. SYNTHETIC data."""
    meta = {"id": "abcdef012345", "name": "test synthetic", "provenance": "unit test", "is_synthetic": True}
    opts = TrainOptions(cv_folds=3, seed=3, n_jobs=-1)
    return train_and_evaluate(synthetic_df, meta, opts)


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        admin_api_key=ADMIN_KEY,
        artifacts_dir=tmp_path / "artifacts",
        data_dir=tmp_path / "data",
        cors_origins="http://localhost:5173",
        _env_file=None,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def client(tmp_path):
    """App with NO trained model (its own empty directory, so it can coexist with model_client)."""
    return TestClient(create_app(make_settings(tmp_path / "empty")))


@pytest.fixture
def model_client(tmp_path, trained):
    """App with the shared trained (synthetic) model active."""
    app = create_app(make_settings(tmp_path))
    app.state.registry.save_run(trained, activate=True)
    return TestClient(app)


@pytest.fixture
def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def random_answers(rng: random.Random, item9: int | None = None) -> dict[str, int]:
    a = {i: min(3, max(0, int(rng.random() ** 2 * 4))) for i in ITEM_IDS}
    if item9 is not None:
        a["phq9_9"] = item9
    return a


# --- test_basics ---
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_security_headers_and_no_store(client):
    r = client.get("/api/health")
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_questionnaire_shape(client):
    q = client.get("/api/questionnaire").json()
    assert q["item_count"] == 24 == len(ITEM_IDS) == len(set(ITEM_IDS))
    sizes = {s["id"]: len(s["items"]) for s in q["sections"]}
    assert sizes == {"phq9": 9, "gad7": 7, "supplementary": 8}
    std = {s["id"]: s["standardized"] for s in q["sections"]}
    assert std == {"phq9": True, "gad7": True, "supplementary": False}
    assert [o["value"] for o in q["scales"][0]["options"]] == [0, 1, 2, 3]
    crit = [i["id"] for s in q["sections"] for i in s["items"] if i["safety_critical"]]
    assert crit == ["phq9_9"]


def test_project_info_declares_pending_references(client):
    info = client.get("/api/project-info").json()
    assert info["references"] == [] and "Pending" in info["references_status"]
    assert "not available" in info["report_source_status"].lower()


def test_derived_totals_are_exact_sums():
    ans = random_answers(random.Random(1))
    t = totals(ans)
    for d in DERIVED_FEATURES:
        assert t[d["id"]] == sum(ans[i] for i in d["items"])
    assert t["phq9_total"] <= 27 and t["gad7_total"] <= 21 and t["sup_total"] <= 24


def test_feature_frame_columns_and_missing_totals():
    df = pd.DataFrame([{i: 1 for i in ITEM_IDS}, {**{i: 1 for i in ITEM_IDS}, "phq9_1": None}])
    f = build_feature_frame(df, True)
    assert list(f.columns) == feature_columns(True)
    assert f.loc[0, "phq9_total"] == 9 and pd.isna(f.loc[1, "phq9_total"])  # NaN, not a silent partial sum
    assert list(build_feature_frame(df, False).columns) == ITEM_IDS


# --- test_dataset_validation ---
@pytest.fixture(scope="module")
def df():
    return generate_synthetic_dataset(500, seed=5)


def test_valid_synthetic_dataset(df):
    r = validate_dataframe(df)
    assert r["ok"] and set(r["trainable_targets"]) == {"depression_risk", "anxiety_risk", "stress_risk"}
    assert r["is_synthetic"] is True
    assert sum(r["class_distribution"]["depression_risk"].values()) == 500


def test_missing_item_column_is_error(df):
    r = validate_dataframe(df.drop(columns=["gad7_3"]))
    assert not r["ok"] and any("gad7_3" in e for e in r["errors"])


def test_no_target_is_error(df):
    r = validate_dataframe(df.drop(columns=["depression_risk", "anxiety_risk", "stress_risk"]))
    assert not r["ok"] and any("No target" in e for e in r["errors"])


def test_out_of_range_and_non_numeric_items(df):
    bad = df.copy()
    bad.loc[0, "phq9_1"] = 7
    assert any("outside" in e for e in validate_dataframe(bad)["errors"])
    bad = df.copy().astype({"phq9_2": object})
    bad.loc[0, "phq9_2"] = "often"
    assert any("Non-numeric" in e for e in validate_dataframe(bad)["errors"])


def test_invalid_label_values(df):
    bad = df.copy()
    bad.loc[0, "stress_risk"] = "Severe"
    r = validate_dataframe(bad)
    assert not r["ok"] or "stress_risk" in r["blocked_targets"]
    assert any("stress_risk" in e for e in r["errors"])


def test_single_class_target_blocked(df):
    bad = df.copy()
    bad["anxiety_risk"] = "Low"
    r = validate_dataframe(bad)
    assert "anxiety_risk" in r["blocked_targets"] and "anxiety_risk" not in r["trainable_targets"]
    assert r["ok"]  # other targets still trainable


def test_missing_condition_is_reported_not_faked(df):
    r = validate_dataframe(df.drop(columns=["stress_risk"]))
    assert r["ok"] and "stress" not in r["conditions_covered"]
    assert any("will not predict" in w for w in r["warnings"])


def test_label_leakage_detected_for_derived_total(df):
    leaky = df.copy()
    phq = leaky[[c for c in leaky.columns if c.startswith("phq9_")]].sum(axis=1)
    leaky["depression_risk"] = np.where(phq >= 10, "High", np.where(phq >= 5, "Moderate", "Low"))
    r = validate_dataframe(leaky)
    assert any("depression_risk" in w and "phq9_total" in w and "include_derived_totals=false" in w for w in r["leakage_warnings"])


def test_pii_columns_flagged_and_dropped(df):
    d = df.copy()
    d["student_name"] = "x"
    d["email"] = "a@b.c"
    r = validate_dataframe(d)
    assert any("personal identifiers" in w for w in r["warnings"])
    clean = sanitise(d)
    assert "student_name" not in clean.columns and "email" not in clean.columns


def test_missing_values_reported(df):
    d = df.copy()
    d.loc[:20, "gad7_1"] = np.nan
    r = validate_dataframe(d)
    assert r["missing_values"]["gad7_1"] == 21 and r["ok"]


def test_read_csv_rejects_bad_input():
    with pytest.raises(DatasetError):
        read_csv_bytes(b"\xff\xfe\x00bad", 100)
    with pytest.raises(DatasetError):
        read_csv_bytes(b"a,b\n" + b"1,2\n" * 10, 5)


def test_upload_endpoint_limits_and_auth(tmp_path, df):
    from fastapi.testclient import TestClient
    from server import create_app

    app = create_app(make_settings(tmp_path, max_upload_mb=1))
    c = TestClient(app)
    h = {"Authorization": f"Bearer {ADMIN_KEY}"}
    csv = df.to_csv(index=False).encode()
    files = lambda name, data: {"file": (name, io.BytesIO(data), "text/csv")}
    # wrong extension
    assert c.post("/api/admin/dataset/validate", files=files("x.exe", csv), headers=h).status_code == 415
    # too large
    assert c.post("/api/admin/dataset/validate", files=files("x.csv", b"a," * 2_000_000), headers=h).status_code == 413
    # empty
    assert c.post("/api/admin/dataset/validate", files=files("x.csv", b""), headers=h).status_code == 400
    # ok
    assert c.post("/api/admin/dataset/validate", files=files("x.csv", csv), headers=h).json()["ok"] is True
    # invalid dataset not stored
    bad = df.drop(columns=["phq9_1"]).to_csv(index=False).encode()
    r = c.post(
        "/api/admin/dataset/upload",
        files=files("x.csv", bad),
        data={"name": "bad", "provenance": "test"},
        headers=h,
    )
    assert r.status_code == 422 and c.get("/api/admin/datasets", headers=h).json() == []
    # valid dataset stored, extra columns dropped from disk
    d = df.copy()
    d["email"] = "a@b.c"
    r = c.post(
        "/api/admin/dataset/upload",
        files=files("x.csv", d.to_csv(index=False).encode()),
        data={"name": "good", "provenance": "unit test data"},
        headers=h,
    )
    assert r.status_code == 201
    stored = pd.read_csv(next((tmp_path / "data" / "uploads").glob("*.csv")))
    assert "email" not in stored.columns


# --- test_training ---
def test_report_structure_and_json_serialisable(trained):
    r = trained.report
    json.dumps(r)  # no numpy leftovers
    assert r["dataset"]["is_synthetic"] is True
    assert any("SYNTHETIC" in w for w in r["warnings"])
    assert set(r["targets"]) == {"depression_risk", "anxiety_risk", "stress_risk"}
    for k in ("python", "scikit-learn", "numpy", "pandas", "xgboost", "lightgbm"):
        assert k in r["software_versions"]
    assert r["split"]["n_train"] + r["split"]["n_test"] == r["dataset"]["n_rows"]
    assert r["tuning"]["selection_metric"] == "f1_macro"
    assert "metric_definitions" in r and r["feature_schema"]["columns"]


def test_all_five_models_evaluated_or_skipped_with_reason(trained):
    for t, tr_ in trained.report["targets"].items():
        assert set(tr_["models"]) == set(tr.MODEL_ORDER)
        for key, m in tr_["models"].items():
            assert m["status"] in ("ok", "skipped", "failed")
            if m["status"] != "ok":
                assert m["reason"]
        assert tr_["models"]["cart"]["status"] == "ok"
        assert tr_["models"]["logistic_regression"]["status"] == "ok"


def test_metric_ranges_and_confusion_matrix_consistency(trained):
    for t, tr_ in trained.report["targets"].items():
        n_test = trained.report["split"]["n_test"]
        assert sum(tr_["class_distribution"]["test"].values()) == n_test
        assert sum(tr_["class_distribution"]["train"].values()) == trained.report["split"]["n_train"]
        for m in tr_["models"].values():
            if m["status"] != "ok":
                continue
            mt = m["metrics"]
            for k in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"):
                assert 0.0 <= mt[k] <= 1.0
            assert mt["roc_auc_ovr_macro"] is None or 0.0 <= mt["roc_auc_ovr_macro"] <= 1.0
            cm = np.array(mt["confusion_matrix"])
            assert cm.sum() == mt["n_test"] == n_test
            assert mt["accuracy"] == pytest.approx(np.trace(cm) / cm.sum())
            # per-class recall recomputed from the confusion matrix
            for k, c in enumerate(tr_["classes"]):
                support = cm[k].sum()
                assert mt["per_class"][c]["support"] == support
                assert mt["per_class"][c]["recall"] == pytest.approx(cm[k, k] / support)
            assert mt["balanced_accuracy"] == pytest.approx(
                np.mean([mt["per_class"][c]["recall"] for c in tr_["classes"]])
            )


def test_cart_is_pruned_and_depth_limited(trained):
    for t, tr_ in trained.report["targets"].items():
        cart = tr_["models"]["cart"]
        assert cart["tree"]["depth"] <= 6
        assert cart["selected_params"]["ccp_alpha"] in tr.CCP_ALPHAS
        assert cart["tree"]["n_leaves"] >= 2


def test_imputer_fit_on_training_split_only(synthetic_df, trained):
    """Median imputer statistics must equal the TRAIN-split medians, not full-data medians."""
    opts = tr.TrainOptions(**{k: v for k, v in trained.report["options"].items() if k in ("test_size", "seed")})
    df = synthetic_df.reset_index(drop=True)
    targets = list(trained.report["targets"])
    train_idx, test_idx, _ = tr._split(df, targets, opts, [])
    assert set(train_idx).isdisjoint(test_idx)
    X = build_feature_frame(df, True)
    pipe = trained.bundle["targets"]["depression_risk"]["pipeline"]
    stats = pipe.named_steps["imputer"].statistics_
    np.testing.assert_allclose(stats, X.iloc[train_idx].median().to_numpy())


def test_node_counts_come_from_training_rows_only(trained):
    for t, tb in trained.bundle["targets"].items():
        assert tb["node_counts"][0].sum() == tb["n_train"]  # root holds every training row
        assert tb["n_train"] <= trained.report["split"]["n_train"]


def test_no_oversampling_test_split_untouched(trained):
    # Without resampling, class_distribution (all) == train + test per class.
    for tr_ in trained.report["targets"].values():
        d = tr_["class_distribution"]
        for c in tr_["classes"]:
            assert d["all"][c] == d["train"][c] + d["test"][c]


def test_training_is_deterministic_for_a_seed():
    df = sanitise(generate_synthetic_dataset(400, seed=2))
    opts = tr.TrainOptions(cv_folds=3, seed=9, models=["cart"], targets=["depression_risk"])
    meta = {"id": "x", "is_synthetic": True}
    a = tr.train_and_evaluate(df, meta, opts).report["targets"]["depression_risk"]["models"]["cart"]
    b = tr.train_and_evaluate(df, meta, opts).report["targets"]["depression_risk"]["models"]["cart"]
    assert a["selected_params"] == b["selected_params"] and a["metrics"] == b["metrics"]


def test_participant_grouped_split_has_no_overlap():
    df = sanitise(generate_synthetic_dataset(400, seed=4))
    df["participant_id"] = [f"P{i // 2}" for i in range(len(df))]  # two rows per participant
    train_idx, test_idx, strategy = tr._split(df, ["depression_risk"], tr.TrainOptions(), [])
    assert "participant-grouped" in strategy
    assert set(df.loc[train_idx, "participant_id"]).isdisjoint(df.loc[test_idx, "participant_id"])


def test_options_validation():
    with pytest.raises(ValueError):
        tr.TrainOptions(selection_metric="accuracy").validate()
    with pytest.raises(ValueError):
        tr.TrainOptions(models=["random_forest"]).validate()
    with pytest.raises(ValueError):
        tr.TrainOptions(test_size=0.9).validate()


def test_unavailable_optional_library_is_skipped_not_fatal(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("xgboost"):
            raise ImportError("simulated missing xgboost")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = sanitise(generate_synthetic_dataset(300, seed=6))
    res = tr.train_and_evaluate(
        df, {"id": "x", "is_synthetic": True},
        tr.TrainOptions(cv_folds=2, models=["cart", "xgboost", "logistic_regression"], targets=["stress_risk"]),
    )
    m = res.report["targets"]["stress_risk"]["models"]["xgboost"]
    assert m["status"] == "skipped" and "simulated missing xgboost" in m["reason"]
    assert res.report["targets"]["stress_risk"]["models"]["cart"]["status"] == "ok"


def test_csv_export_has_one_row_per_target_and_model(trained):
    csv = tr.report_to_csv(trained.report)
    df = pd.read_csv(pd.io.common.StringIO(csv))
    assert len(df) == 3 * 5
    assert {"target", "model", "f1_macro", "recall_High", "synthetic_data"} <= set(df.columns)
    assert df["synthetic_data"].all()


def test_overall_target_trained_only_when_present():
    df = sanitise(generate_synthetic_dataset(400, seed=8, include_overall=True))
    res = tr.train_and_evaluate(
        df, {"id": "x", "is_synthetic": True},
        tr.TrainOptions(cv_folds=2, models=["cart"], targets=["overall_risk", "stress_risk"]),
    )
    assert set(res.report["targets"]) == {"overall_risk", "stress_risk"}
    assert "depression_risk" in res.report["skipped_targets"]


# --- test_explanations ---
"""Explanations must be the model's own: rules + decision path must reproduce its predictions."""




def _evaluate_rule(rule, features: dict[str, float]) -> bool:
    for c in rule["conditions"]:
        v = np.float32(features[c["feature"]])
        ok = v <= c["threshold"] if c["operator"] == "<=" else v > c["threshold"]
        if not ok:
            return False
    return True


def test_rules_reproduce_pipeline_predictions_independently(trained):
    rng = random.Random(0)
    for target, tb in trained.bundle["targets"].items():
        rules = trained.explain["targets"][target]["rules"]
        classes = tb["classes"]
        assert len(rules) == trained.explain["targets"][target]["n_leaves"]
        for _ in range(60):
            ans = random_answers(rng)
            frame = answers_to_frame(ans, trained.bundle["include_derived_totals"])
            feats = {c: float(frame.iloc[0][c]) for c in frame.columns}
            matched = [r for r in rules if _evaluate_rule(r, feats)]
            assert len(matched) == 1  # leaves partition the input space
            expected = classes[int(tb["pipeline"].predict(frame)[0])]
            assert matched[0]["prediction"] == expected


def test_api_decision_path_matches_prediction_and_rules(model_client, trained):
    rng = random.Random(1)
    for _ in range(25):
        ans = random_answers(rng)
        r = model_client.post("/api/screening/predict", json={"answers": ans}).json()
        assert r["status"] == "ok" and len(r["conditions"]) == 3
        feats = r["features_used"]
        for cond in r["conditions"]:
            path = cond["decision_path"]
            rules = trained.explain["targets"][cond["target"]]["rules"]
            rule = next(x for x in rules if x["leaf_id"] == path["leaf_id"])
            assert _evaluate_rule(rule, feats)  # the reported leaf's rule holds for these inputs
            assert rule["prediction"] == cond["risk_level"] == path["prediction"]
            # each step is consistent with the recorded value, threshold and direction
            for s in path["steps"]:
                assert (s["value"] <= s["threshold"]) == (s["direction"] == "left")
                assert s["value"] == feats[s["feature"]]
            assert [s["node_id"] for s in path["steps"]] == path["node_ids"][:-1]
            # leaf evidence counts are real training counts
            ev = path["leaf_evidence"]
            assert ev["n_train_samples"] == sum(ev["class_counts"].values()) > 0
            assert abs(sum(ev["class_share"].values()) - 1) < 1e-9


def test_features_used_match_submitted_answers_and_totals(model_client):
    ans = random_answers(random.Random(5))
    r = model_client.post("/api/screening/predict", json={"answers": ans}).json()
    used = r["features_used"]
    assert all(used[k] == v for k, v in ans.items())
    assert used["phq9_total"] == sum(v for k, v in ans.items() if k.startswith("phq9_"))
    assert used["gad7_total"] == sum(v for k, v in ans.items() if k.startswith("gad7_"))
    assert r["reference_scores"]["phq9"]["total"] == used["phq9_total"]


def test_tree_endpoint_serves_the_fitted_tree(model_client, trained):
    r = model_client.get("/api/model/tree", params={"target": "anxiety_risk"}).json()
    assert r["target"] == "anxiety_risk" and r["is_synthetic"] is True
    exp = trained.explain["targets"]["anxiety_risk"]
    assert r["n_leaves"] == exp["n_leaves"] and r["tree"]["n_samples"] == trained.bundle["targets"]["anxiety_risk"]["n_train"]

    def count_leaves(n):
        return 1 if n["is_leaf"] else count_leaves(n["left"]) + count_leaves(n["right"])

    assert count_leaves(r["tree"]) == r["n_leaves"] == len(r["rules"])
    imp = r["feature_importance"]
    assert abs(sum(i["impurity"] for i in imp) - 1.0) < 1e-6
    assert model_client.get("/api/model/tree", params={"target": "nope"}).status_code == 404


# --- test_screening_api ---
def zeros(**over):
    a = {i: 0 for i in ITEM_IDS}
    a.update(over)
    return a


# ---------- validation ------------------------------------------------------
@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.pop("gad7_4"),  # missing
        lambda a: a.update(phq9_1=4),  # above range
        lambda a: a.update(phq9_1=-1),  # below range
        lambda a: a.update(phq9_1="2"),  # string (strict)
        lambda a: a.update(phq9_1=1.5),  # float
        lambda a: a.update(phq9_1=True),  # bool
        lambda a: a.update(phq9_1=None),
        lambda a: a.update(student_name="x"),  # unknown/identifying field rejected
    ],
)
def test_invalid_payloads_rejected(client, mutate):
    ans = zeros()
    mutate(ans)
    r = client.post("/api/screening/predict", json={"answers": ans})
    assert r.status_code == 422


def test_malformed_bodies(client):
    assert client.post("/api/screening/predict", json={}).status_code == 422
    assert client.post("/api/screening/predict", content=b"not json", headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/api/screening/predict", json={"answers": zeros(), "email": "a@b.c"}).status_code == 422


def test_oversized_body_rejected(client):
    r = client.post("/api/screening/predict", json={"answers": zeros(), "pad": "x" * 40_000})
    assert r.status_code == 413


# ---------- no model --------------------------------------------------------
def test_model_unavailable_state_is_explicit(client):
    assert client.get("/api/model/status").json()["trained"] is False
    m = client.get("/api/model/metrics").json()
    assert m["available"] is False and "report" not in m
    assert client.get("/api/model/tree").status_code == 503
    r = client.post("/api/screening/predict", json={"answers": random_answers(random.Random(2))})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "model_unavailable" and body["model"]["available"] is False
    assert body["conditions"] == [] and body["summary"] is None and body["features_used"] is None
    assert body["reference_scores"]["phq9"]["total"] >= 0  # model-independent scoring still works
    assert "not a medical diagnosis" in body["disclaimer"]


def test_features_endpoint_works_without_model(client):
    f = client.get("/api/model/features").json()
    assert f["trained"] is False and len(f["item_ids"]) == 24 and f["levels"] == ["Low", "Moderate", "High"]


# ---------- safety flow -----------------------------------------------------
def test_safety_not_triggered_when_item9_is_zero(model_client):
    r = model_client.post("/api/screening/predict", json={"answers": zeros(phq9_1=3, phq9_2=3)}).json()
    assert r["safety"]["triggered"] is False and r["safety"]["resources"] is None and r["safety"]["level"] == "none"


@pytest.mark.parametrize("value,level", [(1, "support"), (2, "urgent"), (3, "urgent")])
def test_safety_triggered_by_any_nonzero_item9(model_client, value, level):
    r = model_client.post("/api/screening/predict", json={"answers": zeros(phq9_9=value)}).json()
    s = r["safety"]
    assert s["triggered"] is True and s["level"] == level
    assert s["resources"]["emergency"] and s["resources"]["helplines"] and s["resources"]["human_support"]
    assert "emergency" in s["message"].lower()
    assert any("today" in x for x in r["recommendations"])


def test_safety_is_independent_of_model_result_and_of_model_availability(client, model_client):
    # All other answers zero => every condition should be low-ish, yet safety still triggers.
    ans = zeros(phq9_9=1)
    with_model = model_client.post("/api/screening/predict", json={"answers": ans}).json()
    without = client.post("/api/screening/predict", json={"answers": ans}).json()
    assert with_model["safety"]["triggered"] and without["safety"]["triggered"]
    assert without["status"] == "model_unavailable"
    assert with_model["safety"]["resources"] == without["safety"]["resources"]


def test_low_result_text_never_promises_good_health(model_client):
    r = model_client.post("/api/screening/predict", json={"answers": zeros()}).json()
    text = " ".join([r["disclaimer"], *r["recommendations"], *(c["meaning"] for c in r["conditions"])]).lower()
    assert "guarantee" in text and "diagnos" in text


def test_summary_is_labelled_derived_and_equals_max_level(model_client):
    rng = random.Random(9)
    order = ["Low", "Moderate", "High"]
    for _ in range(15):
        r = model_client.post("/api/screening/predict", json={"answers": random_answers(rng)}).json()
        top = max((c["risk_level"] for c in r["conditions"]), key=order.index)
        assert r["summary"]["highest_level"] == top and r["summary"]["derived"] is True


def test_reference_score_bands(client):
    r = client.post("/api/screening/predict", json={"answers": zeros(**{f"phq9_{i}": 3 for i in range(1, 10)})}).json()
    assert r["reference_scores"]["phq9"] == {"total": 27, "max": 27, "band": "Severe"}
    r = client.post("/api/screening/predict", json={"answers": zeros(**{f"gad7_{i}": 2 for i in range(1, 8)})}).json()
    assert r["reference_scores"]["gad7"]["total"] == 14 and r["reference_scores"]["gad7"]["band"] == "Moderate"
    assert r["reference_scores"]["stress"]["available"] is False


# ---------- metrics / status with a model -----------------------------------
def test_metrics_and_status_are_the_real_report(model_client, trained):
    st = model_client.get("/api/model/status").json()
    assert st["trained"] and st["is_synthetic"] and "SYNTHETIC" in st["message"]
    m = model_client.get("/api/model/metrics").json()
    assert m["available"] and m["report"]["run_id"] == trained.report["run_id"]
    assert m["report"]["targets"]["stress_risk"]["models"]["cart"]["metrics"] == \
        trained.report["targets"]["stress_risk"]["models"]["cart"]["metrics"]


# ---------- privacy: logging ------------------------------------------------
def test_answers_are_never_logged(model_client, caplog):
    ans = random_answers(random.Random(3), item9=2)
    with caplog.at_level(logging.DEBUG):
        r = model_client.post("/api/screening/predict?probe=SECRETQUERY", json={"answers": ans})
    assert r.status_code == 200
    # Only the application's own loggers are under test (httpx's client-side logger belongs to the test client).
    app_records = [rec for rec in caplog.records if rec.name.startswith("app")]
    assert app_records, "access log should still record method/path/status"
    blob = "\n".join(rec.getMessage() for rec in app_records)
    assert "/api/screening/predict" in blob
    for forbidden in ("phq9", "gad7", "sup_", "answers", "SECRETQUERY", "risk_level"):
        assert forbidden not in blob


def test_answers_are_not_persisted(model_client, tmp_path):
    model_client.post("/api/screening/predict", json={"answers": random_answers(random.Random(4))})
    root = tmp_path
    files = [p for p in root.rglob("*") if p.is_file()]
    assert all("uploads" not in str(p) for p in files)  # nothing but model artifacts written


# ---------- request controls ------------------------------------------------
def test_predict_rate_limit(tmp_path):
    c = TestClient(create_app(make_settings(tmp_path, predict_rate_limit_per_min=3)))
    codes = [c.post("/api/screening/predict", json={"answers": zeros()}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    r = c.post("/api/screening/predict", json={"answers": zeros()})
    assert int(r.headers["retry-after"]) >= 1


def test_cors_allows_only_configured_origins(client):
    ok = client.options(
        "/api/screening/predict",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options(
        "/api/screening/predict",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers


# --- test_admin ---
ADMIN_GETS = ["/api/admin/datasets", "/api/admin/evaluations", "/api/admin/jobs"]


@pytest.mark.parametrize("path", ADMIN_GETS)
def test_admin_requires_credentials(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong-key-wrong-key-wrong"}).status_code == 401
    assert client.get(path, headers={"Authorization": f"Basic {ADMIN_KEY}"}).status_code == 401
    assert client.get(path, headers={"X-Admin-Key": ADMIN_KEY}).status_code == 401  # only Bearer accepted


def test_all_mutating_admin_routes_are_protected(client):
    f = {"file": ("d.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")}
    assert client.post("/api/admin/dataset/validate", files=f).status_code == 401
    assert client.post("/api/admin/dataset/synthetic", json={}).status_code == 401
    assert client.post("/api/admin/model/train", json={"dataset_id": "0" * 12}).status_code == 401
    assert client.post("/api/admin/model/activate/20250101T000000Z-abcd").status_code == 401
    assert client.get("/api/admin/evaluations/20250101T000000Z-abcd/export").status_code == 401


def test_valid_key_accepted(client, admin_headers):
    assert client.get("/api/admin/datasets", headers=admin_headers).json() == []


def test_admin_disabled_without_or_with_weak_key(tmp_path):
    for key in (None, "", "short"):
        c = TestClient(create_app(make_settings(tmp_path, admin_api_key=key)))
        r = c.get("/api/admin/datasets", headers={"Authorization": f"Bearer {key or 'x'}"})
        assert r.status_code == 503 and "disabled" in r.json()["detail"].lower()


def test_failed_auth_lockout(tmp_path):
    c = TestClient(create_app(make_settings(tmp_path, admin_failed_auth_per_min=3)))
    bad = {"Authorization": "Bearer nope-nope-nope-nope-nope"}
    codes = [c.get("/api/admin/datasets", headers=bad).status_code for _ in range(5)]
    assert codes[:3] == [401, 401, 401] and codes[3:] == [429, 429]
    # even the right key is throttled during lockout (prevents brute-force probing)
    assert c.get("/api/admin/datasets", headers={"Authorization": f"Bearer {ADMIN_KEY}"}).status_code == 429


def test_public_endpoints_need_no_key(client):
    for p in ("/api/health", "/api/project-info", "/api/questionnaire", "/api/model/status", "/api/model/metrics"):
        assert client.get(p).status_code == 200


def _wait(c, h, job_id, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/admin/jobs/{job_id}", headers=h).json()
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.5)
    raise AssertionError("job timeout")


def test_full_admin_workflow(tmp_path, admin_headers):
    # Long training jobs are polled quickly here, so lift the per-minute admin limit for this test only.
    c = TestClient(create_app(make_settings(tmp_path / 'wf', admin_rate_limit_per_min=100000)))
    h = admin_headers
    # 1. synthetic dataset registered and flagged
    d = c.post("/api/admin/dataset/synthetic", json={"n_rows": 300, "seed": 1}, headers=h)
    assert d.status_code == 201
    ds = d.json()["dataset"]
    assert ds["is_synthetic"] and "SYNTHETIC" in ds["name"]
    assert c.get("/api/admin/datasets", headers=h).json()[0]["id"] == ds["id"]

    # 2. unknown dataset / bad options
    assert c.post("/api/admin/model/train", json={"dataset_id": "0" * 12}, headers=h).status_code == 404
    assert c.post("/api/admin/model/train", json={"dataset_id": "../../etc/x"}, headers=h).status_code == 422
    assert c.post("/api/admin/model/train", json={"dataset_id": ds["id"], "selection_metric": "accuracy"}, headers=h).status_code == 422
    assert c.post("/api/admin/model/train", json={"dataset_id": ds["id"], "models": ["random_forest"]}, headers=h).status_code == 422

    # 3. train (small/fast config); a second job while running is refused
    body = {"dataset_id": ds["id"], "cv_folds": 2, "models": ["cart", "logistic_regression"]}
    r = c.post("/api/admin/model/train", json=body, headers=h)
    assert r.status_code == 202
    job = r.json()
    assert c.post("/api/admin/model/train", json=body, headers=h).status_code == 409
    done = _wait(c, h, job["id"])
    assert done["status"] == "succeeded", done
    run1 = done["result"]["run_id"]
    assert done["result"]["activated"] is True  # first run auto-activates

    # 4. model is live and flagged synthetic
    st = c.get("/api/model/status").json()
    assert st["trained"] and st["run_id"] == run1 and st["is_synthetic"]

    # 5. second run is appended, does NOT overwrite or auto-activate
    job2 = c.post("/api/admin/model/train", json={**body, "seed": 7}, headers=h).json()
    done2 = _wait(c, h, job2["id"])
    run2 = done2["result"]["run_id"]
    assert run2 != run1 and done2["result"]["activated"] is False
    runs = c.get("/api/admin/evaluations", headers=h).json()
    assert {r["run_id"] for r in runs} == {run1, run2}
    assert [r["run_id"] for r in runs if r["active"]] == [run1]
    assert c.get(f"/api/admin/evaluations/{run1}", headers=h).json()["run_id"] == run1  # still intact

    # 6. exports
    j = c.get(f"/api/admin/evaluations/{run1}/export?format=json", headers=h)
    assert j.headers["content-type"].startswith("application/json") and j.json()["run_id"] == run1
    cs = c.get(f"/api/admin/evaluations/{run1}/export?format=csv", headers=h)
    assert cs.headers["content-type"].startswith("text/csv") and cs.text.splitlines()[0].startswith("run_id,")
    assert c.get(f"/api/admin/evaluations/{run1}/export?format=xml", headers=h).status_code == 422

    # 7. explicit activation + unknown / traversal ids
    assert c.post(f"/api/admin/model/activate/{run2}", headers=h).json() == {"active_run_id": run2}
    assert c.get("/api/model/status").json()["run_id"] == run2
    assert c.post("/api/admin/model/activate/20250101T000000Z-abcd", headers=h).status_code == 404
    assert c.get("/api/admin/evaluations/..%2F..%2Fetc", headers=h).status_code in (404, 422)
    assert c.get("/api/admin/evaluations/not-a-run", headers=h).status_code == 404


def test_training_failure_reported_not_crashing(client, admin_headers, tmp_path):
    # A dataset whose only target has a single class cannot be uploaded (validation blocks it).
    import pandas as pd
    from server import generate_synthetic_dataset

    df = generate_synthetic_dataset(300, seed=3)
    df = df.drop(columns=["anxiety_risk", "stress_risk"])
    df["depression_risk"] = "Low"
    r = client.post(
        "/api/admin/dataset/upload",
        files={"file": ("d.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")},
        data={"name": "one-class", "provenance": "unit test"},
        headers=admin_headers,
    )
    assert r.status_code == 422
