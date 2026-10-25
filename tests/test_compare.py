import numpy as np
import pytest

from chemdner_bench.cem import _f1
from chemdner_bench.compare import (boot_f1, classify, curve_rule, doc_weights, drift_rule, error_breakdown,
                                    f1_from_weights, group_mean, overall_type)


@pytest.mark.parametrize("lo, hi, expected", [
    (0.02, 0.05, "優越"),
    (0.005, 0.03, "正だが幅は不明"),
    (-0.008, 0.009, "同等"),
    (0.002, 0.008, "同等"),  # 0 の片側でも ±幅 の中なら同等
    (-0.009, -0.001, "同等"),
    (-0.03, -0.004, "負だが幅は不明"),
    (-0.05, -0.02, "劣後"),
    (-0.02, 0.005, "不明"),
    (-0.005, 0.02, "不明"),
    (-0.02, 0.02, "不明"),
])
def test_classify(lo, hi, expected):
    assert classify(lo, hi) == expected


def test_overall_type_order():
    assert overall_type(["優越"] * 3, "優越") == "A"
    assert overall_type(["優越"] * 3, "正だが幅は不明") == "B"  # 曲線の最高点にも優越でなければ A にしない
    assert overall_type(["優越", "劣後", "優越"], "優越") == "D"
    assert overall_type(["同等"] * 3, "同等") == "C"
    assert overall_type(["同等", "同等", "劣後"], "同等") == "D"  # D は C より先
    assert overall_type(["優越", "同等", "不明"], "優越") == "B"


def test_curve_and_drift_rules():
    assert curve_rule(0.015, 0.001) == "伸び続けた"
    assert curve_rule(0.015, -0.001) == "頭打ちに近い"
    assert curve_rule(-0.012, -0.03) == "下がった"
    assert curve_rule(0.004, -0.002) == "頭打ちに近い"
    assert drift_rule(-0.006, 0.008) == "モデルの変化は見えない"
    assert drift_rule(-0.006, 0.012) == "外れた"


def test_doc_weights_count_draws():
    idx = np.array([[0, 0, 2], [1, 2, 2]])
    assert doc_weights(idx, 3).tolist() == [[2, 0, 1], [0, 1, 2]]


def test_f1_from_weights_matches_resummed_rows():
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 5, size=(2, 6, 3))
    idx = rng.integers(0, 6, size=(4, 6))
    got = f1_from_weights(doc_weights(idx, 6), rows)
    want = np.stack([_f1(rows[:, i, :].sum(axis=1)) for i in idx])
    assert np.allclose(got, want)


def test_boot_f1_pairs_documents_across_runs():
    rows = np.random.default_rng(2).integers(0, 5, size=(1, 50, 3))
    f1 = boot_f1(np.concatenate([rows, rows]), 300, np.random.default_rng(0), chunk=70)
    assert f1.shape == (300, 2)
    assert np.array_equal(f1[:, 0], f1[:, 1])  # 同じ予測の 2 回は、同じ抽出で同じ F1 になる


def test_group_mean_resamples_runs_within_group():
    f1 = np.array([[0.1, 0.2, 0.3, 0.9]] * 1000)
    assert np.allclose(group_mean(f1, [0, 1, 2], None), 0.2)
    dist = group_mean(f1, [0, 1, 2], np.random.default_rng(0))
    assert dist.min() >= 0.1 and dist.max() <= 0.3  # 群の外の回（0.9）は混ざらない
    assert dist.std() > 0
    assert np.allclose(group_mean(f1, [3], np.random.default_rng(0)), 0.9)  # 1 回だけの群は抽出しない


def test_error_breakdown():
    gold = {"1": {("A", 0, 5), ("A", 10, 15), ("T", 0, 3), ("A", 30, 30)}}
    pred = {"1": {("A", 0, 5), ("A", 9, 15), ("A", 20, 25), ("T", 0, 3)}}
    e = error_breakdown(gold, pred, ["1"])
    assert (e["tp"], e["fp_boundary"], e["fp_spurious"]) == (2, 1, 1)
    assert (e["fn_boundary"], e["fn_missed"]) == (1, 1)  # 長さ 0 の正解はどの予測とも重ならない
    assert e["overlap_precision"] == pytest.approx(3 / 4)
    assert e["overlap_recall"] == pytest.approx(3 / 4)
