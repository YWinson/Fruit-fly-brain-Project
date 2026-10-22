"""M2 unit and smoke tests.

Run with the ``eeg`` environment (torch is required for the masked MLP).
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "M2_先验注入"
    / "configs"
    / "m2_synthetic.yaml"
)
EEGNET_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "M2_先验注入"
    / "configs"
    / "m2_bnci2014_001_eegnet.yaml"
)

from common_utils.metadata import normalize_metadata
from M1_评测管线.data import EpochDataset, load_synthetic_dataset
from M2_先验注入.clusters import Granularity
from M2_先验注入.config import load_m2_config
from M2_先验注入.evaluate import run_all
from M2_先验注入.features import MultiBandCSP
from M2_先验注入.masks import (
    balanced_topk_mask,
    bipartite_degree_preserving_rewire,
    build_bio_mask,
    er_random_mask,
)
from M2_先验注入.model import MaskedMLPClassifier
from M2_先验注入.motifs import bifan_count, ffl_count


def _fake_granularity(seed: int = 0):
    rng = np.random.default_rng(seed)
    W = rng.random((64, 32))
    W_exc = W * 0.6
    W_inh = W * 0.4
    return SimpleNamespace(
        name="fine",
        k_out=64,
        k_in=32,
        W_syn_expanded=W,
        W_exc_expanded=W_exc,
        W_inh_expanded=W_inh,
    )


def test_balanced_mask_density_and_rewiring_invariants():
    result = _fake_granularity()
    mask, sign, info = build_bio_mask(result, density=0.10, selection="balanced")
    assert mask.shape == (64, 32)
    assert 190 <= int(mask.sum()) <= 220
    assert sign.shape == (64, 32)

    rewired, rewire_info = bipartite_degree_preserving_rewire(mask, seed=0)
    assert np.array_equal(mask.sum(axis=0), rewired.sum(axis=0))
    assert np.array_equal(mask.sum(axis=1), rewired.sum(axis=1))
    assert int(rewired.sum()) == int(mask.sum())
    assert rewire_info["hamming"] >= int(mask.sum())

    random_mask = er_random_mask(64, 32, int(mask.sum()), seed=0)
    assert int(random_mask.sum()) == int(mask.sum())


def test_balanced_topk_is_row_spread():
    rng = np.random.default_rng(1)
    W = rng.random((64, 32))
    mask = balanced_topk_mask(W, density=0.10)
    assert 190 <= int(mask.sum()) <= 220
    assert (mask.sum(axis=1) > 0).sum() >= 60


def test_masked_mlp_shapes_and_probabilities():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 32)).astype(np.float32)
    y = np.repeat(np.arange(4), 30)
    mask = rng.random((64, 32)) < 0.10
    sign = np.ones((64, 32), dtype=np.int8)

    clf = MaskedMLPClassifier(
        mask=mask,
        sign=sign,
        epochs=20,
        batch_size=32,
        random_state=0,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    assert proba.shape == (120, 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert pred.shape == (120,)
    assert set(np.unique(pred)).issubset(set(range(4)))


def test_multiband_csp_output_dimension():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 22, 100)).astype(np.float32)
    y = np.repeat(np.arange(4), 10)
    transformer = MultiBandCSP(
        bands=((8, 13), (13, 30)),
        n_components=16,
        sfreq=100.0,
    )
    features = transformer.fit_transform(X, y)
    assert features.shape == (40, 32)
    assert transformer.transform(X[:5]).shape == (5, 32)


def test_eegnet_feature_extractor_output():
    from M2_先验注入.eegnet import EEGNetFeatureExtractor

    rng = np.random.default_rng(5)
    X = rng.normal(size=(24, 8, 100)).astype(np.float32)
    y = np.repeat(np.arange(4), 6)
    extractor = EEGNetFeatureExtractor(
        feature_dim=32,
        epochs=2,
        batch_size=8,
        random_state=0,
    )
    features = extractor.fit_transform(X, y)
    assert features.shape == (24, 32)
    assert extractor.transform(X[:5]).shape == (5, 32)


def test_motif_counts_on_toy_graphs():
    ffl = np.zeros((3, 3), dtype=int)
    ffl[0, 1] = ffl[0, 2] = ffl[1, 2] = 1
    assert ffl_count(ffl) == 1

    bifan = np.zeros((4, 4), dtype=int)
    bifan[0, 2] = bifan[0, 3] = bifan[1, 2] = bifan[1, 3] = 1
    assert bifan_count(bifan) == 1


def test_m2_config_synthetic():
    config = load_m2_config(CONFIG_PATH)
    assert config["dataset"]["name"] == "synthetic"
    assert len(config["preprocess"]["bands"]) * config["preprocess"]["csp_components"] == 32


def test_m2_eegnet_config():
    config = load_m2_config(EEGNET_CONFIG_PATH)
    assert config["preprocess"]["stage1"] == "eegnet"
    assert int(config["preprocess"]["eegnet"]["feature_dim"]) == 32


def test_m2_end_to_end_synthetic_fold():
    config = load_m2_config(CONFIG_PATH)
    dataset = load_synthetic_dataset(config)
    granularity = _fake_granularity(seed=4)
    results, predictions = run_all(
        dataset=dataset,
        config=config,
        arms=["bio_sparse", "dense_baseline"],
        seeds=[0],
        granularity_result=granularity,
        max_folds=1,
    )
    assert len(results) > 0
    assert set(results["session"].astype(str)).issuperset({"all"})
    assert len(predictions) == len(dataset.metadata) // 3 * 2  # 2 arms x one subject
    assert results["balanced_acc"].between(0, 1).all()
