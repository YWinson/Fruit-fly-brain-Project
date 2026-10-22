"""Masked MLP for M2 (Stage 2 + Stage 3).

Architecture (fixed across all arms)::

    z (32) -> W_eff = (W * Sigma * M) -> ReLU (64) -> linear (4 classes)

``M`` and ``Sigma`` are fixed buffers, so masked weights receive zero
gradient. The only difference between arms is ``M`` (and, for the
``equiv_dropout`` arm, training-time connection dropout). The initial weights
are seeded identically for every arm, which keeps the comparison paired.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin


class _MaskedMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_classes: int,
        mask: np.ndarray,
        sign: np.ndarray,
        dropout_keep_prob: float,
    ):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))
        self.register_buffer("sign", torch.as_tensor(sign, dtype=torch.float32))
        self.dropout_keep_prob = float(dropout_keep_prob)

    def effective_weight(self) -> torch.Tensor:
        weight = self.fc1.weight * self.sign * self.mask
        if self.training and self.dropout_keep_prob < 1.0:
            # Connection-level inverted dropout. The expected number of active
            # connections is E = keep_prob * hidden_dim * n_features, matching
            # the sparse arms' edge budget.
            keep = self.dropout_keep_prob
            drop = torch.bernoulli(torch.full_like(weight, keep)) / keep
            weight = weight * drop
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.effective_weight()
        hidden = F.relu(F.linear(x, weight, self.fc1.bias))
        return self.fc2(hidden)


class MaskedMLPClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn-compatible wrapper around the masked PyTorch MLP."""

    def __init__(
        self,
        mask: np.ndarray,
        sign: np.ndarray,
        hidden_dim: int = 64,
        epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout_keep_prob: float = 1.0,
        random_state: int = 0,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.mask = mask
        self.sign = sign
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_keep_prob = dropout_keep_prob
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _check_arrays(self, n_features: int) -> tuple[np.ndarray, np.ndarray]:
        mask = np.asarray(self.mask, dtype=np.float32)
        sign = np.asarray(self.sign, dtype=np.float32)
        expected = (int(self.hidden_dim), int(n_features))
        if mask.shape != expected:
            raise ValueError(f"mask shape {mask.shape} != expected {expected}")
        if sign.shape != expected:
            raise ValueError(f"sign shape {sign.shape} != expected {expected}")
        return mask, sign

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.shape}")
        mask, sign = self._check_arrays(X.shape[1])

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("MaskedMLPClassifier needs at least two classes")
        class_to_index = {value: index for index, value in enumerate(self.classes_)}
        y_index = np.array([class_to_index[value] for value in y], dtype=np.int64)
        self.n_features_in_ = X.shape[1]

        # The MLP is tiny (32 -> 64 -> 4); PyTorch's default thread pool adds
        # substantial overhead on these shapes. One thread is much faster.
        torch.set_num_threads(1)
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state) % (2**32 - 1))
        self.model_ = _MaskedMLP(
            n_features=X.shape[1],
            hidden_dim=int(self.hidden_dim),
            n_classes=len(self.classes_),
            mask=mask,
            sign=sign,
            dropout_keep_prob=float(self.dropout_keep_prob),
        ).to(self.device)
        self.model_.train()

        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y_index, dtype=torch.long, device=self.device)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = nn.CrossEntropyLoss()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.random_state))

        self.loss_curve_ = []
        n_samples = len(X_tensor)
        for epoch in range(int(self.epochs)):
            permutation = torch.randperm(n_samples, generator=generator)
            epoch_loss = 0.0
            for start in range(0, n_samples, int(self.batch_size)):
                indices = permutation[start : start + int(self.batch_size)]
                batch_x = X_tensor[indices]
                batch_y = y_tensor[indices]
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(batch_x)
            self.loss_curve_.append(epoch_loss / max(1, n_samples))
            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"  epoch {epoch + 1}/{self.epochs} loss={self.loss_curve_[-1]:.4f}")

        self.model_.eval()
        return self

    def predict_proba(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("MaskedMLPClassifier must be fitted first")
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model_(tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
