"""EEGNet-based Stage 1 feature extractor for M2.

The M2 draft's Stage 1 is a 32-D feature alignment layer. Here we replace the
multi-band CSP with a compact EEGNet trunk trained **once per LOSO fold** on
the training subjects; its 32-D penultimate representation is then frozen and
shared by all masks. The masked 64x32 head is trained on those features by the
existing M2 evaluation code.

The trunk is a standard EEGNet-8,2-style network (F1=8, D=2, F2=16) with a
32-D feature layer. A temporary 4-class linear head is used only to train the
trunk; it is discarded before feature extraction, so the mask comparison stays
paired across arms.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin


class EEGNetTrunk(nn.Module):
    """Compact EEGNet trunk producing a ``feature_dim`` representation."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        feature_dim: int = 32,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(
            F1 * D, F2, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        self.conv4 = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flatten_dim = int(self._features(dummy).numel())
        self.fc = nn.Linear(flatten_dim, feature_dim)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = F.elu(x)
        x = self.drop1(self.pool1(x))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.drop2(self.pool2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        return self.fc(x.flatten(1))


class EEGNetFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn-compatible Stage 1 transformer.

    ``fit`` trains the EEGNet trunk + a temporary 4-class head on the training
    fold. ``transform`` returns the frozen trunk's 32-D features. The
    per-channel scaler is fitted on the training fold only.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.25,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        random_state: int = 0,
        device: str = "cpu",
        verbose: bool = False,
        num_threads: int = 4,
    ):
        self.feature_dim = feature_dim
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.num_threads = num_threads

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"expected (epochs, channels, times), got {X.shape}")
        return (X - self.mean_) / self.std_

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"expected (epochs, channels, times), got {X.shape}")
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True)
        self.std_[self.std_ < 1e-6] = 1.0
        Xs = self._prepare(X)

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("EEGNetFeatureExtractor needs at least two classes")
        class_to_index = {value: index for index, value in enumerate(self.classes_)}
        y_index = np.array([class_to_index[value] for value in y], dtype=np.int64)

        # Conv2d benefits from a few threads; unlike the tiny masked MLP,
        # one thread is too slow for EEGNet.
        torch.set_num_threads(max(1, int(self.num_threads)))
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state) % (2**32 - 1))

        self.trunk_ = EEGNetTrunk(
            n_channels=X.shape[1],
            n_times=X.shape[2],
            feature_dim=int(self.feature_dim),
            F1=int(self.F1),
            D=int(self.D),
            F2=int(self.F2),
            kernel_length=int(self.kernel_length),
            dropout=float(self.dropout),
        ).to(self.device)
        head = nn.Linear(int(self.feature_dim), len(self.classes_)).to(self.device)

        X_tensor = torch.as_tensor(Xs, dtype=torch.float32, device=self.device)
        X_tensor = X_tensor.unsqueeze(1)
        y_tensor = torch.as_tensor(y_index, dtype=torch.long, device=self.device)
        optimizer = torch.optim.Adam(
            list(self.trunk_.parameters()) + list(head.parameters()),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = nn.CrossEntropyLoss()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.random_state))

        self.loss_curve_ = []
        n_samples = len(X_tensor)
        for epoch in range(int(self.epochs)):
            self.trunk_.train()
            head.train()
            permutation = torch.randperm(n_samples, generator=generator)
            epoch_loss = 0.0
            for start in range(0, n_samples, int(self.batch_size)):
                indices = permutation[start : start + int(self.batch_size)]
                batch_x = X_tensor[indices]
                batch_y = y_tensor[indices]
                optimizer.zero_grad(set_to_none=True)
                logits = head(self.trunk_(batch_x))
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(batch_x)
            self.loss_curve_.append(epoch_loss / max(1, n_samples))
            if self.verbose and (epoch + 1) % 20 == 0:
                print(
                    f"  EEGNet epoch {epoch + 1}/{self.epochs} "
                    f"loss={self.loss_curve_[-1]:.4f}"
                )

        self.trunk_.eval()
        self.n_features_out_ = int(self.feature_dim)
        return self

    def transform(self, X):
        if not hasattr(self, "trunk_"):
            raise RuntimeError("EEGNetFeatureExtractor must be fitted first")
        Xs = self._prepare(X)
        self.trunk_.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(Xs, dtype=torch.float32, device=self.device)
            tensor = tensor.unsqueeze(1)
            features = self.trunk_(tensor).cpu().numpy()
        return features.astype(np.float32, copy=False)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)
