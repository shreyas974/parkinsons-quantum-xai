"""Hybrid quantum classifiers with a scikit-learn interface.

VQC: angle embedding + trainable entangling layers, <Z> on qubit 0 as the score.
QuantumKernelSVM: SVC on the fidelity kernel |<0|U(x')^dagger U(x)|0>|^2.

Inputs are expected already scaled to [0, pi] (features.angle_scaler).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC


def _device(n_qubits: int, name: str):
    return qml.device(name, wires=n_qubits)


class _Picklable:
    """PennyLane circuits can't be pickled: drop them when saving, rebuild on load."""

    _circuit_attr = ""

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop(self._circuit_attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        n = self._fitted_qubits()
        if n is not None:
            setattr(self, self._circuit_attr, self._build(n))


class VQC(_Picklable, BaseEstimator, ClassifierMixin):
    _circuit_attr = "circuit_"

    def _fitted_qubits(self):
        return getattr(self, "n_qubits_", None)

    def __init__(
        self,
        n_layers: int = 3,
        epochs: int = 60,
        lr: float = 0.05,
        batch_size: int = 32,
        entangle: bool = True,
        device: str = "default.qubit",
        seed: int = 0,
    ):
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.entangle = entangle
        self.device = device
        self.seed = seed

    def _build(self, n_qubits: int):
        dev = _device(n_qubits, self.device)
        wires = range(n_qubits)
        entangle = self.entangle

        @qml.qnode(dev, interface="autograd")
        def circuit(weights, X):
            qml.AngleEmbedding(X, wires=wires, rotation="Y")
            if entangle:
                qml.StronglyEntanglingLayers(weights, wires=wires)
            else:
                # Same trainable rotations, no CNOTs: the "no entanglement" ablation.
                for layer in weights:
                    for w, (a, b, c) in zip(wires, layer):
                        qml.Rot(a, b, c, wires=w)
            return qml.expval(qml.PauliZ(0))

        return circuit

    def _proba(self, weights, bias, X):
        z = self.circuit_(weights, X)
        return pnp.clip((1 - z) / 2 + bias, 1e-6, 1 - 1e-6)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])
        self.n_qubits_ = X.shape[1]
        self.circuit_ = self._build(self.n_qubits_)
        rng = np.random.default_rng(self.seed)
        shape = qml.StronglyEntanglingLayers.shape(self.n_layers, self.n_qubits_)
        weights = pnp.array(rng.uniform(0, 2 * np.pi, shape), requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)
        opt = qml.AdamOptimizer(self.lr)
        self.loss_curve_ = []

        def loss(w, b, Xb, yb):
            p = self._proba(w, b, Xb)
            return -pnp.mean(yb * pnp.log(p) + (1 - yb) * pnp.log(1 - p))

        for _ in range(self.epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                idx = order[start : start + self.batch_size]
                Xb, yb = pnp.array(X[idx], requires_grad=False), pnp.array(y[idx], requires_grad=False)
                (weights, bias, _, _), cost = opt.step_and_cost(loss, weights, bias, Xb, yb)
            self.loss_curve_.append(float(cost))
        self.weights_, self.bias_ = weights, bias
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = np.asarray(self._proba(self.weights_, self.bias_, X), dtype=float).reshape(-1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def draw(self):
        """Matplotlib figure of the circuit, for the paper (figure F6)."""
        x = np.zeros(self.n_qubits_)
        return qml.draw_mpl(self.circuit_)(self.weights_, x)


class QuantumKernelSVM(_Picklable, BaseEstimator, ClassifierMixin):
    _circuit_attr = "kernel_"

    def _fitted_qubits(self):
        X = getattr(self, "X_train_", None)
        return None if X is None else X.shape[1]

    def __init__(self, C: float = 1.0, entangling_layer: bool = True, device: str = "default.qubit", seed: int = 0):
        self.C = C
        self.entangling_layer = entangling_layer
        self.device = device
        self.seed = seed

    def _build(self, n_qubits: int):
        dev = _device(n_qubits, self.device)
        wires = range(n_qubits)
        rng = np.random.default_rng(self.seed)
        # One fixed (untrained) entangling layer, so the kernel is not a plain product of cosines.
        fixed = rng.uniform(0, 2 * np.pi, qml.StronglyEntanglingLayers.shape(1, n_qubits))
        use_layer = self.entangling_layer

        def feature_map(x):
            qml.AngleEmbedding(x, wires=wires, rotation="Y")
            if use_layer:
                qml.StronglyEntanglingLayers(fixed, wires=wires)

        @qml.qnode(dev)
        def overlap(x1, x2):
            feature_map(x1)
            qml.adjoint(feature_map)(x2)
            return qml.probs(wires=wires)

        return lambda a, b: overlap(a, b)[0]

    def _gram(self, A, B):
        return np.asarray(qml.kernels.kernel_matrix(A, B, self.kernel_), dtype=float)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.array([0, 1])
        self.kernel_ = self._build(X.shape[1])
        self.X_train_ = X
        self.svc_ = SVC(kernel="precomputed", C=self.C, probability=True, class_weight="balanced",
                        random_state=self.seed)
        self.svc_.fit(self._gram(X, X), y)
        return self

    def predict_proba(self, X):
        return self.svc_.predict_proba(self._gram(np.asarray(X, dtype=float), self.X_train_))

    def predict(self, X):
        return self.svc_.predict(self._gram(np.asarray(X, dtype=float), self.X_train_))
