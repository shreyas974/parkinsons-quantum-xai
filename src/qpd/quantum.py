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
    noise = 0.0

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
        noise: float = 0.0,
    ):
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.entangle = entangle
        self.device = device
        self.seed = seed
        self.noise = noise

    def _build(self, n_qubits: int):
        dev = _device(n_qubits, "default.mixed" if self.noise else self.device)
        wires = range(n_qubits)
        entangle = self.entangle

        @qml.qnode(dev, interface="autograd")
        def circuit(weights, X):
            qml.AngleEmbedding(X, wires=wires, rotation="Y")
            for index, layer in enumerate(weights):
                if entangle:
                    distance = 1 + index % (n_qubits - 1) if n_qubits > 1 else 0
                    qml.StronglyEntanglingLayers(pnp.expand_dims(layer, 0), wires=wires, ranges=[distance])
                else:
                    for wire, (a, b, c) in zip(wires, layer):
                        qml.Rot(a, b, c, wires=wire)
                if self.noise:
                    for wire in wires:
                        qml.DepolarizingChannel(self.noise, wires=wire)
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
            return -pnp.mean(pnp.log(p) * yb + pnp.log1p(-p) * (1 - yb))

        for _ in range(self.epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                idx = order[start : start + self.batch_size]
                def batch_loss(w, b):
                    return loss(w, b, X[idx], y[idx])

                weights, bias = opt.step(batch_loss, weights, bias)
            self.loss_curve_.append(float(loss(weights, bias, X, y)))
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

    def __setstate__(self, state):
        super().__setstate__(state)
        if "states_train_" not in state and self._fitted_qubits() is not None:
            self.states_train_ = self._states(self.X_train_)

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
        fixed = rng.uniform(0, 2 * np.pi, qml.StronglyEntanglingLayers.shape(1, n_qubits))
        reupload = getattr(self, "reupload_", False)

        @qml.qnode(dev)
        def state(x):
            qml.AngleEmbedding(x, wires=wires, rotation="Y")
            if self.entangling_layer:
                qml.StronglyEntanglingLayers(fixed, wires=wires)
                if reupload:
                    qml.AngleEmbedding(x, wires=wires, rotation="Z")
            return qml.state()

        return state

    def _states(self, X):
        # Statevectors let the simulator reuse each feature map (Havlíček et al., 2019).
        return np.asarray([self.kernel_(row) for row in X])

    @staticmethod
    def _gram(A, B):
        return np.abs(A @ B.conj().T) ** 2

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.array([0, 1])
        self.reupload_ = True
        self.kernel_ = self._build(X.shape[1])
        self.X_train_ = X
        self.svc_ = SVC(kernel="precomputed", C=self.C, probability=True, class_weight="balanced",
                        random_state=self.seed)
        self.states_train_ = self._states(X)
        self.svc_.fit(self._gram(self.states_train_, self.states_train_), y)
        return self

    def _test_gram(self, X):
        return self._gram(self._states(np.asarray(X, dtype=float)), self.states_train_)

    def predict_proba(self, X):
        return self.svc_.predict_proba(self._test_gram(X))

    def predict(self, X):
        return self.svc_.predict(self._test_gram(X))
