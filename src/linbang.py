from __future__ import annotations

import pickle

import numpy as np
from scipy import optimize

from .utils import Rows, Row, Array, Float32


class LogisticBang:
    def __init__(self, bit_precision: int, init_reg: float) -> None:
        self.bit_precision = bit_precision
        feature_range = 2 ** bit_precision
        self._theta = np.zeros((feature_range + 1,), dtype=np.float32)
        self._dtheta = np.zeros((feature_range + 1,), dtype=np.float32)
        self._iHessian = np.zeros((feature_range + 1,), dtype=np.float32)
        self._iHessian.fill(init_reg)
        self._init_reg = init_reg
        self.example_counter = 0
        self.loss = 0
        self._prev_ids = []

    def predict(self, row: Array) -> Float32:
        label, weight, _, features = row
        theta = self.theta
        prediction = self._predict(theta, features)

        # logloss = self._logloss(label, prediction, weight)
        return prediction#, logloss

    def sample_predict(self, row: Array) -> Float32:
        label, weight, _, features = row
        ids = features['id']
        values = features['value']
        theta = self.theta
        iHessian = self.iHessian

        sample = np.random.randn(values.shape[0])
        sample_theta = sample * iHessian[ids] + theta[ids]
        x = sample_theta.T.dot(values)
        prediction = self._sigmoid(x)

        # logloss = self._logloss(label, prediction, weight)
        return prediction#, logloss

    def _predict(self, theta: Array, features: Array) -> Float32:
        ids = features['id']
        values = features['value']
        x = theta[ids].T.dot(values)
        prediction = self._sigmoid(x)
        return prediction

    def fit(self, rows: Rows) -> LogisticBang:
        for row in rows:
            self.partial_fit(row)
        return self

    def partial_fit(self, row: Row) -> LogisticBang:
        self._partial_fit(row)
        return self

    def _partial_fit(self, row):
        # if self.example_counter == 0:
        #     self._mode(row)
        # else:
        #     self._incremental_laplace_approx(row)
        self._incremental_laplace_approx(row)

    def _incremental_laplace_approx(self, row):
        self.example_counter += 1
        label, weight, _, features = row

        prediction = self._predict(self.theta, features)

        self._update_theta(prediction, label, weight, features)
        self._update_loss(prediction, label, weight)

    def _mode(self, row):
        label, weight, _, features = row
        prev_theta = self.theta

        theta, loss, _ = optimize.fmin_l_bfgs_b(self._compute_loss, fprime=self._compute_dtheta, x0=prev_theta,
                                                args=(label, weight, features))

        self.theta = theta
        self.loss += loss

    def _update_theta(self, prediction: Array, label: Float32, weight: Float32, features: Array):
        self._update_iHessian(prediction=prediction,
                              label=label,
                              features=features,
                              weight=weight)

        self._update_dtheta(prediction=prediction,
                            label=label,
                            features=features,
                            weight=weight)
        dtheta = self.dtheta
        iHessian = self.iHessian

        ids = features['id']
        # self.theta[ids] -= dtheta[ids]
        self.theta[ids] -= iHessian[ids] * dtheta[ids]

    def _compute_dtheta(self, theta: Array, label: Float32, weight: Float32, features: Array) -> Array:
        self._update_dtheta(theta, label, features, weight)
        return self.dtheta

    def _update_dtheta(self, prediction: Array, label: Float32, features: Array, weight: Float32) -> None:
        self.dtheta[self._prev_ids] = 0
        ids = features['id']
        values = features['value']

        iHessian = self.iHessian  # TODO check this update with reg
        dtheta = weight * (prediction - label) * values
        self.dtheta[ids] = dtheta
        self._prev_ids = ids.copy()

    def _update_iHessian(self, prediction: Array, label: Float32, features: Array, weight: Float32) -> None:
        ids = features['id']
        values = features['value']

        v = weight * prediction * (1 - prediction) * values ** 2
        # v = weight * (prediction - label) * values

        iHessian = self.iHessian.take(ids)
        norm = (1 + v * iHessian * v)
        iHvviH = iHessian * v * v * iHessian
        self.iHessian[ids] -= iHvviH/norm

    def _update_loss(self, prediction: Array, label: Float32, weight: Float32):
        logloss = self._logloss(label, prediction, weight)
        self.loss += logloss

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        with open(path, "rb") as f:
            attrs = pickle.load(f)
        self.__dict__.update(attrs)

    @staticmethod
    def _logloss(y: Float32, p: Float32, weight: Float32, eps=1e-15):
        p = np.clip(p, eps, 1 - eps)
        return -np.log(p) * weight if y else - np.log(1 - p) * weight

    @property
    def theta(self) -> Array:
        return self._theta

    @theta.setter
    def theta(self, values: Array) -> None:
        self._theta = values

    @property
    def dtheta(self) -> Array:
        return self._dtheta

    @dtheta.setter
    def dtheta(self, values: Array) -> None:
        self._dtheta = values

    @property
    def iHessian(self) -> Array:
        return self._iHessian

    @staticmethod
    def _sigmoid(x: Float32) -> Float32:
        return 1 / (1 + np.exp(-x))

    @property
    def coef_(self) -> Array:
        return self.theta[:-1]

    @property
    def intercept_(self) -> Float32:
        return self.theta[-1]

    @property
    def average_loss(self) -> float:
        return self.loss / max(self.example_counter, 1)
