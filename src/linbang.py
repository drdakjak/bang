from __future__ import annotations

import pickle

import numpy as np
from scipy import optimize

from .spirit import Transformer
from .utils import Rows, Row, Array, Float32


class LogisticBang:
    def __init__(self, bit_precision: int, init_reg: float, transformer=Transformer(), eps: float = 1e-15) -> None:
        self.bit_precision = bit_precision
        feature_range = 2 ** bit_precision
        self._eps = eps
        self._theta = np.zeros((feature_range + 1,), dtype=np.float32)
        self._dtheta = np.zeros((feature_range + 1,), dtype=np.float32)
        self._iHessian = np.zeros((feature_range + 1,), dtype=np.float32)
        self._iHessian.fill(1 / init_reg)
        self._init_reg = init_reg
        self.example_counter = 0
        self.loss = 0
        self._prev_ids = []
        self.transformer = transformer
        np.seterr(divide="raise")

    def predict(self, row: Array) -> Float32:
        label, weight, _, features = row
        features = self.transformer.fit_transform(features)

        theta = self.theta
        prediction = self._predict(theta, features)

        # logloss = self._logloss(label, prediction, weight)
        return prediction  # , logloss

    def sample_predict(self, row: Array) -> Float32:
        label, weight, _, features = row
        features = self.transformer.fit_transform(features)

        ids = features['id']
        values = features['value']
        theta = self.theta
        iHessian = self.iHessian

        sample = np.random.randn(values.shape[0])
        sample_theta = sample * iHessian[ids] + theta[ids]
        x = sample_theta.T.dot(values)
        prediction = self._sigmoid(x)

        # logloss = self._logloss(label, prediction, weight)
        return prediction  # , logloss

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
        features = self.transformer.fit_transform(features)

        prediction = self._predict(self.theta, features)

        self._update_theta(prediction, label, weight, features)
        self._update_loss(prediction, label, weight)

    def _mode(self, row):
        label, weight, _, features = row
        features = self.transformer.fit_transform(features)

        prev_theta = self.theta

        theta, loss, _ = optimize.fmin_l_bfgs_b(self._compute_loss, fprime=self._compute_dtheta, x0=prev_theta,
                                                args=(label, weight, features))

        self.theta = theta
        self.loss += loss

    def _update_theta(self, prediction: Array, label: Float32, weight: Float32, features: Array):
        # update based on https://arxiv.org/abs/1605.05697 (https://www.diigo.com/item/pdf/65klm/dhwi)
        self._prev_theta = self.theta.copy()
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
        self.theta[ids] += iHessian[ids] * dtheta[ids]

    def _compute_dtheta(self, theta: Array, label: Float32, weight: Float32, features: Array) -> Array:
        self._update_dtheta(theta, label, features, weight)
        return self.dtheta

    def _update_iHessian(self, prediction: Array, label: Float32, features: Array, weight: Float32) -> None:
        ids = features['id']
        values = features['value']
        iHessian = self.iHessian.take(ids)

        variance = weight * prediction * (1 - prediction)

        norm = variance / (1 + variance * values * iHessian * values)
        iHvviH = iHessian * values * values * iHessian
        update = norm * iHvviH
        iHessian_updated = iHessian - update
        iHessian_updated = np.clip(iHessian_updated, a_min=self._eps, a_max=np.inf)
        self.iHessian[ids] = iHessian_updated

    def _update_dtheta(self, prediction: Array, label: Float32, features: Array, weight: Float32) -> None:
        self.dtheta[self._prev_ids] = 0
        ids = features['id']
        values = features['value']

        # theta = self.theta.take(ids)
        # prev_theta = self._prev_theta.take(ids)
        # reg = 1 / self.iHessian.take(ids)
        # estimate = theta - prev_theta
        # dtheta = reg * estimate + weight * (prediction - label) * values

        dtheta = weight * (label - prediction) * values

        self.dtheta[ids] = dtheta
        self._prev_ids = ids

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

    def _logloss(self, y: Float32, p: Float32, weight: Float32):
        p = np.clip(p, self._eps, 1 - self._eps)
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
