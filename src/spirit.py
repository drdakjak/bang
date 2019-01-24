import numpy as np
from tqdm import tqdm
from utils import Array, Float32


class Spirit():
    def __init__(self, n_features: int, k_hidden: int, d: float, lambda_: Float32, dynamic_k: bool) -> None:
        self.k_hidden = k_hidden
        self.dynamic_k = dynamic_k
        self.n_features = n_features
        self.W = np.eye(N=n_features, M=k_hidden, dtype=np.float32)
        self.ds = np.empty((k_hidden,), dtype=np.float32)
        self.ds.fill(np.float32(d))
        self.d = d
        self.lambda_ = np.float32(lambda_)
        self.y = np.zeros((k_hidden,), dtype=np.float32)

        self.E = 0
        self.E_hat = np.zeros(k_hidden)

        self.energy_lower_bound = 0.95
        self.energy_upper_bound = 0.98
        self.n_iter = 0

    def partial_fit(self, x: Array) -> Array:
        self.n_iter += 1

        lambda_ = self.lambda_
        xi = x
        for i in range(self.k_hidden):
            yi = self.W.take(i, axis=1).T.dot(xi)
            ei = xi - yi * self.W.take(i, axis=1)
            self.ds[i] = lambda_ * self.ds[i] + yi ** 2
            self.W[:, i] += 1 / self.ds[i] * yi * ei
            xi -= yi * self.W.take(i, axis=1)

            self.y[i] = yi

        return self.y

    def fit_transform(self, x: Array) -> Array:
        self.E = self.compute_energy(x)
        y = self.partial_fit(x)
        E_hat = self.compute_reconstruction_energy(y)
        if self.dynamic_k:
            self.update_k_hidden(self.E, E_hat)
        return y

    def compute_energy(self, x):
        E = (self.n_iter - 1) * self.E + np.linalg.norm(x)
        return E / self.n_iter

    def compute_reconstruction_energy(self, ys):
        for i, y in enumerate(ys):
            E_hat = (self.n_iter - 1) * self.E_hat[i] + y ** 2
            self.E_hat[i] = E_hat / self.n_iter
        return self.E_hat.sum()

    def update_k_hidden(self, E, E_hat):
        if E_hat < self.energy_lower_bound * E:
            eye = np.eye(N=self.n_features, M=1, k=-self.k_hidden-1)
            self.W = np.append(self.W, eye, axis=1)
            self.ds = np.append(self.ds, [self.d])
            self.y = np.append(self.y, [0])

            self.k_hidden += 1
            self.E_hat = np.append(self.E_hat, [0])
            self.E_hat[-1] = 0

        elif E_hat >= self.energy_upper_bound * E:
            np.delete(self.W, -1, axis=1)
            np.delete(self.E_hat, -1)
            np.delete(self.ds, -1)
            np.delete(self.y, -1)
            self.k_hidden -= 1


if __name__ == "__main__":
    N = 10000
    F, K = 1000, 1
    X = np.random.randn(N, F).astype(np.float32)

    spirit = Spirit(n_features=F, k_hidden=2, d=0.0005, lambda_=1, dynamic_k=False)

    Y = []
    for i, x in tqdm(enumerate(X)):
        y = spirit.fit_transform(x)
        Y.append(y.tolist())

    Y = np.array(Y)
    YY = Y.T.dot(Y)
    YY
