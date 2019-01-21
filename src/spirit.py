import numpy as np


class Spirit():
    def __init__(self, n_features, n_hidden, d, lambda_):
        self.n_hidden = n_hidden
        self.W = np.eye(N=n_hidden, M=n_features, dtype=np.float32)
        self.ds = np.empty((n_hidden,), dtype=np.float32)
        self.ds.fill(np.float32(d))
        self.lambda_ = np.float32(lambda_)
        self.y = np.zeros((n_hidden,), dtype=np.float32)

    def partial_fit(self, x):
        lambda_ = self.lambda_
        xi = x
        for i in range(self.n_hidden):
            yi = self.W.take(i, axis=0).dot(xi)
            self.ds[i] = lambda_ * self.ds[i] + yi ** 2
            ei = xi - yi * self.W.take(i, axis=0)
            self.W[i, :] += 1 / self.ds[i] * yi * ei

            xi -= yi * self.W.take(i, axis=0)
            self.y[i] = yi
        return self.y

    def fit_transform(self, x):
        y = self.partial_fit(x)
        return y.ravel()


if __name__ == "__main__":
    N = 100000
    H, M = 100, 10
    X = np.random.randn(N, H)
    Y = np.empty((N, M), dtype=np.float32)
    spirit = Spirit(n_features=H, n_hidden=M, d=0.5, lambda_=1)
    for i, x in enumerate(X):
        Y[i, :] = spirit.fit_transform(x)

    YY = Y.T.dot(Y)
    YY
