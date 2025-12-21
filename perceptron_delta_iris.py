import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DeltaPerceptronBinary:
    def __init__(self, lr=0.01, epochs=50, shuffle=True, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.w = None
        self.b = 0.0

    def fit(self, x, y):
        n, d = x.shape
        self.w = np.zeros(d, dtype=float)
        self.b = 0.0

        for ep in range(self.epochs):
            idx = np.arange(n)
            if self.shuffle:
                self.rng.shuffle(idx)

            mse_acc = 0.0

            for i in idx:
                xi = x[i]
                yi = y[i]

                a = float(xi @ self.w + self.b)
                e = yi - a

                self.w += self.lr * e * xi
                self.b += self.lr * e

                mse_acc += e * e

            mse = mse_acc / n
            if (ep + 1) % 10 == 0:
                print(f"[Binary] epoch={ep+1:3d} MSE={mse:.6f}")

        return self

    def decision_function(self, x):
        return x @ self.w + self.b

    def predict(self, x):
        scores = self.decision_function(x)
        return np.where(scores >= 0, 1, -1)


class DeltaPerceptronOVA:
    """
    Multiclasse via One-vs-All usando DeltaPerceptronBinary.
    Treina K modelos binários:
      classe k = +1, resto = -1
    Prediz pelo maior score (w_k^T x + b_k).
    """
    def __init__(self, lr=0.01, epochs=50, shuffle=True, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.shuffle = shuffle
        self.seed = seed
        self.models = []
        self.classes_ = None

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        self.models = []

        for k, cls in enumerate(self.classes_):
            y_bin = np.where(y == cls, 1, -1)
            model = DeltaPerceptronBinary(
                lr=self.lr, epochs=self.epochs, shuffle=self.shuffle, seed=self.seed + k
            )
            print(f"\nTreinando OvA para classe {cls} vs resto...")
            model.fit(x, y_bin)
            self.models.append(model)

        return self

    def predict(self, x):
        scores = np.column_stack([m.decision_function(x) for m in self.models])
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]


def accuracy(y_pred, y_true):
    return float(np.mean(y_pred == y_true))


def main():
    iris = load_iris()
    x = iris.data
    y = iris.target  # 0,1,2

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 1) Binário: setosa (0) vs resto
    y_train_bin = np.where(y_train == 0, 1, -1)
    y_test_bin = np.where(y_test == 0, 1, -1)

    print("=== Perceptron (Regra Delta) - Binário: setosa vs resto ===")
    bin_model = DeltaPerceptronBinary(lr=0.05, epochs=60)
    bin_model.fit(x_train, y_train_bin)

    pred_bin = bin_model.predict(x_test)
    print("Acurácia binária:", accuracy(pred_bin, y_test_bin))

    # 2) Multiclasse: One-vs-All (3 perceptrons)
    print("\n=== Perceptron (Regra Delta) - Multiclasse (OvA) ===")
    ova_model = DeltaPerceptronOVA(lr=0.05, epochs=60)
    ova_model.fit(x_train, y_train)

    pred_ova = ova_model.predict(x_test)
    print("Acurácia multiclasse (OvA):", accuracy(pred_ova, y_test))


if __name__ == "__main__":
    main()
