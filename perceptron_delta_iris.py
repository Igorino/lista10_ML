import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Perceptron simples treinado com REGRA DELTA (LMS)
# - Modelo linear: a = w^T x + b
# - Atualização: w <- w + lr * (y - a) * x
# - Target esperado: y ∈ { -1, +1 }
class DeltaPerceptronBinary:
    def __init__(self, lr=0.01, epochs=50, shuffle=True, seed=42):
        # taxa de aprendizado
        self.lr = lr
        # número de épocas
        self.epochs = epochs
        # se True, embaralha os dados a cada época
        self.shuffle = shuffle
        # gerador de números aleatórios (reprodutibilidade)
        self.rng = np.random.default_rng(seed)
        # pesos (inicializados no fit)
        self.w = None
        # bias (intercepto)
        self.b = 0.0

    def fit(self, x, y):
        # x: matriz (n_amostras x n_features)
        # y: vetor de rótulos {-1, +1}

        n, d = x.shape
        # inicializa pesos com zero
        self.w = np.zeros(d, dtype=float)
        self.b = 0.0

        # loop de treinamento
        for ep in range(self.epochs):
            # índices das amostras
            idx = np.arange(n)
            if self.shuffle:
                # embaralha para evitar viés de ordem
                self.rng.shuffle(idx)

            mse_acc = 0.0

            # percorre as amostras (SGD)
            for i in idx:
                xi = x[i]
                yi = y[i]

                # saída linear (antes do sinal)
                a = float(xi @ self.w + self.b)
                # erro da regra delta
                e = yi - a

                # atualização dos parâmetros
                self.w += self.lr * e * xi
                self.b += self.lr * e

                # acumula erro quadrático (só para monitorar)
                mse_acc += e * e

            mse = mse_acc / n
            # imprime o MSE a cada 10 épocas
            if (ep + 1) % 10 == 0:
                print(f"[Binary] epoch={ep+1:3d} MSE={mse:.6f}")

        return self

    def decision_function(self, x):
        # retorna o score linear (sem limiar)
        return x @ self.w + self.b

    def predict(self, x):
        # aplica o sinal ao score linear
        scores = self.decision_function(x)
        return np.where(scores >= 0, 1, -1)


# Perceptron Multiclasse via One-vs-All
# - Treina um perceptron binário por classe
# - Classe k = +1, resto = -1
# - Predição: classe com maior score linear
class DeltaPerceptronOVA:
    def __init__(self, lr=0.01, epochs=50, shuffle=True, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.shuffle = shuffle
        self.seed = seed
        # lista de modelos binários
        self.models = []
        # rótulos das classes originais
        self.classes_ = None

    def fit(self, x, y):
        # guarda quais são as classes (ex: 0,1,2 no Iris)
        self.classes_ = np.unique(y)
        self.models = []

        # treina um classificador por classe
        for k, cls in enumerate(self.classes_):
            # classe atual vira +1, resto vira -1
            y_bin = np.where(y == cls, 1, -1)

            model = DeltaPerceptronBinary(
                lr=self.lr,
                epochs=self.epochs,
                shuffle=self.shuffle,
                seed=self.seed + k
            )

            print(f"\nTreinando OvA para classe {cls} vs resto...")
            model.fit(x, y_bin)
            self.models.append(model)

        return self

    def predict(self, x):
        # empilha os scores de todos os classificadores
        # shape: (n_amostras x n_classes)
        scores = np.column_stack(
            [m.decision_function(x) for m in self.models]
        )

        # escolhe a classe com maior score
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]


# Métrica simples de avaliação
def accuracy(y_pred, y_true):
    return float(np.mean(y_pred == y_true))


# Execução principal
def main():
    # carrega o dataset Iris
    iris = load_iris()
    x = iris.data
    y = iris.target   # classes 0,1,2

    # split treino / teste com estratificação
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    # padronização (importante para a regra delta)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 1) Classificação binária: setosa (classe 0) vs resto
    y_train_bin = np.where(y_train == 0, 1, -1)
    y_test_bin = np.where(y_test == 0, 1, -1)

    print("=== Perceptron (Regra Delta) - Binário: setosa vs resto ===")
    bin_model = DeltaPerceptronBinary(lr=0.05, epochs=60)
    bin_model.fit(x_train, y_train_bin)

    pred_bin = bin_model.predict(x_test)
    print("Acurácia binária:", accuracy(pred_bin, y_test_bin))

    # 2) Classificação multiclasse: One-vs-All
    print("\n=== Perceptron (Regra Delta) - Multiclasse (OvA) ===")
    ova_model = DeltaPerceptronOVA(lr=0.05, epochs=60)
    ova_model.fit(x_train, y_train)

    pred_ova = ova_model.predict(x_test)
    print("Acurácia multiclasse (OvA):", accuracy(pred_ova, y_test))


# ponto de entrada do script
if __name__ == "__main__":
    main()
