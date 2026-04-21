
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# PARTIE 1 : HYPER-PARAMÈTRES D'OPTIMISATION

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', optimizer='sgd',
                 learning_rate=0.01, lambda_reg=0.01, reg_type=None,
                 dropout_rate=0.0, batch_size=32, num_epochs=100):
       
        self.layer_sizes   = layer_sizes
        self.activation    = activation
        self.optimizer     = optimizer
        self.lr            = learning_rate
        self.lambda_reg    = lambda_reg
        self.reg_type      = reg_type
        self.dropout_rate  = dropout_rate
        self.batch_size    = batch_size
        self.num_epochs    = num_epochs
        self.history       = {'train_loss': [], 'val_loss': []}

        self.params = {}
        self._init_weights()

        self.m, self.v = {}, {}
        self.t = 0
        if optimizer == 'adam':
            for key in self.params:
                self.m[key] = np.zeros_like(self.params[key])
                self.v[key] = np.zeros_like(self.params[key])

    def _init_weights(self):
        np.random.seed(42)
        L = len(self.layer_sizes)
        for l in range(1, L):
            n_in  = self.layer_sizes[l - 1]
            n_out = self.layer_sizes[l]
            if self.activation == 'relu':
                scale = np.sqrt(2.0 / n_in)            
            else:
                scale = np.sqrt(1.0 / n_in)            
            self.params[f'W{l}'] = np.random.randn(n_in, n_out) * scale
            self.params[f'b{l}'] = np.zeros((1, n_out))

    # Fonctions d'activation
    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_deriv(self, Z):
        return (Z > 0).astype(float)

    def _sigmoid(self, Z):
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))

    def _sigmoid_deriv(self, Z):
        s = self._sigmoid(Z)
        return s * (1 - s)

    def _tanh(self, Z):
        return np.tanh(Z)

    def _tanh_deriv(self, Z):
        return 1.0 - np.tanh(Z) ** 2

    def _activate(self, Z, deriv=False):
        funcs = {
            'relu':    (self._relu,    self._relu_deriv),
            'sigmoid': (self._sigmoid, self._sigmoid_deriv),
            'tanh':    (self._tanh,    self._tanh_deriv),
        }
        f, df = funcs[self.activation]
        return df(Z) if deriv else f(Z)

    def _softmax(self, Z):
        expZ = np.exp(Z - Z.max(axis=1, keepdims=True))
        return expZ / expZ.sum(axis=1, keepdims=True)

    #Forward Pass
    def _forward(self, X, training=True):
        cache = {'A0': X}
        L     = len(self.layer_sizes) - 1
        A     = X
        for l in range(1, L + 1):
            Z = A @ self.params[f'W{l}'] + self.params[f'b{l}']
            cache[f'Z{l}'] = Z

            if l == L:                          
                A = self._softmax(Z)
            else:                               
                A = self._activate(Z)
                # Dropout sur les couches cachées uniquement
                if training and self.dropout_rate > 0:
                    mask = (np.random.rand(*A.shape) > self.dropout_rate)
                    A   *= mask / (1 - self.dropout_rate)   # Inverted dropout
                    cache[f'mask{l}'] = mask
            cache[f'A{l}'] = A
        return A, cache
        
    # Calcul de Loss avec régularisation
    def _compute_loss(self, Y_hat, Y):
        m = Y.shape[0]
        log_likelihood = -np.log(Y_hat[range(m), Y] + 1e-9)
        loss = np.mean(log_likelihood)
        # terme de régularisation
        L = len(self.layer_sizes) - 1
        reg_term = 0.0
        if self.reg_type == 'L2':
            for l in range(1, L + 1):
                reg_term += np.sum(self.params[f'W{l}'] ** 2)
            loss += (self.lambda_reg / (2 * m)) * reg_term
        elif self.reg_type == 'L1':
            for l in range(1, L + 1):
                reg_term += np.sum(np.abs(self.params[f'W{l}']))
            loss += (self.lambda_reg / m) * reg_term
        return loss

    # Rétropropagation 
    def _backward(self, X, Y, cache):
        grads = {}
        m     = X.shape[0]
        L     = len(self.layer_sizes) - 1
        dA = cache[f'A{L}'].copy()
        dA[range(m), Y] -= 1
        dA /= m
        for l in reversed(range(1, L + 1)):
            A_prev = cache[f'A{l-1}']
            Z      = cache[f'Z{l}']

            dW = A_prev.T @ dA
            db = np.sum(dA, axis=0, keepdims=True)
            # Gradient de régularisation
            if self.reg_type == 'L2':
                dW += (self.lambda_reg / m) * self.params[f'W{l}']
            elif self.reg_type == 'L1':
                dW += (self.lambda_reg / m) * np.sign(self.params[f'W{l}'])

            grads[f'dW{l}'] = dW
            grads[f'db{l}'] = db

            if l > 1:
                Z_prev = cache[f'Z{l-1}']
                dA = dA @ self.params[f'W{l}'].T * self._activate(Z_prev, deriv=True)
                # Dropout backward
                if self.dropout_rate > 0 and f'mask{l-1}' in cache:
                    dA *= cache[f'mask{l-1}'] / (1 - self.dropout_rate)
        return grads

    # Mise à jour des paramètres (SGD ou Adam)
    def _update_params(self, grads):
        L = len(self.layer_sizes) - 1
        if self.optimizer == 'adam':
            self.t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for l in range(1, L + 1):
                for p in ['W', 'b']:
                    key = f'{p}{l}'
                    g   = grads[f'd{key}']
                    self.m[key] = beta1 * self.m[key] + (1 - beta1) * g
                    self.v[key] = beta2 * self.v[key] + (1 - beta2) * g ** 2
                    m_hat = self.m[key] / (1 - beta1 ** self.t)
                    v_hat = self.v[key] / (1 - beta2 ** self.t)
                    self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        else:   
            for l in range(1, L + 1):
                self.params[f'W{l}'] -= self.lr * grads[f'dW{l}']
                self.params[f'b{l}'] -= self.lr * grads[f'db{l}']
 
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        m = X_train.shape[0]
        for epoch in range(self.num_epochs):
            indices = np.random.permutation(m)
            X_sh, y_sh = X_train[indices], y_train[indices]
 
            for start in range(0, m, self.batch_size):
                Xb = X_sh[start:start + self.batch_size]
                yb = y_sh[start:start + self.batch_size]
                Y_hat, cache = self._forward(Xb, training=True)
                grads = self._backward(Xb, yb, cache)
                self._update_params(grads)

            Y_hat_train, _ = self._forward(X_train, training=False)
            train_loss      = self._compute_loss(Y_hat_train, y_train)
            self.history['train_loss'].append(train_loss)
            if X_val is not None:
                Y_hat_val, _ = self._forward(X_val, training=False)
                val_loss      = self._compute_loss(Y_hat_val, y_val)
                self.history['val_loss'].append(val_loss)

    def predict(self, X):
        Y_hat, _ = self._forward(X, training=False)
        return np.argmax(Y_hat, axis=1)


# PARTIE 2 : IMPLÉMENTATION DES RÉGULARISATIONS

def l2_regularization_demo(X_train, y_train, X_val, y_val, lambdas):
    results = {}
    for lam in lambdas:
        model = NeuralNetwork(
            layer_sizes=[X_train.shape[1], 64, 32, 2],
            activation='relu', optimizer='adam',
            learning_rate=0.001, lambda_reg=lam,
            reg_type='L2', dropout_rate=0.0,
            batch_size=32, num_epochs=100
        )
        model.fit(X_train, y_train, X_val, y_val)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc   = accuracy_score(y_val,   model.predict(X_val))
        results[lam] = {
            'train_acc': train_acc, 'val_acc': val_acc,
            'history': model.history}
        print(f"  λ={lam:.4f} -> Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")
    return results
def l1_regularization_demo(X_train, y_train, X_val, y_val, lambdas):
    results = {}
    for lam in lambdas:
        model = NeuralNetwork(
            layer_sizes=[X_train.shape[1], 64, 32, 2],
            activation='relu', optimizer='adam',
            learning_rate=0.001, lambda_reg=lam,
            reg_type='L1', dropout_rate=0.0,
            batch_size=32, num_epochs=100
        )
        model.fit(X_train, y_train, X_val, y_val)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc   = accuracy_score(y_val,   model.predict(X_val))
        results[lam] = {
            'train_acc': train_acc, 'val_acc': val_acc,
            'history': model.history}
        print(f"  λ={lam:.4f} -> Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")
    return results

def dropout_demo(X_train, y_train, X_val, y_val, dropout_rates):
    results = {}
    for rate in dropout_rates:
        model = NeuralNetwork(
            layer_sizes=[X_train.shape[1], 128, 64, 2],
            activation='relu', optimizer='adam',
            learning_rate=0.001, lambda_reg=0.0,
            reg_type=None, dropout_rate=rate,
            batch_size=32, num_epochs=100
        )
        model.fit(X_train, y_train, X_val, y_val)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc   = accuracy_score(y_val,   model.predict(X_val))
        results[rate] = {
            'train_acc': train_acc, 'val_acc': val_acc,
            'history': model.history }
        print(f"  Dropout={rate:.1f} → Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")
    return results

# COMPARAISON DES OPTIMISEURS

def compare_optimizers(X_train, y_train, X_val, y_val):
    results = {}
    configs = [
        ('SGD  lr=0.01', 'sgd',  0.01),
        ('SGD  lr=0.1',  'sgd',  0.1),
        ('Adam lr=0.001','adam', 0.001),
    ]
    for name, opt, lr in configs:
        model = NeuralNetwork(
            layer_sizes=[X_train.shape[1], 64, 32, 2],activation='relu', optimizer=opt,
            learning_rate=lr, lambda_reg=0.0,reg_type=None, dropout_rate=0.0, batch_size=32, num_epochs=100 )
        model.fit(X_train, y_train, X_val, y_val)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        results[name] = {'history': model.history, 'val_acc': val_acc}
        print(f"  {name} → Val Acc={val_acc:.4f}")
    return results
 
#VISUALISATION

def plot_results(l2_results, l1_results, dropout_results, opt_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Régularisation et Hyper-paramètres – Résultats', fontsize=14, fontweight='bold')

    # Graph1 : L2 Regularisation  
    ax = axes[0, 0]
    lambdas = list(l2_results.keys())
    train_accs = [l2_results[l]['train_acc'] for l in lambdas]
    val_accs   = [l2_results[l]['val_acc']   for l in lambdas]
    ax.plot(range(len(lambdas)), train_accs, 'o-b', label='Train Accuracy')
    ax.plot(range(len(lambdas)), val_accs,   's-r', label='Val Accuracy')
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f'λ={l}' for l in lambdas], fontsize=8)
    ax.set_title('Régularisation L2 : effet de λ')
    ax.set_ylabel('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Graph2 : L1 vs L2 convergence
    ax = axes[0, 1]
    lam = lambdas[1]  
    ax.plot(l2_results[lam]['history']['val_loss'], label=f'L2 λ={lam}', color='blue')
    ax.plot(l1_results[lam]['history']['val_loss'], label=f'L1 λ={lam}', color='orange', linestyle='--')
    ax.set_title('Convergence : L1 vs L2')
    ax.set_xlabel('Époque'); ax.set_ylabel('Loss (validation)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Graph3 : Dropout
    ax = axes[1, 0]
    rates = list(dropout_results.keys())
    d_train = [dropout_results[r]['train_acc'] for r in rates]
    d_val   = [dropout_results[r]['val_acc']   for r in rates]
    x = range(len(rates))
    ax.bar([i - 0.2 for i in x], d_train, 0.4, label='Train', color='steelblue')
    ax.bar([i + 0.2 for i in x], d_val,   0.4, label='Val',   color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([f'p={r}' for r in rates])
    ax.set_title('Dropout : effet du taux p')
    ax.set_ylabel('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # Graph4 : Comparaison optimiseurs
    ax = axes[1, 1]
    for name, res in opt_results.items():
        ax.plot(res['history']['val_loss'], label=f"{name} ({res['val_acc']:.3f})")
    ax.set_title('Comparaison Optimiseurs (SGD vs Adam)')
    ax.set_xlabel('Époque'); ax.set_ylabel('Loss (validation)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
 
 
if __name__ == '__main__':
    print("=" * 60)
    print("  Chargement et préparation des données")
    print("=" * 60)
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val,   X_test, y_val,   y_test  = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"  Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    print("\n[1] Régularisation L2")
    l2_res = l2_regularization_demo(X_train, y_train, X_val, y_val,
                                     lambdas=[0.0001, 0.001, 0.01, 0.1])

    print("\n[2] Régularisation L1")
    l1_res = l1_regularization_demo(X_train, y_train, X_val, y_val,
                                     lambdas=[0.0001, 0.001, 0.01, 0.1])

    print("\n[3] Dropout")
    do_res = dropout_demo(X_train, y_train, X_val, y_val,
                           dropout_rates=[0.0, 0.2, 0.4, 0.6])

    print("\n[4] Comparaison des Optimiseurs")
    op_res = compare_optimizers(X_train, y_train, X_val, y_val)

    print("\n[5] Génération des graphiques")
    plot_results(l2_res, l1_res, do_res, op_res)
 
