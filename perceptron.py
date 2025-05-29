import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

alpha = 0.1
epochs = 150

X_train_df = pd.read_csv('train_set_normalized.csv')
X_test_df = pd.read_csv('test_set_normalized.csv')
y_train_df = pd.read_csv('train_set_y.csv')
y_test_df = pd.read_csv('test_set_y.csv')

X_train = X_train_df.to_numpy()
X_test = X_test_df.to_numpy()
y_train = y_train_df.to_numpy().ravel()
y_test = y_test_df.to_numpy().ravel()


class Perceptron:
    def __init__(self, n_features):
        self.w = np.random.randn(n_features, 1) * 0.01
        self.b = np.zeros((1,))

    def __call__(self, x):
        z = np.dot(x, self.w) + self.b
        return 1 / (1 + np.exp(-np.clip(z, -500, 500))).squeeze()


def cross_entropy(y, y_pred):
    y_pred = np.clip(y_pred, 1e-5, 1 - 1e-5)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def w_grad(x, y, y_pred):
    return np.dot(x.T, (y_pred - y)) / len(y)


def b_grad(y, y_pred):
    return np.mean(y_pred - y)


def get_accuracy(y_true, y_pred):
    predicted = (y_pred > 0.5).astype(int)
    return 100 * np.mean(predicted == y_true)


model = Perceptron(n_features=X_train.shape[1])
losses = []
test_losses = []

for epoch in range(epochs):
    y_pred = model(X_train)
    loss = cross_entropy(y_train, y_pred)

    grad_w = w_grad(X_train, y_train, y_pred)
    grad_b = b_grad(y_train, y_pred)

    model.w -= alpha * grad_w.reshape(-1, 1)
    model.b -= alpha * grad_b

    losses.append(loss)
    test_losses.append(cross_entropy(y_test, model(X_test)))

    if epoch % 50 == 0:
        train_accuracy = get_accuracy(y_train, y_pred)
        test_accuracy = get_accuracy(y_test, model(X_test))
        print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Accuracy: {test_accuracy:.2f}%")

y_test_pred = model(X_test)
test_loss = cross_entropy(y_test, y_test_pred)
test_accuracy = get_accuracy(y_test, y_test_pred)
print(f"\nFinal Test Loss: {test_loss:.6f}")
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig('training_loss_plot.png')
plt.show()

# Сохранение
np.save('perceptron_weights.npy', model.w)
np.save('perceptron_bias.npy', model.b)
print("Weights and bias saved")
print("Plot saved to 'training_loss_plot.png'")