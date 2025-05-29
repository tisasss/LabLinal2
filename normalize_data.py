import numpy as np
import pandas as pd

NUM_SAMPLES = 1_000_000
TRAIN_PART = 0.8

df = pd.read_csv('data.csv')

train_size = int(NUM_SAMPLES * TRAIN_PART)
train_df = df.sample(n=train_size, random_state=42)
test_df = df.drop(train_df.index)

X_train = train_df.drop(columns=['default']).to_numpy()
y_train = train_df['default'].to_numpy()
X_test = test_df.drop(columns=['default']).to_numpy()
y_test = test_df['default'].to_numpy()

# Нормализация
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std[std == 0] = 1 #не делить на 0

X_train_normalized = (X_train - mean) / std
X_test_normalized = (X_test - mean) / std

# Сохранение
X_train_df = pd.DataFrame(X_train_normalized, columns=train_df.columns[:-1])
X_test_df = pd.DataFrame(X_test_normalized, columns=test_df.columns[:-1])
y_train_df = pd.DataFrame(y_train, columns=['default'])
y_test_df = pd.DataFrame(y_test, columns=['default'])

X_train_df.to_csv('train_set_normalized.csv', index=False)
X_test_df.to_csv('test_set_normalized.csv', index=False)
y_train_df.to_csv('train_set_y.csv', index=False)
y_test_df.to_csv('test_set_y.csv', index=False)

# Проверка
print(f"Размер обучающей выборки: {X_train_normalized.shape}")
print(f"Размер тестовой выборки: {X_test_normalized.shape}")
print(f"Средние значения признаков (train): {np.mean(X_train_normalized, axis=0)}")
print(f"Стандартные отклонения (train): {np.std(X_train_normalized, axis=0)}")
print(f"Доля дефолтов (train): {np.mean(y_train):.4f}")
print(f"Доля дефолтов (test): {np.mean(y_test):.4f}")
print("Нормализованные данные сохранены")