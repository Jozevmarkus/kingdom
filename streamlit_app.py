import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('winequality-dataset_updated.csv')
data

# Добавление: гистограмма для целевой переменной 'quality'
plt.figure(figsize=(8, 6))
sns.histplot(data['quality'], kde=True, color='purple')
plt.title('Распределение качества вина')
plt.xlabel('Качество')
plt.ylabel('Частота')
plt.show()

# Добавление: корреляционная матрица для всех признаков
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='viridis', linewidths=0.5)
plt.title('Корреляционная матрица признаков')
plt.show()
# Разделение на признаки и целевую переменную
X = data.drop(['quality'], axis=1)
y = data['quality']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Линейная регрессия
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Оценка линейной регрессии
r2_lr = r2_score(y_test, y_pred_lr)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
print(f'Линейная регрессия R2: {r2_lr:.2f}')
print(f'Линейная регрессия MAPE: {mape_lr:.2f}')

# Добавление: график предсказаний линейной регрессии
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_lr, color='blue')
plt.title('Предсказания vs Настоящие значения (Линейная регрессия)')
plt.xlabel('Настоящие значения')
plt.ylabel('Предсказанные значения')
plt.show()

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Дерево решений
model_tree = DecisionTreeRegressor(max_depth=10)
model_tree.fit(X_train, y_train)
y_pred_tree = model_tree.predict(X_test)

# Оценка дерева решений
r2_tree = r2_score(y_test, y_pred_tree)
mape_tree = mean_absolute_percentage_error(y_test, y_pred_tree)
print(f'Дерево решений R2: {r2_tree:.2f}')
print(f'Дерево решений MAPE: {mape_tree:.2f}')

# Случайный лес
model_rf = RandomForestRegressor(n_estimators=100)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Оценка случайного леса
r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)
print(f'Случайный лес R2: {r2_rf:.2f}')
print(f'Случайный лес MAPE: {mape_rf:.2f}')

# Градиентный бустинг
model_gb = GradientBoostingRegressor(n_estimators=100)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)

# Оценка градиентного бустинга
r2_gb = r2_score(y_test, y_pred_gb)
mape_gb = mean_absolute_percentage_error(y_test, y_pred_gb)
print(f'Градиентный бустинг R2: {r2_gb:.2f}')
print(f'Градиентный бустинг MAPE: {mape_gb:.2f}')

# Сравнение метрик моделей
models_results = pd.DataFrame({
    'Модель': ['Линейная регрессия', 'Дерево решений', 'Случайный лес', 'Градиентный бустинг'],
    'R2': [r2_lr, r2_tree, r2_rf, r2_gb],
    'MAPE': [mape_lr, mape_tree, mape_rf, mape_gb]
})

print(models_results)

# Итоговое сравнение моделей с визуализацией
plt.figure(figsize=(10, 6))
sns.barplot(x='Модель', y='R2', data=models_results)
plt.title('Сравнение R2 моделей')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Модель', y='MAPE', data=models_results)
plt.title('Сравнение MAPE моделей')
plt.show()
