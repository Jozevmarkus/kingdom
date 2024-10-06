# Импорт библиотек
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
# Загрузка данных
data = pd.read_csv('winequality-dataset_updated.csv')

# Пример правильного рендеринга графика
import streamlit as st
import matplotlib.pyplot as plt

st.title("График с правильным обновлением")

# Построение графика
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])

# Отображение через Streamlit
st.pyplot(fig)

# Просмотр данных
print("Первые 5 строк данных:")
print(data.head())

# Гистограмма для целевой переменной 'quality'
plt.figure(figsize=(8, 6))
plt.hist(data['quality'], bins=10, color='purple', edgecolor='black')
plt.title('Распределение качества вина')
plt.xlabel('Качество')
plt.ylabel('Частота')
plt.show()

# Корреляционная матрица для всех признаков
plt.figure(figsize=(10, 8))
plt.matshow(data.corr(), cmap='viridis')
plt.colorbar()
plt.title('Корреляционная матрица признаков')
plt.show()

# Разделение данных на признаки и целевую переменную
X = data.drop(['quality'], axis=1)
y = data['quality']

# Разделение данных на обучающую и тестовую выборки
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

# График предсказаний линейной регрессии
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, color='blue')
plt.title('Предсказания vs Настоящие значения (Линейная регрессия)')
plt.xlabel('Настоящие значения')
plt.ylabel('Предсказанные значения')
plt.show()

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

# График сравнения R2
plt.figure(figsize=(10, 6))
plt.bar(models_results['Модель'], models_results['R2'], color=['blue', 'green', 'orange', 'red'])
plt.title('Сравнение R2 моделей')
plt.show()

# График сравнения MAPE
plt.figure(figsize=(10, 6))
plt.bar(models_results['Модель'], models_results['MAPE'], color=['blue', 'green', 'orange', 'red'])
plt.title('Сравнение MAPE моделей')
plt.show()
