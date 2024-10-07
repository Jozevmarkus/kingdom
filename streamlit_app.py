import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

try:
    import xgboost as xgb
    from catboost import CatBoostRegressor
    has_boosting = True
except ImportError:
    has_boosting = False
st.title('Анализ качества вина с использованием машинного обучения')

data = pd.read_csv('winequality-dataset_updated.csv')
st.write("Просмотр данных:")
st.write(data)

st.write("Распределение качества вина:")
fig, ax = plt.subplots()
ax.hist(data['quality'], bins=10, color='purple', edgecolor='black')
ax.set_title('Распределение качества вина')
ax.set_xlabel('Качество')
ax.set_ylabel('Частота')
st.pyplot(fig)
plt.close(fig)
st.write("Корреляционная матрица признаков:")
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(data.corr(), cmap='viridis')
fig.colorbar(cax)
plt.xticks(range(len(data.columns)), data.columns, rotation=90)
plt.yticks(range(len(data.columns)), data.columns)
st.pyplot(fig)
plt.close(fig)

X = data.drop(['quality'], axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
st.write(f'Линейная регрессия R2: {r2_lr:.2f}')
st.write(f'Линейная регрессия MAPE: {mape_lr:.2f}')

st.write("Предсказания vs Настоящие значения (Линейная регрессия):")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_lr, color='blue')
ax.set_title('Предсказания vs Настоящие значения (Линейная регрессия)')
ax.set_xlabel('Настоящие значения')
ax.set_ylabel('Предсказанные значения')
st.pyplot(fig)
plt.close(fig)  # Закрытие графика

model_tree = DecisionTreeRegressor(max_depth=10)
model_tree.fit(X_train, y_train)
y_pred_tree = model_tree.predict(X_test)

r2_tree = r2_score(y_test, y_pred_tree)
mape_tree = mean_absolute_percentage_error(y_test, y_pred_tree)
st.write(f'Дерево решений R2: {r2_tree:.2f}')
st.write(f'Дерево решений MAPE: {mape_tree:.2f}')

model_rf = RandomForestRegressor(n_estimators=100)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)
st.write(f'Случайный лес R2: {r2_rf:.2f}')
st.write(f'Случайный лес MAPE: {mape_rf:.2f}')

model_gb = GradientBoostingRegressor(n_estimators=100)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)

r2_gb = r2_score(y_test, y_pred_gb)
mape_gb = mean_absolute_percentage_error(y_test, y_pred_gb)
st.write(f'Градиентный бустинг R2: {r2_gb:.2f}')
st.write(f'Градиентный бустинг MAPE: {mape_gb:.2f}')

if has_boosting:
    model_cat = CatBoostRegressor(verbose=False)
    model_cat.fit(X_train, y_train)
    y_pred_cat = model_cat.predict(X_test)

    r2_cat = r2_score(y_test, y_pred_cat)
    mape_cat = mean_absolute_percentage_error(y_test, y_pred_cat)
    st.write(f'CatBoost R2: {r2_cat:.2f}')
    st.write(f'CatBoost MAPE: {mape_cat:.2f}')

    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, n_estimators=100)
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_test)

    r2_xgb = r2_score(y_test, y_pred_xgb)
    mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)
    st.write(f'XGBoost R2: {r2_xgb:.2f}')
    st.write(f'XGBoost MAPE: {mape_xgb:.2f}')

models_results = pd.DataFrame({
    'Модель': ['Линейная регрессия', 'Дерево решений', 'Случайный лес', 'Градиентный бустинг'],
    'R2': [r2_lr, r2_tree, r2_rf, r2_gb],
    'MAPE': [mape_lr, mape_tree, mape_rf, mape_gb]
})

if has_boosting:
    models_results = models_results.append({
        'Модель': 'CatBoost', 'R2': r2_cat, 'MAPE': mape_cat
    }, ignore_index=True)
    models_results = models_results.append({
        'Модель': 'XGBoost', 'R2': r2_xgb, 'MAPE': mape_xgb
    }, ignore_index=True)

st.write("Результаты моделей:")
st.write(models_results)

st.write("Сравнение R2 моделей:")
fig, ax = plt.subplots()
ax.bar(models_results['Модель'], models_results['R2'], color=['blue', 'green', 'orange', 'red', 'purple', 'brown'])
ax.set_title('Сравнение R2 моделей')
ax.set_xticklabels(models_results['Модель'], rotation=45, ha='right')  # Поворот текста на оси X
st.pyplot(fig)
plt.close(fig) 

st.write("Сравнение MAPE моделей:")
fig, ax = plt.subplots()
ax.bar(models_results['Модель'], models_results['MAPE'], color=['blue', 'green', 'orange', 'red', 'purple', 'brown'])
ax.set_title('Сравнение MAPE моделей')
ax.set_xticklabels(models_results['Модель'], rotation=45, ha='right')  # Поворот текста на оси X
st.pyplot(fig)
plt.close(fig)  # Закрытие графика
