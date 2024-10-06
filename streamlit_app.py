import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Заголовок
st.title('Анализ качества вина с использованием машинного обучения')

# Загрузка данных
uploaded_file = st.file_uploader("Загрузите CSV файл с данными о вине", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    # Гистограмма для целевой переменной 'quality' с использованием matplotlib
    plt.figure(figsize=(8, 6))
    plt.hist(data['quality'], bins=10, color='purple', edgecolor='black')
    plt.title('Распределение качества вина')
    plt.xlabel('Качество')
    plt.ylabel('Частота')
    st.pyplot(plt.gcf())  # Отображение графика через Streamlit

    # Корреляционная матрица для всех признаков
    plt.figure(figsize=(10, 8))
    plt.matshow(data.corr(), cmap='viridis', fignum=1)
    plt.colorbar()
    plt.title('Корреляционная матрица признаков', pad=40)
    st.pyplot(plt.gcf())  # Отображение графика через Streamlit

    # Разделение на признаки и целевую переменную
    X = data.drop(['quality'], axis=1)
    y = data['quality']

    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Линейная регрессия
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)

    # Оценка линейной регрессии
    r2_lr = r2_score(y_test, y_pred_lr)
    mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
    st.write(f'Линейная регрессия R2: {r2_lr:.2f}')
    st.write(f'Линейная регрессия MAPE: {mape_lr:.2f}')

    # График предсказаний линейной регрессии
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_lr, color='blue')
    plt.title('Предсказания vs Настоящие значения (Линейная регрессия)')
    plt.xlabel('Настоящие значения')
    plt.ylabel('Предсказанные значения')
    st.pyplot(plt.gcf())

    # Дерево решений
    model_tree = DecisionTreeRegressor(max_depth=10)
    model_tree.fit(X_train, y_train)
    y_pred_tree = model_tree.predict(X_test)

    # Оценка дерева решений
    r2_tree = r2_score(y_test, y_pred_tree)
    mape_tree = mean_absolute_percentage_error(y_test, y_pred_tree)
    st.write(f'Дерево решений R2: {r2_tree:.2f}')
    st.write(f'Дерево решений MAPE: {mape_tree:.2f}')

    # Случайный лес
    model_rf = RandomForestRegressor(n_estimators=100)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)

    # Оценка случайного леса
    r2_rf = r2_score(y_test, y_pred_rf)
    mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)
    st.write(f'Случайный лес R2: {r2_rf:.2f}')
    st.write(f'Случайный лес MAPE: {mape_rf:.2f}')

    # Градиентный бустинг
    model_gb = GradientBoostingRegressor(n_estimators=100)
    model_gb.fit(X_train, y_train)
    y_pred_gb = model_gb.predict(X_test)

    # Оценка градиентного бустинга
    r2_gb = r2_score(y_test, y_pred_gb)
    mape_gb = mean_absolute_percentage_error(y_test, y_pred_gb)
    st.write(f'Градиентный бустинг R2: {r2_gb:.2f}')
    st.write(f'Градиентный бустинг MAPE: {mape_gb:.2f}')

    # Сравнение метрик моделей
    models_results = pd.DataFrame({
        'Модель': ['Линейная регрессия', 'Дерево решений', 'Случайный лес', 'Градиентный бустинг'],
        'R2': [r2_lr, r2_tree, r2_rf, r2_gb],
        'MAPE': [mape_lr, mape_tree, mape_rf, mape_gb]
    })

    st.write(models_results)

    # Итоговое сравнение моделей с визуализацией
    plt.figure(figsize=(10, 6))
    plt.bar(models_results['Модель'], models_results['R2'], color=['blue', 'green', 'orange', 'red'])
    plt.title('Сравнение R2 моделей')
    st.pyplot(plt.gcf())

    plt.figure(figsize=(10, 6))
    plt.bar(models_results['Модель'], models_results['MAPE'], color=['blue', 'green', 'orange', 'red'])
    plt.title('Сравнение MAPE моделей')
    st.pyplot(plt.gcf())

