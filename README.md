# Loan-granting-manager
# 🏦 Credit Scoring with CatBoost on Tinkoff Data

## 📋 Описание проекта
End-to-end ML-решение для задачи кредитного скоринга на основе открытых данных Tinkoff. 
Модель предсказывает вероятность дефолта заемщика (бинарная классификация).

## 🛠 Стек технологий
- **Machine Learning**: CatBoost, Scikit-learn, Pandas, NumPy
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Сериализация**: Joblib
- **Визуализация**: Matplotlib, Seaborn

## 📊 Дадасет
- Источник: Kagle Tinkoff credit scoring data
- Признаки: демография, кредитная история, скоринговые баллы
- Целевая переменная: default_flg (0/1)

## 🧠 Модель
- Алгоритм: CatBoostClassifier (градиентный бустинг)
- Метрики: AUC-ROC, Precision, Recall, F1-score
- Оптимизация: GridSearchCV / RandomizedSearchCV
- Балансировка: class_weights для работы с дисбалансом классов

## 🔧 Функциональность
- Обучение и валидация модели
- REST API для получения предсказаний (/predict)
- Веб-интерфейс для тестирования модели
- Сохранение/загрузка обученной модели

## 🚀 Быстрый старт
```bash
# 1. Клонировать репозиторий
git clone https://github.com/maks-0123/Loan-granting-manger.git

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Обучить модель
python train.py

# 4. Запустить API
uvicorn service:app

# 5. Запустить Streamlit интерфейс
streamlit run app.py
