import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Загрузка данных
data = pd.read_csv('data/application_info.csv')
target = pd.read_csv('data/default_flg.csv')['default_flg']

if len(data) > len(target):
    data = data.iloc[:len(target)]

# Разделение на признаки и целевая переменная
cat_col = ['gender_cd', 'car_own_flg', 'education_cd']
num_col = ['age', 'appl_rej_cnt', 'income', 'Score_bki']

# Разделение на train/test
train_data, test_data, y_train, y_test = train_test_split(
    data, target, train_size=0.8, random_state=42, stratify=target
)

# Кодирование категориальных данных
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat_enc = ohe.fit_transform(train_data[cat_col])
X_test_cat_enc = ohe.transform(test_data[cat_col])

cat_columns = ohe.get_feature_names_out(cat_col)
X_train_cat_df = pd.DataFrame(
    X_train_cat_enc,
    columns=cat_columns,
    index=train_data.index
)
X_test_cat_df = pd.DataFrame(
    X_test_cat_enc,
    columns=cat_columns,
    index=test_data.index
)

# Объединение числовых и категориальных признаков
X_train_final = pd.concat([train_data[num_col], X_train_cat_df], axis=1)
X_test_final = pd.concat([test_data[num_col], X_test_cat_df], axis=1)

# Веса классов для дисбаланса
class_weights = [1, (len(y_train) - y_train.sum()) / y_train.sum()]

# МОДЕЛЬ 1: CatBoost
print("="*50)
print("ОБУЧЕНИЕ CATBOOST")
print("="*50)

model1 = CatBoostClassifier(
    iterations=500,
    learning_rate=0.01,
    depth=8,
    loss_function='Logloss',
    custom_metric=['AUC'], 
    random_seed=63,
    class_weights=class_weights,
    verbose=100,
    early_stopping_rounds=50  # Добавим раннюю остановку
)

# Используем Pool для CatBoost (опционально, можно и напрямую)
train_pool = Pool(X_train_final, y_train, cat_features=[])  # категориальные уже закодированы
model1.fit(train_pool, eval_set=(X_test_final, y_test), verbose=100)

# МОДЕЛЬ 2: Random Forest с GridSearch
print("\n" + "="*50)
print("ОБУЧЕНИЕ RANDOM FOREST С GRIDSEARCH")
print("="*50)

# Уменьшим сетку параметров для ускорения (или оставим полную, если время позволяет)
param_grid_detailed = {
    'n_estimators': [100],  # Уменьшил для скорости
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [2],
    'max_features': ['log2'],
    'bootstrap': [True],
    'class_weight': ['balanced']
}

# GridSearch
grid_search_detailed = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid_detailed,
    cv=3,  
    scoring='roc_auc',
    n_jobs=1,
    verbose=False
)

grid_search_detailed.fit(X_train_final, y_train)

# Лучшая модель Random Forest
model2 = grid_search_detailed.best_estimator_
print(f"\nЛучшие параметры Random Forest: {grid_search_detailed.best_params_}")
print(f"Лучший AUC на кросс-валидации: {grid_search_detailed.best_score_:.4f}")

# Предсказания для CatBoost
y_pred_proba_cb = model1.predict_proba(X_test_final)[:, 1]
y_pred_cb = model1.predict(X_test_final)

# Предсказания для Random Forest
y_pred_proba_rf = model2.predict_proba(X_test_final)[:, 1]
y_pred_rf = model2.predict(X_test_final)



# Функция для вывода метрик
def print_metrics(y_true, y_pred, y_pred_proba, model_name):
    auc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"РЕЗУЛЬТАТЫ МОДЕЛИ: {model_name}")
    print(f"{'='*50}")
    print(f"Доля отказов в предсказаниях: {y_pred.mean() * 100:.2f}%")
    print(f"Фактическая доля отказов: {y_true.mean() * 100:.2f}%")
    print(f"\nМетрики классификации:")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Точность (Precision): {precision*100:.2f}%")
    print(f"Полнота (Recall): {recall*100:.2f}%")
    print(f"F1-мера: {f1:.4f}")
    
    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nМатрица ошибок:")
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")
    
    return {'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'cm': cm}

# Вывод метрик для обеих моделей
metrics_cb = print_metrics(y_test, y_pred_cb, y_pred_proba_cb, "CatBoost")
metrics_rf = print_metrics(y_test, y_pred_rf, y_pred_proba_rf, "Random Forest (оптимизированный)")

# Сравнение моделей
print("\n" + "="*50)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*50)
print(f"{'Метрика':<15} {'CatBoost':<15} {'Random Forest':<15}")
print(f"{'-'*45}")
print(f"{'AUC-ROC':<15} {metrics_cb['auc']:.4f}{'':11} {metrics_rf['auc']:.4f}")
print(f"{'Precision':<15} {metrics_cb['precision']*100:.2f}%{'':10} {metrics_rf['precision']*100:.2f}%")
print(f"{'Recall':<15} {metrics_cb['recall']*100:.2f}%{'':10} {metrics_rf['recall']*100:.2f}%")
print(f"{'F1-score':<15} {metrics_cb['f1']:.4f}{'':11} {metrics_rf['f1']:.4f}")

# Визуализация ROC-кривых
plt.figure(figsize=(10, 6))

# CatBoost
fpr_cb, tpr_cb, _ = roc_curve(y_test, y_pred_proba_cb)
plt.plot(fpr_cb, tpr_cb, label=f'CatBoost (AUC = {metrics_cb["auc"]:.4f})', linewidth=2)

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {metrics_rf["auc"]:.4f})', linewidth=2)

# Диагональная линия
plt.plot([0, 1], [0, 1], 'k--', label='Случайная модель', alpha=0.5)

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC-кривые моделей', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Сохранение моделей, сохраняем только модель catboost так как она имеет лучшие метрики 

joblib.dump(model1, 'catboost_model.pkl')
joblib.dump(ohe, 'encoder.pkl')