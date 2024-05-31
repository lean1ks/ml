import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('result.csv')

# Преобразование категориальных признаков
cat = {}
categorical_columns = ['income_type', 'gender', 'total_income_category', 'purpose_category']

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    cat[col] = encoder

# Определение признаков и целевой переменной
features = ['income_type', 'children', 'days_employed', 'dob_years', 'education_id',
            'family_status_id', 'gender', 'purpose_category', 'total_income', 'total_income_category']
target = 'debt'

X = df[features]
y = df[target]

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Обучение моделей
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

print(f'Точность логистической регрессии: {lr_accuracy:.2f}')
print(f'Точность случайного леса: {rf_accuracy:.2f}')
df['debt'] = y

corr_matrix = df[features + ['debt']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Корреляционная матрица')
plt.show()
