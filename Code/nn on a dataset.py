import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Titanic dataset from seaborn or Kaggle (fallback to URL)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# --- Raw Features Model ---
raw_features = ['Pclass', 'Sex', 'Age', 'Fare']
df_raw = df[raw_features + ['Survived']].copy()

# Minimal preprocessing
df_raw['Sex'] = LabelEncoder().fit_transform(df_raw['Sex'])
df_raw['Age'].fillna(df_raw['Age'].median(), inplace=True)

X_raw = df_raw.drop('Survived', axis=1)
y_raw = df_raw['Survived']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train_r = scaler.fit_transform(X_train_r)
X_test_r = scaler.transform(X_test_r)

# Build model
def build_model(input_dim):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model_raw = build_model(X_train_r.shape[1])
model_raw.fit(X_train_r, y_train_r, epochs=20, verbose=0)
raw_acc = model_raw.evaluate(X_test_r, y_test_r, verbose=0)[1]
print(f"Raw Features Accuracy: {raw_acc:.4f}")

# --- Engineered Features Model ---
df_eng = df.copy()

# Feature engineering
df_eng['Sex'] = LabelEncoder().fit_transform(df_eng['Sex'])
df_eng['Embarked'] = LabelEncoder().fit_transform(df_eng['Embarked'].astype(str))
df_eng['FamilySize'] = df_eng['SibSp'] + df_eng['Parch'] + 1
df_eng['IsAlone'] = (df_eng['FamilySize'] == 1).astype(int)

engineered_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
df_eng = df_eng[engineered_features + ['Survived']]

# Impute missing
imputer = SimpleImputer(strategy='median')
df_eng[engineered_features] = imputer.fit_transform(df_eng[engineered_features])

X_eng = df_eng.drop('Survived', axis=1)
y_eng = df_eng['Survived']

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_eng, y_eng, test_size=0.2, random_state=42)

X_train_e = scaler.fit_transform(X_train_e)
X_test_e = scaler.transform(X_test_e)

# Train model
model_eng = build_model(X_train_e.shape[1])
model_eng.fit(X_train_e, y_train_e, epochs=20, verbose=0)
eng_acc = model_eng.evaluate(X_test_e, y_test_e, verbose=0)[1]
print(f"Engineered Features Accuracy: {eng_acc:.4f}")

# Comparison
print("\n📊 Comparison:")
print(f"- Raw Accuracy       : {raw_acc:.4f}")
print(f"- Engineered Accuracy: {eng_acc:.4f}")
