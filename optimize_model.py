import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("Starting Data Optimization Pipeline...")

# 1. Load Data
df = pd.read_csv('malnutrition_children_ethiopia (1).csv')
print(f"Data shape initially: {df.shape}")

# 2. Feature Engineering
# Create BMI feature
df['BMI'] = df['Weight_kg'] / ((df['Height_cm'] / 100) ** 2)

# Create Disease Context Score
df['Disease_Context_Score'] = 0
for col in ['Malaria', 'Diarrhea', 'TB', 'Anemia']:
    if col in df.columns:
        df['Disease_Context_Score'] += df[col].apply(lambda x: 1 if str(x).lower().strip() == 'yes' else 0)

# Drop ID variable if exists
if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)

# 3. Handle Missing values and Encode Categorical Variables
categorical_cols = df.select_dtypes(include=['object']).columns
categorical_cols = [c for c in categorical_cols if c != 'Nutrition_Status']  # Target to be handled separately

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Simple imputation by mode for categorical
    df[col].fillna(df[col].mode()[0], inplace=True)
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Map Target manually or with LabelEncoder
if 'Nutrition_Status' in df.columns:
    target_le = LabelEncoder()
    df['Nutrition_Status'].fillna(df['Nutrition_Status'].mode()[0], inplace=True)
    df['Target'] = target_le.fit_transform(df['Nutrition_Status'])
    df.drop('Nutrition_Status', axis=1, inplace=True)
    le_dict['Target'] = target_le
else:
    raise ValueError("Target 'Nutrition_Status' not found!")

# Numeric Imputation (Median)
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# 4. Define X and y
X = df.drop(['Target'], axis=1)
y = df['Target']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Pre-SMOTE Training Class distribution:")
print(y_train.value_counts())

# 6. Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Post-SMOTE Training Class distribution:")
print(y_train_sm.value_counts())

# 7. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# 8. Train & Hyperparameter Tune RandomForest
print("Starting RandomizedSearchCV for RandomForest...")
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

rf = RandomForestClassifier(random_state=42)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                               n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=None)
rf_random.fit(X_train_scaled, y_train_sm)

best_rf = rf_random.best_estimator_

# 9. Evaluate Best Model
y_pred = best_rf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Optimized Random Forest Accuracy: {acc * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_le.classes_))

# 10. Save Assets
with open('malnutrition_rf_v2_optimized.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

with open('malnutrition_scaler_v2_optimized.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Saved highly-optimized model to 'malnutrition_rf_v2_optimized.pkl'")
print("Saved Scaler to 'malnutrition_scaler_v2_optimized.pkl'")
