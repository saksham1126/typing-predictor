import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set plot style
sns.set_theme(style="whitegrid")

def main():
    print("--- Starting ML Pipeline ---")
    
    # 1. Load Data
    data_path = 'data/typing_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run data_generation.py first.")
        return
    
    df = pd.read_csv(data_path)
    print(f"Initial data shape: {df.shape}")
    
    # 2. Data Preprocessing (Handling missing values)
    print("\n--- Preprocessing Data ---")
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before imputation: {missing_before}")
    
    # Fill missing values with the median of their respective columns
    df['speed_wpm'] = df['speed_wpm'].fillna(df['speed_wpm'].median())
    df['accuracy_percent'] = df['accuracy_percent'].fillna(df['accuracy_percent'].median())
    
    print(f"Missing values after imputation: {df.isnull().sum().sum()}")
    
    # 3. Feature Engineering
    print("\n--- Feature Engineering ---")
    # Feature 1: consistency_score. High accuracy and low error rate means high consistency.
    # We add 1 to error_rate to avoid division by zero.
    df['consistency_score'] = df['accuracy_percent'] / (df['error_rate'] + 1)
    
    # Drop user_id as it's not a predictive feature
    df = df.drop(columns=['user_id'])
    
    # 4. Exploratory Data Analysis (EDA)
    print("\n--- Performing EDA ---")
    os.makedirs('output', exist_ok=True)
    
    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('output/correlation_heatmap.png')
    plt.close()
    
    # Speed vs Efficiency Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='speed_wpm', y='efficiency_score', hue='accuracy_percent', palette='viridis')
    plt.title('Speed vs Efficiency (Colored by Accuracy)')
    plt.tight_layout()
    plt.savefig('output/speed_vs_efficiency.png')
    plt.close()
    
    print("EDA plots saved in 'output/' folder.")
    
    # 5. Model Training
    print("\n--- Training Models ---")
    X = df.drop(columns=['efficiency_score'])
    y = df['efficiency_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({
            "Model": name,
            "R2 Score": r2,
            "MAE": mae,
            "RMSE": rmse
        })
        print(f"{name} -> R²: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    
    # 6. Hyperparameter Tuning (Random Forest as an example)
    print("\n--- Hyperparameter Tuning (Random Forest) ---")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Evaluate best model
    best_pred = best_rf.predict(X_test)
    best_r2 = r2_score(y_test, best_pred)
    print(f"Tuned Random Forest R²: {best_r2:.4f}")
    
    # 7. Model Interpretation
    print("\n--- Feature Importance ---")
    importances = best_rf.feature_importances_
    for feature, imp in zip(X.columns, importances):
        print(f"- {feature}: {imp:.4f}")
        
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=X.columns, palette='mako')
    plt.title('Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('output/feature_importance.png')
    plt.close()
    
    # 8. Save the best model
    print("\n--- Saving Model ---")
    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_model.pkl'
    joblib.dump(best_rf, model_path)
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    main()
