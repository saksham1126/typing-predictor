import pandas as pd
import numpy as np
import os

def generate_typing_data(num_samples=1500):
    """
    Generates a synthetic dataset for typing speed and accuracy.
    """
    np.random.seed(42)
    
    # 1. Generate Speed (WPM): normally distributed around 50 WPM, range ~10 to 120
    speed_wpm = np.random.normal(loc=50, scale=20, size=num_samples)
    speed_wpm = np.clip(speed_wpm, 10, 140) # Ensure realistic bounds
    
    # 2. Generate Accuracy (%): mostly high, heavily skewed towards 100%
    # Beta distribution is good for percentages
    accuracy = np.random.beta(a=8, b=2, size=num_samples) * 100
    accuracy = np.clip(accuracy, 30, 100)
    
    # 3. Generate Error Rate: Inverse relationship with accuracy roughly, plus some noise
    # Let's say max errors per minute is around 20
    error_rate = ((100 - accuracy) / 100) * (speed_wpm / 5) + np.random.normal(0, 1, num_samples)
    error_rate = np.clip(error_rate, 0, None) # Cannot be negative
    
    # 4. Calculate Target Variable: Efficiency Score (0-100)
    # A mix of speed and accuracy, heavily penalizing low accuracy and high errors
    # Base score driven by accuracy and scaled by speed
    efficiency_score = (speed_wpm * (accuracy / 100)) - (error_rate * 2)
    
    # Normalize to 0-100 range roughly
    min_score = np.min(efficiency_score)
    max_score = np.max(efficiency_score)
    efficiency_score = ((efficiency_score - min_score) / (max_score - min_score)) * 100
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': range(1, num_samples + 1),
        'speed_wpm': speed_wpm,
        'accuracy_percent': accuracy,
        'error_rate': error_rate,
        'efficiency_score': efficiency_score
    })
    
    # Inject some missing values to demonstrate preprocessing
    # Randomly set 2% of speed and accuracy to NaN
    nan_indices_speed = np.random.choice(df.index, size=int(num_samples * 0.02), replace=False)
    nan_indices_acc = np.random.choice(df.index, size=int(num_samples * 0.02), replace=False)
    
    df.loc[nan_indices_speed, 'speed_wpm'] = np.nan
    df.loc[nan_indices_acc, 'accuracy_percent'] = np.nan
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    file_path = 'data/typing_data.csv'
    df.to_csv(file_path, index=False)
    print(f"Successfully generated dataset with {num_samples} records and saved to {file_path}")
    print(f"Missing values injected: {df.isnull().sum().sum()}")

if __name__ == "__main__":
    generate_typing_data()
