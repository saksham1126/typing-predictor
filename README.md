# Typing Speed vs Accuracy Predictor

## 📖 Problem Statement
Typing proficiency is typically measured by two main factors: **Speed (Words Per Minute or WPM)** and **Accuracy (%)**. However, a high typing speed with low accuracy can lead to inefficiencies due to the time required to correct mistakes.

The goal of this project is to build a Machine Learning model that predicts a comprehensive **Typing Efficiency Score** (from 0 to 100) based on three key metrics:
1. **Speed (WPM)**
2. **Accuracy (%)**
3. **Error Rate**

This efficiency score can help typing tutors and professional assessment platforms evaluate a user's true productivity.

## 📂 Folder Structure
```
typing-predictor/
├── data/
│   └── typing_data.csv          # Generated dataset
├── models/
│   └── best_model.pkl           # Saved trained model
├── output/                      # EDA plots and visual outputs
├── src/
│   ├── data_generation.py       # Script to generate synthetic data
│   └── model_training.py        # Script for EDA, modeling, and evaluation
├── app.py                       # Streamlit web application
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## 🚀 How to Run the Project Locally

### 1. Install Dependencies
Make sure you have Python installed. Then, run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
Run the data generation script to create the synthetic dataset:
```bash
python src/data_generation.py
```
This will create a `data/` folder and save `typing_data.csv`.

### 3. Train the Model
Run the machine learning pipeline to preprocess data, generate exploratory data analysis (EDA) plots, train models, and save the best model:
```bash
python src/model_training.py
```
This will output performance metrics in the console, save plots in the `output/` folder, and save `best_model.pkl` in the `models/` folder.

### 4. Run the Streamlit Web App
Launch the web interface to make interactive predictions:
```bash
streamlit run app.py
```
This will open a local web server (usually at `http://localhost:8501`) where you can adjust sliders and see real-time efficiency predictions.

## 🔍 Feature Interpretation
- **Consistency Score**: A derived feature that acts as a proxy for how smoothly a person types. Higher consistency means fewer bursts of errors.
- **Accuracy**: Is typically the most important feature. The model heavily penalizes low accuracy because correcting mistakes takes significant time.
- **Speed**: Contributes to a higher score, provided accuracy is maintained.
- **Error Rate**: Direct negative correlation with the final score.

## 🌟 Future Improvements
- **Real-time typing test**: Integrate an actual typing box in the Streamlit app where users type a paragraph, and the app calculates WPM/Accuracy on the fly before predicting the score.
- **Deep Learning**: For a larger dataset, recurrent neural networks (RNNs) could be used if sequential keystroke timing data was available.
- **User Profiles**: Save historical scores for a user to track their progress over time.
