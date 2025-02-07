# Self-Analysis-Mental-Health

## Overview
This project is a **Mental Health Prediction System** that analyzes user input to predict **depression and anxiety severity levels** using machine learning models. It also provides **natural language explanations** and **suggested coping mechanisms** using an open-source LLM.

## Features
- Predicts **depression** and **anxiety severity levels** based on user responses.
- Provides **natural language explanations** for the predictions.
- Suggests **coping mechanisms** based on severity levels.
- **User Interface**:
  - **Command-line script** (`predict_mental_health.py`)
  - **Streamlit UI** for an interactive experience.
  - **Jupyter Notebook** for model testing.
- **PDF Report** summarizing dataset insights, model performance, and findings.

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mental-health-prediction.git
cd mental-health-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model Files
Ensure `rf_depression_model.pkl` and `rf_anxiety_model.pkl` are in the project directory.

## Usage
### 1. Run the Streamlit UI
```bash
streamlit run mental_health_ui.py
```

### 2. Run the CLI-based Predictor
```bash
python predict_mental_health.py
```

### 3. Test in Jupyter Notebook
```bash
jupyter notebook
# Open and run model_testing.ipynb
```

## Model Details
- **Random Forest Classifier** trained on mental health datasets.
- **Feature Engineering:** BMI interactions, total mental health score.
- **LLM Integration:** Uses an open-source model to generate explanations.

## Contributions
Feel free to fork the repo, create a new branch, and submit a pull request.

## License
This project is licensed under the MIT License.

---

🚀 **Developed by [Your Name]**

