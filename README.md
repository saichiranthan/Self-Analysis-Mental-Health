# Self-Analysis-Mental-Health

## Overview
This project is a **Mental Health Prediction System** that analyzes user input to predict **depression and anxiety severity levels** using machine learning models. It also provides **natural language explanations** and **suggested coping mechanisms**. The detailed explanation of the process is in EDA_&_ModelDevelopment.ipynb and Video of the explanation will be uploaded within EOD.

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
git clone https://github.com/saichiranthan/Self-Analysis-Mental-Health.git
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
Make sure that saved models are in the same directory or specify their location in the above code

## Model Details
- **Random Forest Classifier** trained on mental health datasets.
- **Feature Engineering:** BMI interactions, total mental health score.
- **LLM Integration:** Uses an open-source model to generate explanations.

## Contributions
Feel free to fork the repo, create a new branch, and submit a pull request.


---

🚀 **Developed by Sai CHiranthan H M**

