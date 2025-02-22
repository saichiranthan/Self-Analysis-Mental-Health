import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

class MentalHealthAnalyzer:
    def __init__(self):
        try:
            # Load the trained models and scaler
            self.rf_depression = joblib.load(r"C:\Users\chira\Downloads\Arogo AI\rf_depression_model.pkl")
            self.rf_anxiety = joblib.load(r"C:\Users\chira\Downloads\Arogo AI\rf_anxiety_model.pkl")
            self.scaler = joblib.load(r"C:\Users\chira\Downloads\Arogo AI\scaler.pkl")
            self.feature_names = joblib.load(r"C:\Users\chira\Downloads\Arogo AI\feature_names.pkl")
            # Initialize encoders with default mappings
            self.gender_map = {'female': 0, 'male': 1}
            self.bmi_category_map = {
                "Underweight": 0,
                "Normal": 1,
                "Overweight": 2,
                "Class I Obesity": 3,
                "Class II Obesity": 4,
                "Class III Obesity": 5
            }
            
        except FileNotFoundError as e:
            st.error(f"Required model files not found: {str(e)}")
            st.stop()
        
        self.depression_severity_map = {
            0: "Mild",
            1: "Moderate",
            2: "Moderately Severe",
            3: "Severe",
            4: "Very Severe",
            5: "Extremely Severe"
        }
        
        self.anxiety_severity_map = {
            0: "Minimal",
            1: "Mild",
            2: "Moderate",
            3: "Severe",
            4: "Extremely Severe"
        }
        
        self.coping_mechanisms = {
            "Mild": [
                "Practice deep breathing exercises daily",
                "Maintain a regular sleep schedule",
                "Engage in light physical activity",
                "Keep a mood journal",
                "Try mindfulness meditation"
            ],
            "Moderate": [
                "Schedule regular counseling sessions",
                "Join support groups",
                "Practice stress management techniques",
                "Establish a daily routine",
                "Engage in regular exercise"
            ],
            "Severe": [
                "Seek immediate professional help",
                "Consider medication management with a healthcare provider",
                "Build a strong support network",
                "Practice crisis management techniques",
                "Use grounding exercises when feeling overwhelmed"
            ]
        }

    def _preprocess_input(self, input_data):
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Convert categorical variables to numeric
        df['gender'] = df['gender'].map(self.gender_map)
        df['who_bmi'] = df['who_bmi'].map(self.bmi_category_map)
        
        # Convert boolean fields to integers
        boolean_fields = ['depressiveness', 'suicidal', 'depression_diagnosis', 
                         'depression_treatment', 'anxiousness', 'anxiety_diagnosis', 
                         'anxiety_treatment', 'sleepiness']
        for field in boolean_fields:
            df[field] = df[field].astype(int)
        
        # Add placeholder for required features that aren't in input
        df['id'] = 0  # Placeholder ID
        df['school_year'] = 0  # Placeholder school year
        
        # Scale numerical features
        numerical_features = ['age', 'bmi', 'phq_score', 'gad_score', 'epworth_score']
        try:
            df[numerical_features] = self.scaler.transform(df[numerical_features])
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            st.error("Please ensure the scaler was properly fitted during model training.")
            st.stop()
        
        # Create derived features
        df['total_mental_health_score'] = df['phq_score'] + df['gad_score']
        df['bmi_who_interaction'] = df['bmi'] * df['who_bmi']
        
        # Ensure all required features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                st.error(f"Missing required feature: {feature}")
                st.stop()
        
        # Return DataFrame with exact column order from training
        return df[self.feature_names]

    def generate_explanation(self, depression_severity, anxiety_severity, scores):
        explanation = []
        
        # Depression explanation
        depression_level = self.depression_severity_map.get(depression_severity, "Unknown")
        phq_score = scores.get('phq_score', 0)
        explanation.append(f"\n🔍 Depression Analysis:")
        explanation.append(f"Based on your responses, you show indicators of {depression_level.lower()} depression severity.")
        explanation.append(f"Your PHQ-9 score of {phq_score} suggests {depression_level.lower()} depressive symptoms.")
        
        # Anxiety explanation
        anxiety_level = self.anxiety_severity_map.get(anxiety_severity, "Unknown")
        gad_score = scores.get('gad_score', 0)
        explanation.append(f"\n🔍 Anxiety Analysis:")
        explanation.append(f"Your results indicate {anxiety_level.lower()} anxiety severity.")
        explanation.append(f"Your GAD-7 score of {gad_score} suggests {anxiety_level.lower()} anxiety symptoms.")
        
        return "\n".join(explanation)

    def suggest_coping_mechanisms(self, depression_severity, anxiety_severity):
        depression_level = self.depression_severity_map.get(depression_severity, "Mild")
        anxiety_level = self.anxiety_severity_map.get(anxiety_severity, "Mild")
        
        # Get severity level for coping mechanisms
        severity = "Severe" if "Severe" in [depression_level, anxiety_level] else \
                  "Moderate" if "Moderate" in [depression_level, anxiety_level] else "Mild"
        
        mechanisms = self.coping_mechanisms[severity]
        
        suggestions = ["\n💡 Recommended Coping Strategies:"]
        suggestions.extend([f"• {mechanism}" for mechanism in mechanisms])
        
        if severity in ["Moderate", "Severe"]:
            suggestions.append("\n⚠️ Important Note: Please consider consulting with a mental health professional for personalized guidance and support.")
        
        return "\n".join(suggestions)

    def predict(self, input_data):
        try:
            # Prepare input data
            processed_data = self._preprocess_input(input_data)
            
            # Make predictions
            depression_pred = self.rf_depression.predict(processed_data)[0]
            anxiety_pred = self.rf_anxiety.predict(processed_data)[0]
            
            # Generate explanation and coping mechanisms
            explanation = self.generate_explanation(depression_pred, anxiety_pred, input_data)
            coping_mechanisms = self.suggest_coping_mechanisms(depression_pred, anxiety_pred)
            
            return {
                'depression_severity': self.depression_severity_map.get(depression_pred, "Unknown"),
                'anxiety_severity': self.anxiety_severity_map.get(anxiety_pred, "Unknown"),
                'explanation': explanation,
                'coping_mechanisms': coping_mechanisms
            }
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.error("Debug info - Data shape: {processed_data.shape}, Columns: {processed_data.columns}")
            raise e


def main():
    st.title("🧠 Mental Health Analysis Tool")
    st.write("This tool analyzes mental health indicators and provides personalized insights and recommendations.")
    
    try:
        analyzer = MentalHealthAnalyzer()
    except Exception as e:
        st.error("Failed to initialize the analyzer. Please check if all required model files are present.")
        st.stop()
    
    with st.form("mental_health_form"):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=25)
            gender = st.selectbox("Gender", ["male", "female"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
            
        with col2:
            who_bmi = st.selectbox("BMI Category", 
                                 ["Underweight", "Normal", "Overweight", "Class I Obesity",
                                  "Class II Obesity", "Class III Obesity"])
            epworth_score = st.slider("Epworth Sleepiness Score", 0, 24, 7)
            
        st.subheader("Depression Indicators")
        col3, col4 = st.columns(2)
        
        with col3:
            phq_score = st.slider("PHQ-9 Score", 0, 27, 5)
            depressiveness = st.checkbox("Feeling Depressed")
            suicidal = st.checkbox("Suicidal Thoughts")
            
        with col4:
            depression_diagnosis = st.checkbox("Previous Depression Diagnosis")
            depression_treatment = st.checkbox("Currently in Depression Treatment")
            
        st.subheader("Anxiety Indicators")
        col5, col6 = st.columns(2)
        
        with col5:
            gad_score = st.slider("GAD-7 Score", 0, 21, 5)
            anxiousness = st.checkbox("Feeling Anxious")
            
        with col6:
            anxiety_diagnosis = st.checkbox("Previous Anxiety Diagnosis")
            anxiety_treatment = st.checkbox("Currently in Anxiety Treatment")
            
        submitted = st.form_submit_button("Analyze")
        
        if submitted:
            input_data = {
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'who_bmi': who_bmi,
                'phq_score': phq_score,
                'depressiveness': depressiveness,
                'suicidal': suicidal,
                'depression_diagnosis': depression_diagnosis,
                'depression_treatment': depression_treatment,
                'gad_score': gad_score,
                'anxiousness': anxiousness,
                'anxiety_diagnosis': anxiety_diagnosis,
                'anxiety_treatment': anxiety_treatment,
                'epworth_score': epworth_score,
                'sleepiness': epworth_score > 10
            }
            
            try:
                results = analyzer.predict(input_data)
                
                st.subheader("Analysis Results")
                st.markdown(f"**Depression Severity:** {results['depression_severity']}")
                st.markdown(f"**Anxiety Severity:** {results['anxiety_severity']}")
                
                st.subheader("Detailed Analysis")
                st.markdown(results['explanation'])
                
                st.subheader("Recommendations")
                st.markdown(results['coping_mechanisms'])
                
                st.info("Note: This tool is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.error("Please try again or contact support if the problem persists.")

if __name__ == "__main__":
    main()
