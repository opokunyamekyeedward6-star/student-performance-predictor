import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("🎓 Student Academic Performance Predictor")
st.write("Input student habits below to calculate success probability metrics.")

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('student_habits_performance.csv')
    df['pass_fail'] = np.where(df['exam_score'] >= 50, 1, 0)
    df = df.drop(columns=['student_id', 'exam_score'])
    
    categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'extracurricular_participation']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    X = df_encoded.drop(columns=['pass_fail'])
    y = df_encoded['pass_fail']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, list(X.columns)

try:
    model, model_columns = load_and_train_model()
except Exception as e:
    st.error("Dataset file 'student_habits_performance.csv' not found in the repository setup.")

st.subheader("Daily Lifestyle Metrics")
study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=3.0)
attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=85.0)
sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, value=7.0)
social_media = st.number_input("Social Media Hours per Day", min_value=0.0, max_value=24.0, value=2.0)
netflix_hours = st.number_input("Netflix Hours per Day", min_value=0.0, max_value=24.0, value=1.5)
exercise_freq = st.slider("Exercise Frequency (Days per week)", 0, 7, 3)
mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 7)
age = st.number_input("Age", min_value=15, max_value=100, value=20)

st.subheader("Categorical Information")
gender = st.selectbox("Gender", ["Male", "Female"])
part_time_job = st.selectbox("Has Part-Time Job?", ["Yes", "No"])
diet_quality = st.selectbox("Diet Quality", ["Good", "Fair", "Poor"])
parental_edu = st.selectbox("Parental Education Level", ["High School", "Bachelor", "Master", "PhD"])
internet_quality = st.selectbox("Internet Quality", ["Good", "Average", "Poor"])
extracurricular = st.selectbox("Extracurricular Participation?", ["Yes", "No"])

if st.button("Calculate Probability"):
    input_data = pd.DataFrame([{
        'age': age, 'study_hours_per_day': study_hours, 'social_media_hours': social_media,
        'netflix_hours': netflix_hours, 'attendance_percentage': attendance, 'sleep_hours': sleep_hours,
        'exercise_frequency': exercise_freq, 'mental_health_rating': mental_health,
        'gender': gender, 'part_time_job': part_time_job, 'diet_quality': diet_quality,
        'parental_education_level': parental_edu, 'internet_quality': internet_quality,
        'extracurricular_participation': extracurricular
    }])
    
    input_encoded = pd.get_dummies(input_data)
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    probabilities = model.predict_proba(input_final)
    fail_pct = probabilities[0][0] * 100
    pass_pct = probabilities[0][1] * 100
    
    st.markdown("---")
    if pass_pct >= 50:
        st.success(f"### Result: Likely to Pass")
    else:
        st.error(f"### Result: At Academic Risk (Likely to Fail)")
        
    st.metric(label="Success Probability", value=f"{pass_pct:.1f}% Pass")
    st.metric(label="Risk Probability", value=f"{fail_pct:.1f}% Fail")
    st.progress(int(pass_pct))
