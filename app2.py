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

# ==============================================================================
# 🛠️ SESSION STATE MEMORY SYSTEM (PREVENTS NESTED BUTTON RESET)
# ==============================================================================
if "calculated" not in st.session_state:
    st.session_state.calculated = False
    st.session_state.pass_pct = 0.0
    st.session_state.fail_pct = 0.0

# LINE 52: Your original button action
if st.button("Calculate Probability"):
    st.session_state.calculated = True
    
    # 1. Your exact original DataFrame compilation matrix
    input_data = pd.DataFrame([{
        'age': age, 
        'study_hours_per_day': study_hours, 
        'netflix_hours': netflix_hours, 
        'attendance_percentage': attendance_percentage,
        'sleep_hours': sleep_hours,
        'exercise_frequency': exercise_freq, 
        'gender': gender, 
        'part_time_job': part_time_job, 
        'diet_quality': diet_quality,
        'parental_education_level': parental_edu, 
        'internet_quality': internet_quality,
        'extracurricular_participation': extracurricular
    }])
    
    # 2. Your original dummy mapping & machine learning predictions
    input_encoded = pd.get_dummies(input_data)
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
    probabilities = model.predict_proba(input_final)
    
    # Cache metrics inside session state memory so they don't disappear
    st.session_state.fail_pct = probabilities[0][0] * 100
    st.session_state.pass_pct = probabilities[0][1] * 100

# ==============================================================================
# 📊 PERSISTENT DISPLAY WINDOW (RUNS OUTSIDE THE FIRST BUTTON BLOCK)
# ==============================================================================
if st.session_state.calculated:
    pass_pct = st.session_state.pass_pct
    fail_pct = st.session_state.fail_pct
    
    st.markdown("---")
    if pass_pct >= 50:
        st.success("### Result: Likely to Pass")
    else:
        st.error("### Result: At Academic Risk (Likely to Fail)")
        
    st.metric(label="Success Probability", value=f"{pass_pct:.1f}% Pass")
    st.metric(label="Risk Probability", value=f"{fail_pct:.1f}% Fail")
    st.progress(int(pass_pct))
    
    # ==============================================================================
    # 🎯 COMPLETELY ISOLATED PREMIUM TIERS
    # ==============================================================================
    st.markdown("---")
    
    # --------------------------------------------------------------------------
    # SUCCESS/PASSING PREMIUM SUITE
    # --------------------------------------------------------------------------
    if pass_pct >= 50:
        st.info("💡 **Premium Optimization Available:** You qualify for zero-stress mastery resources designed to maximize your final grade boundaries beautifully without adding unnecessary academic pressure.")
        
        if st.button("✨ Unlock Success Maximizer Vault (Premium)", key="premium_pass_features"):
            st.warning("🔒 **Monetization Pending Compliance:** This premium feature is simulated for demonstration purposes pending our official business registration certificate.")
            
            st.markdown("### 💎 Premium Success Optimization Suite")
            
            st.markdown("#### 1. Academic Burnout & Fatigue Index")
            calculated_burnout = min(100, int((study_hours * 12) + (10 - sleep_hours * 5)))
            st.write(f"Current Operational Stress Index: **{calculated_burnout}/100**")
            if study_hours > 6 or sleep_hours < 5:
                st.error("⚠️ CRITICAL FATIGUE WARNING: High study density paired with restricted sleep patterns detected.")
                st.write("**Fatigue Mitigation Schedule:** You can safely scale back daily study blocks by 1.5 hours to reclaim healthy sleep cycles without dropping below a safe passing baseline.")
            else:
                st.write("🟢 Your fatigue index is completely stable. Current routines do not show a critical threat of physical or mental breakdown.")
            
            st.markdown("#### 2. 'Study-to-Free-Time' Optimization Matrix")
            st.write("📈 **Efficiency Routine Generated:** Because your current habits are effective, you can safely scale back independent study blocks by **1.5 hours daily**.")
            st.write("Your modified schedule reallocates that time into guilt-free rest or hobbies while comfortably preserving your high marks.")
            
            st.markdown("#### 3. Automated Sleep & Cognitive Recovery Mapping")
            st.write(f"🧠 **AI Sleep Window Anchors:** To maximize cognitive retention during heavy study weeks, your profile requires a fixed bedtime and wake-up cycle calculated to preserve REM cycles based on your current {sleep_hours} baseline hours.")
            
            st.markdown("#### 4. Personal 'Syllabus Weighting' Routine Builder")
            st.write("🎯 **Active Recall Distribution Matrix Applied to Your Study Blocks:**")
            weighting_data = {
                "Methodology Focus": ["Active Practice Testing (Mock Exams & Retrieval Flashcards)", "Targeted High-Yield Content Review"],
                "Time Allocation": ["60% of Your Time Block", "40% of Your Time Block"]
            }
            st.table(pd.DataFrame(weighting_data))
            
            st.markdown("#### 5. Continuous Assessment Buffer Calculator")
            st.write("🛡️ **Academic Cushion Analysis:** Your input metrics indicate you have built a resilient academic barrier. You can afford to score significantly lower on unexpected exam questions and still pass the semester comfortably.")
            
            st.markdown("#### 6. 'Social Media/Streaming' Recovery Plan")
            st.write(f"📱 **Automated Screen-Time Transition Tracker:** Active. Your system has designed a breakdown routine that safely scales down your logged {netflix_hours} streaming hours by 15 minutes every two days, automatically reallocating it to cognitive recovery.")
            
            st.markdown("#### 7. 'Exam-Week Peak Performance' Protocol")
            st.write("🏁 **7-Day Countdown Taper Protocol Loaded:**")
            taper_data = {
                "Timeline Remaining": ["Days 7 to 5 before Exams", "Days 4 to 2 before Exams", "Day 1 before Exams"],
                "Routine Transition Strategy": [
                    "Maintain current study volumes but shift completely away from passive reading to active recall.",
                    "Reduce independent study blocks by 30%. Increase hydration and expand night rest cycles.",
                    "Halt intense conceptual study by 2:00 PM. Dedicate evening to light review and physical relaxation to peak performance."
                ]
            }
            st.table(pd.DataFrame(taper_data))

    # --------------------------------------------------------------------------
    # RISK/FAILING PREMIUM SUITE
    # --------------------------------------------------------------------------
    else:
        st.info("💡 **Premium Resource Identified:** A targeted AI Lifestyle Calendar and optimized structural timetable are ready to deploy to stabilize your trajectory.")
        
        if st.button("✨ Unlock AI Calendar & Personal Timetable (Premium)", key="premium_fail_features"):
            st.warning("🔒 **Monetization Pending Compliance:** This premium feature is simulated for demonstration purposes pending our official business registration certificate.")
            
            st.markdown("### 🗓️ Customized AI Lifestyle Recovery Roadmap")
            
            optimized_netflix = round(netflix_hours * 0.5, 1)
            reclaimed_hours = netflix_hours - optimized_netflix
            optimized_sleep = sleep_hours + min(2.0, reclaimed_hours)
            target_daily_study = max(5.0, study_hours + 1.5)
            
            st.markdown(f"""
            *   **Automated Screen-Time Reduction:** Your logged digital leisure metrics have been automatically scaled back from **{netflix_hours} hours** to **{optimized_netflix} hours** daily to free up cognitive bandwidth.
            *   **Sleep Optimization Check:** Your profile logged **{sleep_hours} hours** of sleep. Reclaiming time from streaming, your new sleep window has been optimized to **{optimized_sleep} hours** for mental recovery.
            *   **Core Study Hour Expansion:** To balance your model weight profiles, your daily core study target has been upgraded to a baseline of **{target_daily_study} hours** daily.
            """)
            
            st.markdown("### 📋 Your Optimized Structural Routine")
            schedule_data = {
                "Daily Window": ["Morning Window", "Afternoon Block", "Evening Block", "Night Routine"],
                "Actionable Focus Blocks": [
                    "Attend scheduled lectures, compile missing concept cards, and complete active note summaries immediately post-class.",
                    f"Dedicated core study block: Focus the first {round(target_daily_study / 2, 1)} hours purely on critical course assignments.",
                    f"Review weak topics, solve past question sets, and engage in active testing for {round(target_daily_study / 2, 1)} hours.",
                    f"Enforced cognitive wind-down. Transition away from screens to secure your optimized {optimized_sleep}-hour recovery window."
                ]
            }
            st.table(pd.DataFrame(schedule_data))
