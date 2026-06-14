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

# ==============================================================================
# 🔥 GLOBAL EXECUTION (NO INDENTATION)
# ==============================================================================
try:
    model, model_columns = load_and_train_model()
except Exception as e:
    st.error("Dataset file 'student_habits_performance.csv' not found in the repository setup.")

st.subheader("Daily Lifestyle Metrics")
study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=5.0)
attendance = st.number_input("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=75.0)
sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, value=7.0)
social_media = st.number_input("Social Media Hours per Day", min_value=0.0, max_value=24.0, value=3.0)
netflix_hours = st.number_input("Netflix Hours per Day", min_value=0.0, max_value=24.0, value=2.5)
exercise_freq = st.slider("Exercise Frequency (Days per week)", 0, 7, 5)
# 📑 UPDATED: Mental Health default baseline increased from 6 to 8
mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 8)
age = st.number_input("Age", min_value=15, max_value=100, value=20)

st.subheader("Categorical Information")
gender = st.selectbox("Gender", ["Male", "Female"])
part_time_job = st.selectbox("Has Part-Time Job?", ["Yes", "No"], index=1) 
diet_quality = st.selectbox("Diet Quality", ["Good", "Fair", "Poor"], index=0) 
parental_edu = st.selectbox("Parental Education Level", ["High School", "Bachelor", "Master", "PhD"])
internet_quality = st.selectbox("Internet Quality", ["Good", "Average", "Poor"], index=0) 
extracurricular = st.selectbox("Extracurricular Participation?", ["Yes", "No"])

# ==============================================================================
# 🛠️ SESSION STATE MEMORY SYSTEM
# ==============================================================================
if "calculated" not in st.session_state:
    st.session_state.calculated = False
    st.session_state.pass_pct = 0.0
    st.session_state.fail_pct = 0.0

st.markdown("---")

if st.button("Calculate Probability", type="primary"):
    st.session_state.calculated = True
    
    input_data = pd.DataFrame([{
        'age': age, 
        'study_hours_per_day': study_hours, 
        'social_media_hours': social_media,
        'netflix_hours': netflix_hours, 
        'attendance_percentage': attendance,  
        'sleep_hours': sleep_hours,
        'exercise_frequency': exercise_freq, 
        'mental_health_rating': mental_health,
        'gender': gender, 
        'part_time_job': part_time_job, 
        'diet_quality': diet_quality,
        'parental_education_level': parental_edu, 
        'internet_quality': internet_quality,
        'extracurricular_participation': extracurricular
    }])
    
    input_encoded = pd.get_dummies(input_data)
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
    probabilities = model.predict_proba(input_final)
    
    st.session_state.fail_pct = float(probabilities[0][0] * 100)
    st.session_state.pass_pct = float(probabilities[0][1] * 100)

# ==============================================================================
# 📊 DISPLAY WINDOW WITH CUSTOM RISK TIERS
# ==============================================================================
if st.session_state.calculated:
    pass_pct = st.session_state.pass_pct
    fail_pct = st.session_state.fail_pct
    
    if pass_pct >= 75.0:
        st.success("### Result: Passing (Safe Baseline)")
    elif pass_pct >= 65.0:
        st.info("### Result: Low Risk")
    elif pass_pct >= 50.0:
        st.warning("### Result: Medium Risk")
    else:
        st.error("### Result: High Risk (Likely to Fail)")
        
    st.metric(label="Success Probability", value=f"{pass_pct:.1f}% Pass")
    st.metric(label="Risk Probability", value=f"{fail_pct:.1f}% Fail")
    st.progress(int(pass_pct))
    
    st.markdown("---")
    
    # --------------------------------------------------------------------------
    # PATHWAY A: SAFE / PASSING PREMIUM SUITE (75% and Above)
    # --------------------------------------------------------------------------
    if pass_pct >= 75.0:
        st.info("💡 **Premium Optimization Available:** You qualify for zero-stress mastery resources designed to maximize your final grade boundaries beautifully without adding unnecessary academic pressure.")
        
        if st.button("✨ Unlock Success Maximizer Vault (Premium)", key="premium_pass_features"):
            st.warning("🔒 **Monetization Pending Compliance:** This premium feature is simulated for demonstration purposes pending our official business registration certificate.")
            
            st.markdown("### 💎 Premium Success Optimization Suite")
            
            st.markdown("#### 1. Academic Burnout & Fatigue Index")
            calculated_burnout = min(100, int((study_hours * 12) + (10 - sleep_hours * 5)))
            st.write(f"Current Operational Stress Index: **{calculated_burnout}/100**")
            if study_hours > 6 or sleep_hours < 5:
                st.error("⚠️ CRITICAL FATIGUE WARNING: High study density paired with compressed recovery patterns detected.")
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
            st.write(f"📱 **Automated Screen-Time Transition Tracker:** Active. Your system has designed a breakdown routine that safely scales down your logged {netflix_hours} streaming hours by 15 minutes every two days, automatically reallocates it to cognitive recovery.")
            
            st.markdown("#### 7. 'Exam-Week Peak Performance' Protocol")
            taper_data = {
                "Timeline Remaining": ["Days 7 to 5 before Exams", "Days 4 to 2 before Exams", "Day 1 before Exams"],
                "Routine Transition Strategy": [
                    "Maintain current study volumes but shift completely away from passive reading to active recall.",
                    "Reduce independent study blocks by 30%. Increase hydration and expand night rest cycles.",
                    "Halt intense conceptual study by 2:00 PM. Dedicate evening to light review and physical relaxation to peak performance."
                ]
            }
            st.table(pd.DataFrame(taper_data))

            st.markdown("#### 🗓️ 8. Your High-Performance AI Lifestyle Timetable")
            st.write("This structured schedule optimizes your high performance while protecting your free time:")
            
            passing_schedule = {
                "Daily Window": ["Morning Lecture Window", "Afternoon Focus Block", "Guilt-Free Leisure Window", "Night Maintenance Routine"],
                "Actionable High-Yield Strategy": [
                    f"Attend lectures fully to maintain your strong {attendance}% attendance baseline. Review new notes for 15 minutes immediately post-class.",
                    f"Deep work block: Spend {study_hours} highly-focused hours clearing assignments and reading ahead to stay ahead of the class curves.",
                    f"Reclaim leisure blocks: Enjoy uninhibited streaming ({netflix_hours} hrs) or social tracking. Your system efficiency leaves ample room for rest.",
                    f"Protect cognitive retention: Wind down devices 30 minutes before bed to guarantee your solid {sleep_hours}-hour neural processing window."
                ]
            }
            st.table(pd.DataFrame(passing_schedule))

    # --------------------------------------------------------------------------
    # PATHWAY B: RISK TIERS PREMIUM SUITE (Anything under 75% - LOADED EXPANSION)
    # --------------------------------------------------------------------------
    else:
        st.info(f"💡 **Premium Strategic Recovery Resource Activated:** Your profile falls within a monitored risk boundary ({pass_pct:.1f}% Passing Confidence). A hyper-targeted academic recovery suite and turnaround timetable are ready to deploy to stabilize your trajectory.")
        
        if st.button("✨ Unlock AI Academic Recovery Suite & Timetable (Premium)", key="premium_fail_features"):
            st.warning("🔒 **Monetization Pending Compliance:** This premium feature is simulated for demonstration purposes pending our official business registration certificate.")
            
            # --- SECTOR 1: ROUTINE CORRECTION ---
            st.markdown("### 🗓️ 1. Customized AI Lifestyle Recovery Roadmap")
            optimized_netflix = round(netflix_hours * 0.5, 1)
            reclaimed_hours = netflix_hours - optimized_netflix
            optimized_sleep = sleep_hours + min(2.0, reclaimed_hours)
            target_daily_study = max(5.0, study_hours + 1.5)
            
            st.markdown(f"""
            *   **Automated Screen-Time Reduction:** Digital leisure throttled from **{netflix_hours} hours** down to **{optimized_netflix} hours** daily to reclaim mental clarity.
            *   **Sleep Boundary Reset:** Sleep window recalibrated to **{optimized_sleep} hours** to repair memory consolidation deficits caused by exhaustion.
            *   **Core Study Density Scaling:** Upgraded daily core study target to a mandatory baseline of **{target_daily_study} hours** to aggressively catch up on model deficit weights.
            """)
            
            # --- SECTOR 2: PERFORMANCE TIMETABLE ---
            st.markdown("### 📋 2. Your Optimized Structural Routine")
            schedule_data = {
                "Daily Window": ["Morning Window", "Afternoon Block", "Evening Block", "Night Routine"],
                "Actionable Focus Blocks": [
                    f"Attend scheduled lectures directly to protect your {attendance}% attendance drop-off. Pack down core lecture concepts within 30 minutes of dismissal.",
                    f"First deep core study window: Dedicate {round(target_daily_study / 2, 1)} hours strictly to resolving immediate due assignments and tackling high-priority course tasks.",
                    f"Second active review window: Dedicate {round(target_daily_study / 2, 1)} hours to drilling past exam question sets, patching up weak topics, and active self-testing.",
                    f"Enforced systemic shutdown. Cut off all mobile apps and streaming platforms to lock in your optimized {optimized_sleep}-hour cognitive recovery frame."
                ]
            }
            st.table(pd.DataFrame(schedule_data))
            
            # --- SECTOR 3: NEW ACADEMIC DEFICIT TRIAGE ---
            st.markdown("### 🛠️ 3. High-Yield Syllabus Triage Strategy")
            st.write("When backing up a failing grade, trying to read everything is a trap. You must follow the **80/20 Pareto Principle Rule**:")
            
            triage_data = {
                "Priority Category": ["🚨 Priority 1: High-Weight Core Concepts", "⚠️ Priority 2: High-Yield Past Question Patterns", "📉 Priority 3: Low-Yield Reading Material"],
                "Operational Execution Plan": [
                    "Identify the 3 foundational topics that account for at least 50% of the total course marks. Master these fully before moving on.",
                    "Review the last 5 sets of past department exams. Isolate recurring questions and memorize their exact scoring rubrics.",
                    "Postpone deep readings of heavy supplementary textbook chapters until your baseline passing scores are safely secured."
                ]
            }
            st.table(pd.DataFrame(triage_data))
            
            # --- SECTOR 4: EMERGENCY MATRIX ---
            st.markdown("### 🏁 4. Emergency Grade Recovery Matrix")
            st.write("Deploy these tactical operational protocols immediately to systematically secure missing grade boundaries:")
            
            st.markdown("""
            *   **The 15-Minute Lecture Audit:** Before every lecture, spend exactly 15 minutes reviewing the previous week's lecture slides. This constructs an immediate contextual anchor in your brain, preventing confusion during the live session.
            *   **Continuous Assessment (IA) Buffer Capture:** Treat all mini-quizzes, lab assessments, and interim tests as uncompromisabled milestones. Securing a strong internal assessment mark drastically lowers the pressure required during final exam sittings.
            *   **The Active Recall Feedback Loop:** Never reread chapters passively. Close the book, take a blank sheet of paper, write down every concept you can remember from your study block in red ink, and open the book only to verify missing details.
            """)
