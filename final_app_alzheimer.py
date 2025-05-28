import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64 # for image display 
from scipy import stats
from fpdf import FPDF # report download
from sklearn.base import BaseEstimator, TransformerMixin
import shap
import re
import scipy.special

# ---- Header & Branding ---
st.set_page_config(
    page_title="Alzheimer's Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- Sidebar for Info ----
    
with st.sidebar:
    st.markdown("## 🎯 Project Goal")
    st.markdown("This project supports **Sustainable Development Goal 3**: _Good Health and Well-Being_.")
    st.markdown("It focuses on the **early detection of Alzheimer’s disease**, addressing a major public health challenge through data-driven prediction.")
    st.info("ℹ️ **About This App**\n\nBuilt to raise awareness of Alzheimer’s risks. No information is stored.")
    st.markdown("---")
    st.markdown("### 👥 Contact Us")
    st.markdown("#### WQD7006/7012 OCC2 Group 5 Team")
    st.markdown("""
    - HAU JIA QI
    - LOH JING WEI  
    - GOH WEE KEN  
    - HO WEI WEN
    - CHEONG MENG BEN
    """)
    st.markdown("[View GitHub Repo](https://github.com/sharein0305/alzheimer_G5)", unsafe_allow_html=True)

# ---- Styling ----
st.markdown(
    """
    <style>
    .section-header { font-size:2rem; font-weight:600; color:#4f8bf9; margin-top:2em;}
    .risk-badge { font-size:1.4rem; padding:0.5em 1em; border-radius:999px;}
    </style>
    """, unsafe_allow_html=True)


# ---- Main Content ----
st.title("Alzheimer's Disease Risk Prediction")

st.markdown(
    "Answer the following questions to estimate your risk for Alzheimer's disease based on clinically relevant factors. "
    "**Please note:** This tool is for informational purposes only and does not constitute a clinical diagnosis."
)
st.markdown(
    "**If you're over 60 years of age**, or experiencing memory-related symptoms, it's recommended to complete this screening to support early detection efforts."
)


# ---- Image Display ----
try:
    with open("2103.i003.001_dementia_alzheimer_illustration.jpg", "rb") as file:
        encoded_image = file.read()
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <img src='data:image/jpeg;base64,{base64.b64encode(encoded_image).decode()}' width='300'/>
        </div>
        """,
        unsafe_allow_html=True
    )
except:
    st.markdown(
        """
        <div style='text-align: center;'>
            <img src='https://i.imgur.com/Z52e6mA.jpeg' width='300'/>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("#### 💡 Why This Matters")
st.markdown("""
**Sustainable Development Goal 3 (Good Health and Well-Being)** aims to ensure healthy lives and promote well-being for all. 

One pressing issue under SDG 3 is the **early diagnosis of neurodegenerative diseases**, such as Alzheimer’s. Despite growing awareness, early detection and preventive strategies for Alzheimer’s remain limited, often leading to diagnoses when treatment is less effective.
""")

# Data Preprocessing

num_mean_features = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL']
num_median_features = ['SystolicBP', 'DiastolicBP']
num_mode_features = ['Gender', 'EducationLevel', 'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness']
categorical_features = ['Ethnicity']

all_features_order = num_mean_features + num_median_features + num_mode_features + categorical_features

def to_dataframe(X):
    """Convert numpy array back to DataFrame with correct column names."""
    return pd.DataFrame(X, columns=all_features_order)

symptom = ["MemoryComplaints", "BehavioralProblems", "Confusion",
                "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"]

comorbid_cols = ['Hypertension', 'Diabetes', 'CardiovascularDisease', 'Depression', 'HeadInjury']

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy() # To avoid changing the original DataFrame
        X['AgeGroup'] = pd.cut(X['Age'], bins=[59, 70, 80, 90], labels=['60-69', '70-79', '80-90'])
        X['AnySymptom'] = X[symptom].sum(axis=1) > 0
        X['AnySymptom'] = X['AnySymptom'].astype(int)
        X['ComorbidityCount'] = X[comorbid_cols].sum(axis=1)
        X['LifestyleRisk'] = (
            X['Smoking'] + 
            (X['AlcoholConsumption'] > 14).astype(int) +
            (X['PhysicalActivity'] < 2.5).astype(int) +
            (X['DietQuality'] < 4).astype(int) +
            (X['SleepQuality'] < 6).astype(int)
        )
        return X
    
# ---- Load Model and Data ----
@st.cache_resource
def load_pipeline():
    return joblib.load("final_pipeline_alzheimer.pkl")
pipeline = load_pipeline()

@st.cache_data
def load_train_df():
    return pd.read_csv('alzheimers_disease_data.csv')
train_df = load_train_df()

# ---- Organize User Inputs (Questionnaire with Explanations) ----
with st.form("user_inputs"):
    # 1. Demographic Details
    st.markdown('<div class="section-header">1. Demographic Details</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 60, 90, 75, help="Alzheimer's risk increases with age.")
        ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])      
    with col2:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        # Add a spacer using markdown or an empty element
        st.markdown("")  # Spacer for better alignment
        education = st.selectbox("Education Level", ["None", "High School", "Bachelor's", "Higher"], help="Higher education is associated with lower risk.")

    
    # 2. Lifestyle
    st.markdown('<div class="section-header">2. Lifestyle</div>', unsafe_allow_html=True)
    st.markdown("Adjust the sliders based on your current lifestyle habits.  \nThese factors play a key role in brain health and Alzheimer’s risk.")

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("💡 Need help calculating your BMI?"):
            weight = st.number_input("Enter your weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            height = st.number_input("Enter your height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.1)
            calculated_bmi = weight / ((height / 100) ** 2)
            st.markdown(f"**Your calculated BMI is:** {calculated_bmi:.1f}")
        bmi = st.slider("BMI (Body Mass Index)", 15.0, 50.0, calculated_bmi, step=0.1,
                    help="Body Mass Index = weight in kg / (height in m)^2")
        diet = st.slider("Diet Quality (0 = poor, 10 = excellent)", 0, 10, 5, help="A healthy diet is protective against cognitive decline.")
        
        with st.expander("💡 Need help rating your diet?"):
                st.markdown("""
<div style="font-size: 0.85rem;">
        Rate your diet from 0 (poor) to 10 (excellent) based on:
        <ul>
            <li>Regular intake of fruits, vegetables, whole grains, lean protein, and healthy fats.</li>
            <li>Limited processed foods, added sugar, and saturated fat.</li>
        </ul>
        Helpful guide:
        <ul>
            <li><a href="https://www.hsph.harvard.edu/nutritionsource/healthy-eating-plate/" target="_blank">Harvard Healthy Eating Plate</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
        
    
    with col2:
        smoking = st.radio( "Do you currently smoke?", ["No", "Yes"], help="Smoking is a known risk factor for cognitive decline.")
        alcohol = st.slider(
        "Alcohol Consumption (units/week)", 0, 20, 1, help="Enter your average weekly alcohol consumption in units (0–20). Higher intake is linked to increased risk.")
        physical = st.slider("Physical Activity (hours/week)", 0.0, 10.0, 5.0, step=0.5, help="More activity is protective against Alzheimer's.")
        sleep = st.slider("Sleep Quality (4 = poor, 10 = excellent)", 4, 10, 7, help="Poor sleep can contribute to memory issues and higher dementia risk.")
        


    # 3. Medical History
    st.markdown('<div class="section-header">3. Medical History</div>', unsafe_allow_html=True)
    st.markdown(
    "Review your medical history.  \nUnderstanding these conditions helps personalize your assessment and can highlight areas where lifestyle or medical care may support brain health.")

    col1, col2 = st.columns(2)    
    with col1:
        family_history = st.radio("Family History of Alzheimer's?", ["No", "Yes"], help="Having a parent or sibling with Alzheimer's increases your risk.")
        diabetes = st.radio("Diabetes?", ["No", "Yes"], help="Diabetes can increase the risk of cognitive decline.")
        head_injury = st.radio("History of Head Injury?", ["No", "Yes"], help="Past head injuries may increase Alzheimer’s risk.")

    with col2:
        cardiovascular = st.radio("Cardiovascular Disease?", ["No", "Yes"], help="Heart conditions like stroke or heart attack are linked to higher risk.")
        depression = st.radio("Depression?", ["No", "Yes"], help="Long-term depression is associated with increased Alzheimer's risk.")
        hypertension = st.radio("Hypertension?", ["No", "Yes"], help="High blood pressure, especially in midlife, increases Alzheimer’s risk.")

    
    # 4. Clinical Measurements
    st.markdown('<div class="section-header">4. Clinical Measurements</div>', unsafe_allow_html=True)
    st.markdown(
    "Review your medical history.  \nUnderstanding these conditions helps personalize your assessment and can highlight areas where lifestyle or medical care may support brain health."
)

    col1, col2 = st.columns(2)
    with col1:
        systolic = st.slider("Systolic Blood Pressure (mmHg)", 90, 180, 120, help="Top number in a blood pressure reading. High systolic pressure (>130 mmHg) is linked to increased dementia risk.")
        chol_total = st.slider("Total Cholesterol (mg/dL)", 150, 300, 200, help="Sum of HDL, LDL, and triglycerides. High total cholesterol can increase dementia risk, especially in midlife.")
        chol_hdl = st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50, help="High-density lipoprotein (HDL) helps remove bad cholesterol. Higher HDL levels are protective.")

    with col2:
        diastolic = st.slider("Diastolic Blood Pressure (mmHg)", 60, 120, 80, help="Bottom number in a blood pressure reading. Elevated diastolic pressure can strain blood vessels over time.")
        chol_ldl = st.slider("LDL Cholesterol (mg/dL)", 50, 200, 100, help="Low-density lipoprotein (LDL) is the 'bad' cholesterol. High levels (>130 mg/dL) are linked to heart and brain disease.")
        chol_trig = st.slider("Triglycerides (mg/dL)", 50, 400, 120,help="Type of fat in the blood. High levels can increase risk of cardiovascular issues, indirectly impacting brain health.")



    # 5. Cognitive and Functional Assessments
    st.markdown('<div class="section-header">5. Cognitive and Functional Assessments</div>', unsafe_allow_html=True)
    st.markdown("These assessments help understand your current cognitive and daily functioning.  \nLower scores may indicate areas where support is needed.")
    with st.expander("ℹ️ Tip: Need help answering?"):
        st.markdown(
        "<span style='font-size: 12px;'>If you're unsure about any of the answers, consider asking a close family member or caregiver. They may remember details or observe symptoms that you might overlook.</span>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        mmse = st.slider("MMSE Score", 0, 30, 28, help="The Mini-Mental State Examination assesses cognitive function. Lower scores may suggest cognitive impairment.")
        func_assess = st.slider("Functional Assessment (0–10)", 0, 10, 8, help="Assesses ability to function independently. Lower scores suggest more difficulty.")
        adl = st.slider("Daily Living Activities (0–10)", 0, 10, 5,
 help="Measures ability to manage daily activities. Higher scores reflect more independence.")

    with col2:
        memory_complaints = st.radio("Memory Complaints", ["No", "Yes"], help="Have you noticed recent memory challenges?")
        behavioral = st.radio("Behavioral Changes", ["No", "Yes"], help="Includes mood swings, irritability, or unusual behaviors.")


    # 6. Symptoms
    st.markdown('<div class="section-header">6. Symptoms</div>', unsafe_allow_html=True)
    st.markdown("These are common signs associated with cognitive decline.  \nIt's helpful to track their frequency or presence.")
    with st.expander("ℹ️ Tip: Need help answering?"):
        st.markdown(
        "<span style='font-size: 12px;'>If you're unsure about any of the answers, consider asking a close family member or caregiver. They may remember details or observe symptoms that you might overlook.</span>",
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)
    with col1:
        confusion = st.radio("Frequent Confusion", ["No", "Yes"], help="Includes losing track of time, place, or events.")
        disorientation = st.radio("Disorientation", ["No", "Yes"], help="Getting lost in familiar places or forgetting the current date.")

    with col2:
        personality = st.radio("Personality Changes", ["No", "Yes"], help="Noticeable changes in personality or behavior.")
        difficulty_tasks = st.radio("Task Difficulty", ["No", "Yes"], help="Difficulty performing familiar tasks or following instructions.")
        forgetfulness = st.radio("Forgetfulness", ["No", "Yes"], help="Forgetting recent events or repeating questions.")
    


    submit = st.form_submit_button(label="Predict My Risk")

if submit:
    # ---- Prepare input for model (convert categorical to numeric as needed!) ----
    user_dict = {
        # 1. Demographic Details
        "Age": age,
        "Gender": 0 if gender == 'Male' else 1,
        "Ethnicity": ["Caucasian", "African American", "Asian", "Other"].index(ethnicity),
        "EducationLevel": ["None", "High School", "Bachelor's", "Higher"].index(education),
        # 2. Lifestyle
        "BMI": bmi,
        "Smoking": 0 if smoking == 'No' else 1,
        "AlcoholConsumption": alcohol,
        "PhysicalActivity": physical,
        "DietQuality": diet,
        "SleepQuality": sleep,
        # 3. Medical History
        "FamilyHistoryAlzheimers": 0 if family_history == 'No' else 1,
        "CardiovascularDisease": 0 if cardiovascular == 'No' else 1,
        "Diabetes": 0 if diabetes == 'No' else 1,
        "Depression": 0 if depression == 'No' else 1,
        "HeadInjury": 0 if head_injury == 'No' else 1,
        "Hypertension": 0 if hypertension == 'No' else 1,
        # 4. Clinical Measurements
        "SystolicBP": systolic,
        "DiastolicBP": diastolic,
        "CholesterolTotal": chol_total,
        "CholesterolLDL": chol_ldl,
        "CholesterolHDL": chol_hdl,
        "CholesterolTriglycerides": chol_trig,
        # 5. Cognitive and Functional Assessments
        "MMSE": mmse,
        "FunctionalAssessment": func_assess,
        "MemoryComplaints": 0 if memory_complaints == 'No' else 1,
        "BehavioralProblems": 0 if behavioral == 'No' else 1,
        "ADL": adl,
        # 6. Symptoms
        "Confusion": 0 if confusion == 'No' else 1,
        "Disorientation": 0 if disorientation == 'No' else 1,
        "PersonalityChanges": 0 if personality == 'No' else 1,
        "DifficultyCompletingTasks": 0 if difficulty_tasks == 'No' else 1,
        "Forgetfulness": 0 if forgetfulness == 'No' else 1
    }

    # -- Prepare Input and Features --
    input_df = pd.DataFrame([user_dict])

    fe = FeatureEngineer()
    train_fe = fe.transform(train_df)
    user_fe = fe.transform(input_df)

    # -- Make Prediction --
    prob = pipeline.predict_proba(input_df)[0][1]
    prediction = pipeline.predict(input_df)[0]
    prob_percent = f"{prob:.1%}"

    # -- Display Result Section --
    st.markdown('<div class="section-header" style="font-size: 1.4em; margin-bottom: 0.5em;">📊 Here\'s Your Result</div>', unsafe_allow_html=True)
    st.success(f"**Predicted Probability of Alzheimer's Disease:** {prob_percent}")

    # Badge Styling
    badge_color = "#ec7063" if prediction else "#58d68d"
    badge_text = "At Risk" if prediction else "Low Risk"
    st.markdown(f"""
        <div style="
            background:{badge_color}; 
            color:#fff; 
            padding: 0.4em 0.9em; 
            border-radius: 0.4em; 
            font-weight:600; 
            display:inline-block; 
            font-size: 1.05em;">
            {badge_text}
        </div>
    """, unsafe_allow_html=True)

    # -- Risk Probability Donut Chart using Plotly --
    fig = go.Figure(go.Pie(
        values=[prob, 1 - prob],
        labels=["Risk", "No Risk"],
        marker_colors=["#e74c3c", "#2ecc71"],
        hole=0.6,
        textinfo='label+percent'
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        width=260,
        height=260
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.plotly_chart(fig, use_container_width=True)


    # -- Percentile Comparison Section --
    st.markdown("### 🧮 How Do You Compare?")

    # Calculate percentile
    user_risk = prob
    all_risks = pipeline.predict_proba(train_df)[:, 1]
    percentile_rank = (all_risks < user_risk).sum() / len(all_risks)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Your Risk Percentile", value=f"{percentile_rank*100:.0f}th")
        st.markdown("Your risk score is higher than this percentage of individuals in our dataset. This reflects how your result compares to others, but it does not indicate whether your score is high or low on its own.")


    with col2:
        st.metric(label="Avg Dataset Risk", value=f"{all_risks.mean():.1%}")
        st.markdown("This is the average Alzheimer's risk across the dataset.")

  # -- Risk Reduction Tip --
    st.markdown("""
        <div style="margin-top: 1.5em; font-size: 0.95em; color: #444;">
            🔍 <strong>Tip:</strong> Staying physically active, eating a healthy diet, and engaging in regular mental exercises can help reduce the risk of Alzheimer's. 
            <a href="https://www.alz.org/alzheimers-dementia/facts-figures" target="_blank">Learn more</a>.
        </div>
    """, unsafe_allow_html=True)
    
    # --- User Profile Context & Comparison ---
    st.markdown("### 📚 Additional Insights Based on Your Profile")
    age_group = user_fe['AgeGroup'].iloc[0]
    group_count = train_fe['AgeGroup'].value_counts().get(age_group, 0)

    # Highlighted Age Group Card
    st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #4f8bf9 0%, #58d68d 100%);
            border-radius: 1em;
            padding: 1.2em 1em;
            margin: 1.5em 0;
            color: white;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 600;">
            🎂 You belong to the <span style='color:#ffe066;'>{age_group}</span> age group!<br>
            <span style='font-size:1.1rem; font-weight:400;'>There are <b>{group_count}</b> people in this age group in our dataset.</span>
        </div>
    """, unsafe_allow_html=True)

    # Fun Fact by Age Group
    age_group_facts = {
        '60-69': "<b>Did you know?</b> Many people in their 60s can boost brain health by <span style='color:#786312;'>learning new skills or languages</span>!",
        '70-79': "<b>Did you know?</b> <span style='color:#786312;'>Staying socially active</span> in your 70s is linked to a sharper mind.",
        '80-90': "<b>Did you know?</b> Regular <span style='color:#786312;'>light exercise—even walking</span>—can help maintain cognitive function in your 80s!"
    }
    fact = age_group_facts.get(str(age_group), "<b>Did you know?</b> Every age group can benefit from <span style='color:#786312;'>brain-healthy habits</span>!")

    st.markdown(f"""
        <div style="
            background: #fff3cd;
            border-left: 8px solid #ffe066;
            border-radius: 0.7em;
            padding: 1em 1.2em;
            margin-bottom: 1.5em;
            color: #7d6608;
            font-size: 1.1rem;">
            🧠 {fact}
        </div>
    """, unsafe_allow_html=True)


    # ---- Data Comparison (Enhanced, beautiful display) ----
    st.markdown('<div class="section-header">📊How You Compare with Others</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<b>Age Distribution</b>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(4,2.5))
        ax2.hist(train_df['Age'], alpha=0.6, label='Dataset', bins=15, color='#4f8bf9')
        ax2.axvline(age, color='red', linestyle='dashed', linewidth=2, label='You')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Count')
        ax2.legend()
        st.pyplot(fig2)
        user_age_percentile = (train_df['Age'] < age).mean() * 100
        st.caption(f"You are older than {user_age_percentile:.0f}% of people in the dataset.")
    with col2:
        st.markdown("<b>MMSE Score Distribution</b>", unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(4,2.5))
        ax3.hist(train_df['MMSE'], alpha=0.6, label='Dataset', bins=15, color='#58d68d')
        ax3.axvline(mmse, color='orange', linestyle='dashed', linewidth=2, label='You')
        ax3.set_xlabel('MMSE Score')
        ax3.set_ylabel('Count')
        ax3.legend()
        st.pyplot(fig3)
        user_mmse_percentile = (train_df['MMSE'] < mmse).mean() * 100
        st.caption(f"Your MMSE score is higher than {user_mmse_percentile:.0f}% of people in the dataset.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<b>Comorbidity Count</b>", unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(4,2.5))
        ax4.hist(train_fe['ComorbidityCount'], bins=range(0, train_fe['ComorbidityCount'].max()+2), alpha=0.6, color='#f7ca18', label='Dataset')
        ax4.axvline(user_fe['ComorbidityCount'].iloc[0], color='purple', linestyle='dashed', linewidth=2, label='You')
        ax4.set_xlabel('Number of Comorbidities')
        ax4.set_ylabel('Count')
        ax4.legend()
        st.pyplot(fig4)
        st.caption(f"You have {user_fe['ComorbidityCount'].iloc[0]} comorbid condition(s). More comorbidities can increase risk.")
    with col4:
        st.markdown("<b>Lifestyle Risk Score</b>", unsafe_allow_html=True)
        fig5, ax5 = plt.subplots(figsize=(4,2.5))
        ax5.hist(train_fe['LifestyleRisk'], bins=range(0, train_fe['LifestyleRisk'].max()+2), alpha=0.6, color='#e67e22', label='Dataset')
        ax5.axvline(user_fe['LifestyleRisk'].iloc[0], color='green', linestyle='dashed', linewidth=2, label='You')
        ax5.set_xlabel('Lifestyle Risk Score')
        ax5.set_ylabel('Count')
        ax5.legend()
        st.pyplot(fig5)
        st.caption(f"Your lifestyle risk score is {user_fe['LifestyleRisk'].iloc[0]}. Lower is better!")



    # --- SHAP Model Feature Importance (global, for all data) ---
    # with st.expander("🔬 Model Explainability: See Which Features Most Influence The Algorithm"):
    #     st.markdown("""
    #         <b>Which features most increased or decreased risk across everyone?</b><br>
    #         The chart below sorts features so the top rows are the most powerful overall.<br>
    #         <i>(Red: increases risk, Blue: reduces risk.)</i>
    #     """, unsafe_allow_html=True)
    #     try:
    #         X_train = train_df.copy()
    #         X_proc = pipeline[:-1].transform(X_train)
    #         model = pipeline.named_steps['classifier']
    #         preprocessor = pipeline.named_steps.get('preprocessor', None) or pipeline.named_steps.get('transformer', None)
    #         if hasattr(preprocessor, 'get_feature_names_out'):
    #             feature_names = preprocessor.get_feature_names_out()
    #         else:
    #             feature_names = all_features_order
    #         exp = shap.Explainer(model, X_proc)
    #         shap_values = exp(X_proc)
    #         fig, ax = plt.subplots(figsize=(8, 6))
            
    #         feature_names_clean = [re.sub(r'^(mean_num__|mode_num__|median_num__|cat__|num__)?', '', str(f)) for f in feature_names]
    #         shap.summary_plot(shap_values, X_proc, feature_names=feature_names_clean, show=False, plot_type="bar")

    #         # shap.summary_plot(shap_values, X_proc, feature_names=feature_names, show=False, plot_type="bar")
    #         st.pyplot(fig)
    #     except Exception as e:
    #         st.error(f"Could not display SHAP plot. Reason: {e}")

    # --- SHAP Personal Impact Table/Chart (per-user percent impact) ---
    with st.expander("🔬 See How Your Answers Affect Your Risk"):
        st.markdown(
            "<b>Impact of each feature on your risk (% points):</b><br>"
            "Shows how much each input increased (red) or decreased (blue) your Alzheimer's risk compared to the average.<br>"
            "<i>Table sorted: biggest effect at the top.</i>",
            unsafe_allow_html=True,
        )
        
        try:
            preprocessor = pipeline.named_steps.get('preprocessor', None)
            model = pipeline.named_steps['classifier']

            user_fe = fe.transform(input_df)  
            
            if preprocessor:
                user_proc = preprocessor.transform(user_fe)
                feature_names = (
                    preprocessor.get_feature_names_out()
                    if hasattr(preprocessor, 'get_feature_names_out')
                    else user_fe.columns
                )
            else:
                user_proc = user_fe.values
                feature_names = user_fe.columns


            # Use background from training set, sample for speed
            bg_proc = pipeline[:-1].transform(train_df.sample(min(200, len(train_df)), random_state=42))

            explainer = shap.Explainer(model, bg_proc)
            shap_vals_user = explainer(user_proc)

            base_value = shap_vals_user.base_values[0]
            shap_contributions = shap_vals_user.values[0]
            logit = base_value + shap_contributions.sum()
            pred_prob = scipy.special.expit(logit)

            # Compute impact per feature
            impacts = []
            for i, (f, shapval) in enumerate(zip(feature_names, shap_contributions)):
                logit_without = logit - shapval
                prob_without = scipy.special.expit(logit_without)
                delta = (pred_prob - prob_without) * 100
                
                impacts.append(dict(Feature=re.sub(r'^(mean_num__|mode_num__|median_num__|cat__|num__)?', '', str(f)), Impact=delta))
                # impacts.append(dict(Feature=f, Impact=delta))
            impacts = sorted(impacts, key=lambda x: abs(x['Impact']), reverse=True)
            df_impacts = pd.DataFrame(impacts)
            df_impacts["Effect"] = df_impacts["Impact"].apply(lambda x: "↑" if x > 0 else "↓")
            df_impacts["Impact"] = df_impacts["Impact"].map(lambda x: f"{x:+.2f}%")
            df_impacts = df_impacts.rename(columns={"Feature": "Feature", "Impact": "Change in Your Risk", "Effect": "Effect"})
            st.dataframe(df_impacts.set_index('Feature'), height=360)
            st.caption("↑ means this answer pushed your risk higher; ↓ means it reduced your risk.")

            # Optional top N chart
            top_n = 10
            fig, ax = plt.subplots(figsize=(7, top_n//2 + 1))
            ylabels = [f"{row['Feature']} {row['Effect']}" for _, row in df_impacts.head(top_n).iterrows()]
            values = [float(row['Change in Your Risk'].replace('%','')) for _, row in df_impacts.head(top_n).iterrows()]
            colors = ['#e74c3c' if "↑" in row['Effect'] else '#3498db' for _, row in df_impacts.head(top_n).iterrows()]
            ax.barh(ylabels, values, color=colors)
            ax.axvline(0, color='k', lw=1)
            ax.set_xlabel('Change in Your Predicted Risk (%)')
            ax.set_title('Top Features Impacting Your Risk')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not display personalized SHAP impact. Reason: {e}")

            
    # Any symptom
    if user_fe['AnySymptom'].iloc[0]:
        st.warning("You reported at least one cognitive symptom. Early detection and intervention are important—consider discussing with a healthcare provider.")
    else:
        st.success("You reported no major cognitive symptoms. Keep monitoring your brain health!")


# Next steps
    def render_next_steps_section(prediction_prob: float):
        st.subheader("🧠 What You Can Do Next")

    # Risk messaging based on prediction probability
        if prediction_prob < 0.2:
            st.success("✅ You are in the **Safe Zone**.")
            st.markdown("Your results suggest a very low risk of Alzheimer. Still, it's important to keep your brain healthy over time. Here are a few suggestions to stay on track:")
        elif prediction_prob < 0.5:
            st.warning("⚠️ You are at **Moderate Risk**.")
            st.markdown("Your result shows some potential risk. Now is a great time to start building stronger brain-healthy habits. Consider the following actions:")
        else:
            st.error("🔴 You are at **High Risk**.")
            st.markdown("This prediction indicates an elevated risk. Don’t panic — risk is not destiny. Early actions can help make a difference. Start here:")

    # Actionable steps
        st.markdown("""
            - 🥦 **Adopt a brain-healthy diet**: Eat more leafy greens, berries, nuts, and fatty fish.
            - 🚶‍♀️ **Exercise regularly**: 30+ minutes of activity most days helps brain function.
            - 🧘‍♂️ **Reduce stress**: Use mindfulness, breathing exercises, or yoga.
            - 🧩 **Challenge your brain**: Try puzzles, reading, or learning a new skill.
            - 😴 **Improve your sleep**: Get 7–9 hours per night. Avoid screens before bed.
            - 💃 **Stay connected**: Social interaction helps protect cognitive function.
            - 🚭 **Avoid smoking and limit alcohol**: Both increase Alzheimer risk.
    """)

        if prediction_prob >= 0.5:
            st.markdown("🩺 **Next step**: Please consult a healthcare provider about your result and ask about memory assessments or brain health screening.")


# Report Generation
    def generate_pdf(probability):
        # Create PDF object
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

    # Content
        pdf.cell(200, 10, txt="Alzheimer Risk Report", ln=True, align="C")
        pdf.ln(10)
        pdf.multi_cell(0, 10, f"Based on your assessment, the predicted probability of developing Alzheimer disease is {probability:.1%}.")

        if probability >= 0.7:
            level = "High"
            action = "We recommend seeing a healthcare professional and taking immediate lifestyle steps."
        elif probability >= 0.4:
            level = "Moderate"
            action = "You may want to adopt more healthy habits and monitor changes closely."
        else:
            level = "Low"
            action = "Continue your current healthy habits, and monitor regularly."

        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Risk Level: {level}")
        pdf.multi_cell(0, 10, f"Recommended Actions: {action}")

        # Save to bytes
        pdf_output = pdf.output(dest='S').encode('latin-1')
        return pdf_output

    # Download Button
    def render_download_button(probability):
        pdf_bytes = generate_pdf(probability)
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="alzheimer_risk_report.pdf">📄 Download Full Risk Report</a>'
        st.markdown(href, unsafe_allow_html=True)

    render_next_steps_section(prob)
    render_download_button(prob)

# Feedback
    st.subheader("💬 We'd Love Your Feedback")

    st.markdown("""
    <div style="font-size: 0.9rem;"> 
    Help us improve this app! If you have suggestions, found something unclear, or want to share your thoughts:<br><br> 
    👉 <a href="https://forms.gle/JtAcXTvkpoWwLgWi6" target="_blank">Give Feedback Here</a>  
    <hr>
    🔄 <strong>Come Back Anytime</strong>: Reassess your risk after lifestyle changes, or share the app with others who might benefit.
    </div> 
    """, unsafe_allow_html=True)

# ---- Footer ----
st.markdown("--\n:lock: *No personal data is stored. For informational purposes only. Consult a healthcare provider for medical advice.*")
