import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="MediGuide Pro",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ---------------------- Session State Management ----------------------
if 'predict' not in st.session_state:
    st.session_state.predict = False
if 'symptoms' not in st.session_state:
    st.session_state.symptoms = []

# ---------------------- Theme-Compatible CSS ----------------------
st.markdown("""
    <style>
    /* Base theme-adaptive styles */
    .main {background-color: var(--background-color);}
    
    .symptom-pill {
        display: inline-block;
        padding: 8px 20px;
        margin: 5px;
        background: color-mix(in srgb, var(--primary-color) 10%, transparent);
        border-radius: 25px;
        color: var(--primary-color);
        font-size: 0.9em;
        transition: all 0.3s ease;
    }
    
    .report-card {
        padding: 25px;
        background: var(--background-color);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 20px 0;
        border: 1px solid var(--secondary-background-color);
    }
    
    .recommendation-list {
        padding-left: 20px;
        margin: 0;
        color: var(--text-color);
    }
    
    .recommendation-list li {
        margin: 12px 0;
        padding-left: 10px;
        line-height: 1.5;
    }
    
    .emergency-alert {
        border-left: 6px solid #dc3545;
        background: color-mix(in srgb, #dc3545 10%, transparent);
    }
    
    @media (max-width: 768px) {
        .stColumn {padding: 5px !important;}
        .report-card {padding: 15px;}
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------- Data Loading ----------------------
@st.cache_data
def load_data():
    with open("svc.pkl", "rb") as file:
        model = pickle.load(file)
    
    return (
        model,
        pd.read_csv("description.csv"),
        pd.read_csv("precautions_df.csv"),
        pd.read_csv("medications.csv"),
        pd.read_csv("diets.csv"),
        pd.read_csv("workout_df.csv")
    )

model, disease_description, precautions, medications, diet, workout_df = load_data()

# ---------------------- Data Processing Functions ----------------------
def clean_recommendation(item):
    """Clean and format recommendation items"""
    # Remove special characters and extra spaces
    item = re.sub(r"[\[\]'\"\\]", "", str(item))
    # Remove leading/trailing commas and whitespace
    item = re.sub(r"^,+|,+$", "", item).strip()
    # Remove extra commas between items
    return re.sub(r",{2,}", ",", item)

def process_recommendations(df, disease):
    """Process recommendations with proper cleaning and splitting"""
    try:
        items = df[df["Disease"] == disease].iloc[:, 1:] \
                  .dropna(axis=1) \
                  .values.flatten() \
                  .tolist()
        
        processed = []
        for item in items:
            # Clean and split comma-separated values
            cleaned = clean_recommendation(item)
            if cleaned:
                processed.extend([i.strip() for i in cleaned.split(",") if i.strip()])
        
        return processed if processed else ["No recommendations available"]
    except Exception as e:
        return ["Recommendations not available"]


# ---------------------- Dictionaries ----------------------
diseases_list = {
    15:'Fungal infection',
    4: 'Allergy',
    16: 'GERD',
    9: 'Chronic cholesterol',
    14: 'Drug Reaction',
    33: 'Peptic ulcer disease',
    1: 'AIDS',
    12: 'Diabetes',
    17: 'Gastroenteritis',
    6: 'Bronchial Asthma',
    23: 'Hypertension',
    30: 'Migraine',
    7: 'Cervical spondylosis',
    32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice',
    29: 'Malaria',
    8: 'Chicken pox',
    11: 'Dengue',
    37: 'Typhoid',
    40: 'Hepatitis A',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    3: 'Alcoholic hepatitis',
    36: 'Tuberculosis',
    10: 'Common Cold',
    34: 'Pneumonia',
    13: 'Dimorphic hemorrhoids (piles)',
    18: 'Heart attack',
    39: 'Varicose veins',
    26: 'Hypothyroidism',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    31: 'Osteoarthritis',
    5: 'Arthritis',
    0: '(vertigo) Paroxysmal Positional Vertigo',
    2: 'Acne',
    38: 'Urinary tract infection',
    35: 'Psoriasis',
    27: 'Impetigo'
}

symptoms_dict = {
     'itching': 0,
    'skin_rash': 1,
    'nodal_skin_eruptions': 2,
    'continuous_sneezing': 3,
    'shivering': 4,
    'chills': 5,
    'joint_pain': 6,
    'stomach_pain': 7,
    'acidity': 8,
    'ulcers_on_tongue': 9,
    'muscle_wasting': 10,
    'vomiting': 11,
    'burning_micturition': 12,
    'spotting_urination': 13,
    'fatigue': 14,
    'weight_gain': 15,
    'anxiety': 16,
    'cold_hands_and_feets': 17,
    'mood_swings': 18,
    'weight_loss': 19,
    'restlessness': 20,
    'lethargy': 21,
    'patches_in_throat': 22,
    'irregular_sugar_level': 23,
    'cough': 24,
    'high_fever': 25,
    'sunken_eyes': 26,
    'breathlessness': 27,
    'sweating': 28,
    'dehydration': 29,
    'indigestion': 30,
    'headache': 31,
    'yellowish_skin': 32,
    'dark_urine': 33,
    'nausea': 34,
    'loss_of_appetite': 35,
    'pain_behind_the_eyes': 36,
    'back_pain': 37,
    'constipation': 38,
    'abdominal_pain': 39,
    'diarrhoea': 40,
    'mild_fever': 41,
    'yellow_urine': 42,
    'yellowing_of_eyes': 43,
    'acute_liver_failure': 44,
    'fluid_overload': 45,
    'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47,
    'malaise': 48,
    'blurred_and_distorted_vision': 49,
    'phlegm': 50,
    'throat_irritation': 51,
    'redness_of_eyes': 52,
    'sinus_pressure': 53,
    'runny_nose': 54,
    'congestion': 55,
    'chest_pain': 56,
    'weakness_in_limbs': 57,
    'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60,
    'bloody_stool': 61,
    'irritation_in_anus': 62,
    'neck_pain': 63,
    'dizziness': 64,
    'cramps': 65,
    'bruising': 66,
    'obesity': 67,
    'swollen_legs': 68,
    'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70,
    'enlarged_thyroid': 71,
    'brittle_nails': 72,
    'swollen_extremeties': 73,
    'excessive_hunger': 74,
    'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76,
    'slurred_speech': 77,
    'knee_pain': 78,
    'hip_joint_pain': 79,
    'muscle_weakness': 80,
    'stiff_neck': 81,
    'swelling_joints': 82,
    'movement_stiffness': 83,
    'spinning_movements': 84,
    'loss_of_balance': 85,
    'unsteadiness': 86,
    'weakness_of_one_body_side': 87,
    'loss_of_smell': 88,
    'bladder_discomfort': 89,
    'foul_smell_of_urine': 90,
    'continuous_feel_of_urine': 91,
    'passage_of_gases': 92,
    'internal_itching': 93,
    'toxic_look_(typhos)': 94,
    'depression': 95,
    'irritability': 96,
    'muscle_pain': 97,
    'altered_sensorium': 98,
    'red_spots_over_body': 99,
    'belly_pain': 100,
    'abnormal_menstruation': 101,
    'dischromic_patches': 102,
    'watering_from_eyes': 103,
    'increased_appetite': 104,
    'polyuria': 105,
    'family_history': 106,
    'mucoid_sputum': 107,
    'rusty_sputum': 108,
    'lack_of_concentration': 109,
    'visual_disturbances': 110,
    'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112,
    'coma': 113,
    'stomach_bleeding': 114,
    'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116,
    'fluid_overload.1': 117,
    'blood_in_sputum': 118,
    'prominent_veins_on_calf': 119,
    'palpitations': 120,
    'painful_walking': 121,
    'pus_filled_pimples': 122,
    'blackheads': 123,
    'scurring': 124,
    'skin_peeling': 125,
    'silver_like_dusting': 126,
    'small_dents_in_nails': 127,
    'inflammatory_nails': 128,
    'blister': 129,
    'red_sore_around_nose': 130,
    'yellow_crust_ooze': 131
    
}

# ---------------------- Main Interface ----------------------
st.title("🏥 MediGuide Pro")
st.markdown("### Your AI-Powered Health Diagnosis Assistant")

# ---------------------- Symptom Selection ----------------------
def reset_form():
    st.session_state.predict = False
    st.session_state.symptoms = []

with st.container():
    st.markdown("#### 🔍 Select Your Symptoms")
    
    selected_symptoms = st.multiselect(
        "Search or select symptoms:",
        list(symptoms_dict.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
        placeholder="Type or choose symptoms...",
        key="symptom_selector",
        default=st.session_state.symptoms
    )
    
    if selected_symptoms:
        st.markdown("**Selected Symptoms:**")
        cols = st.columns(4)
        for i, symptom in enumerate(selected_symptoms):
            cols[i%4].markdown(
                f'<div class="symptom-pill">{symptom.replace("_", " ").title()}</div>', 
                unsafe_allow_html=True
            )

# ---------------------- Prediction ----------------------
col1, col2 = st.columns([3,1])
with col1:
    analyze_btn = st.button("🔬 Analyze Symptoms", use_container_width=True, type="primary")
with col2:
    clear_btn = st.button("🧹 Clear Selections", use_container_width=True, on_click=reset_form)

if clear_btn:
    st.session_state.symptoms = []
    st.rerun()

if analyze_btn:
    if len(selected_symptoms) < 1:
        st.error("⚠️ Please select at least one symptom")
    else:
        st.session_state.symptoms = selected_symptoms
        st.session_state.predict = True
        st.rerun()

if st.session_state.predict:
    with st.spinner("🧠 Analyzing symptoms with AI model..."):
        # ---------------------- Core Prediction Logic ----------------------
        input_vector = [0] * 132
        for symptom in st.session_state.symptoms:
            index = symptoms_dict[symptom]
            input_vector[index] = 1

        predicted_index = model.predict([input_vector])[0]
        predicted_disease = diseases_list.get(predicted_index, "Unknown Disease")

        # ---------------------- Data Processing ----------------------
        desc_row = disease_description[disease_description["Disease"] == predicted_disease]
        description = desc_row["Description"].values[0] if not desc_row.empty else "No description available."
        
        precaution_list = process_recommendations(precautions, predicted_disease)
        medication_list = process_recommendations(medications, predicted_disease)
        diet_list = process_recommendations(diet, predicted_disease)
        workout_list = process_recommendations(workout_df, predicted_disease)

        # ---------------------- Results Display ----------------------
        st.success("✅ Analysis Complete! Here's Your Health Report")
        
        # Disease Header Card
        st.markdown(f"""
            <div class="report-card">
                <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 25px;">
                    <div style="font-size: 2.5em; color: var(--primary-color);">🩺</div>
                    <div>
                        <h2 style="margin: 0; color: var(--primary-color);">{predicted_disease}</h2>
                        <p style="margin: 10px 0 0 0; color: var(--text-color); line-height: 1.5;">{description}</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Recommendations Grid
        cols = st.columns(4)
        recommendations = [
            ("🛡️ Precautions", precaution_list, "#FFD700"),
            ("💊 Medications", medication_list, "#4CAF50"),
            ("🥗 Diet Plan", diet_list, "#FF6B6B"),
            ("🏋️ Fitness", workout_list, "#9C27B0")
        ]

        for col, (title, items, color) in zip(cols, recommendations):
            with col:
                list_items = "".join(
                    [f"<li>{item}</li>" for item in items if item]
                )
                st.markdown(f"""
                    <div class="report-card">
                        <h3 style="color: {color}; margin-bottom: 15px;">{title}</h3>
                        <ul class="recommendation-list">
                            {list_items}
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

        # Safety Notice
        st.markdown("""
            <div class="report-card emergency-alert">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div style="font-size: 2em; color: #dc3545;">⚠️</div>
                    <div>
                        <h3 style="margin: 0; color: #dc3545;">Important Safety Notice</h3>
                        <p style="margin: 10px 0 0 0; color: var(--text-color);">
                            This analysis is not a substitute for professional medical advice. 
                            Always consult a qualified healthcare provider for diagnosis and treatment. 
                            In emergencies, call your local emergency number immediately.
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    if st.button("← Start New Diagnosis", type="primary", on_click=reset_form):
        st.rerun()

# ---------------------- Footer ----------------------
st.markdown("---")
