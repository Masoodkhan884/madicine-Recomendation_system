import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("Model/svc.pkl", "rb") as file:
    model = pickle.load(file)

# Load additional information
disease_description = pd.read_csv("Dataset/description.csv")  # Columns: Disease, Description
precautions = pd.read_csv("Dataset/precautions_df.csv")       # Columns: Disease, Precaution_1, ...
medications = pd.read_csv("Dataset/medications.csv")          # Columns: Disease, Medication_1, ...
diet = pd.read_csv("Dataset/diets.csv")                       # Columns: Disease, Diet_1, ...
workout_df = pd.read_csv("Dataset/workout_df.csv")               # Columns: Disease, Workout_1, ...

# Disease and symptom dictionaries
diseases_list = {
    15: 'Fungal infection',
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

# Create reverse dict to display symptom names
symptom_names = list(symptoms_dict.keys())

# Streamlit UI
st.set_page_config(page_title="Medicine Recommendation System", layout="centered")
st.title("💊 Medicine Recommendation System")

st.markdown("""
This application allows users to select symptoms and get AI-based disease prediction 
along with useful health-related suggestions such as precautions, medications, diet, 
and workout plans.
""")

with st.container():
    st.subheader("🩺 Select Symptoms")
    selected_symptoms = st.multiselect("Choose the symptoms you are experiencing:", symptom_names)

    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            # Initialize feature vector
            input_vector = [0] * 132
            for symptom in selected_symptoms:
                index = symptoms_dict[symptom]
                input_vector[index] = 1

            # Predict using model
            predicted_index = model.predict([input_vector])[0]
            predicted_disease = diseases_list.get(predicted_index, "Unknown Disease")

            # Get details
            desc_row = disease_description[disease_description["Disease"] == predicted_disease]
            description = desc_row["Description"].values[0] if not desc_row.empty else "No description available."

            precaution_list = precautions[precautions["Disease"] == predicted_disease].iloc[:, 1:].dropna(axis=1).values.flatten().tolist()
            medication_list = medications[medications["Disease"] == predicted_disease].iloc[:, 1:].dropna(axis=1).values.flatten().tolist()
            diet_list = diet[diet["Disease"] == predicted_disease].iloc[:, 1:].dropna(axis=1).values.flatten().tolist()
            workout_list = workout_df[workout_df["Disease"] == predicted_disease]["workout"].dropna().tolist()

            # Tabs
            tabs = st.tabs([
                "Predicted Disease", "Disease Description", "Precautions",
                "Recommended Medications", "Diet Recommendations", "Workout Suggestions"
            ])

            with tabs[0]:
                st.header("🧾 Predicted Disease")
                st.success(predicted_disease)

            with tabs[1]:
                st.header("🧠 Disease Description")
                st.info(description)

            with tabs[2]:
                st.header("🛡️ Precautions")
                for item in precaution_list[1:]:
                    st.write("🔹", item)

            with tabs[3]: 
                st.header("💊 Recommended Medications")
                if medication_list:
                    for med in str(medication_list[0]).split(","):
                        st.write("💊", med.strip(" []'\""))

            with tabs[4]:
                st.header("🥗 Diet Recommendations")
                if diet_list:
                    for item in str(diet_list[0]).split(","):
                        st.write("🍽️", item.strip(" []'\""))


    # Display workout suggestions only when the disease matches the predicted disease # to 
    # print only the workout_list for only when when the disease match the predicted disease
            with tabs[5]: 
                st.header("🏋️ Workout Suggestions")
                if workout_list:
                    for workout in workout_list:
                        st.write("🏋️", workout)
                   
            
                    




                   
