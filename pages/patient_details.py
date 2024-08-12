import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from gradio_client import Client
from session import get_session
import time
from PIL import Image
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import base64
st.set_page_config(page_title="Predicting Adherence", page_icon="📈", layout='wide')

email_address = st.secrets["email"]["email_address"]
email_password = st.secrets["email"]["email_password"]
smtp_server = st.secrets["email"]["smtp_server"]
smtp_port = st.secrets["email"]["smtp_port"]

def send_email(to_email, subject, message):
    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = email_address
    msg['To'] = to_email
    msg['Subject'] = subject

    # Add body to email
    msg.attach(MIMEText(message, 'plain'))

    # Send the email
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(email_address, email_password)
    text = msg.as_string()
    server.sendmail(email_address, to_email, text)
    server.quit()

if st.session_state.get('logged_in'):
    
    st.title("Embedded Power BI Dashboard")
    st.markdown(
        """
        <style>
        .report-container {
            position: relative;
            width: 100%;
            height: 0;
            padding-bottom: 56.25%; /* Aspect ratio */
        }
        .report-iframe {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
        }
        </style>
        <div class="report-container">
            <iframe class="report-iframe" src="https://app.powerbi.com/reportEmbed?reportId=a97f5efa-7c3a-4430-bc10-b70154171e32&autoAuth=true&ctid=ac29cc87-d08e-4f42-bee7-3223d3c25249"></iframe>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <style>
        [data-testid="stWidgetLabel"] {
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 10px;
        }
        [class="st-emotion-cache-10trblm e1nzilvr1"] {
            background-color: white;
            padding: 10px;
            border-radius: 10px;
        }
        .st-emotion-cache-eqffof e1nzilvr5{
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
        )
    st.title('Patient Adherence Prediction')
    st.sidebar.title("Algorithm Stats")
    input = st.sidebar.selectbox("Choose model:", ("Current Model", "XGBoost", "MLPClassifier", "Gradient Boost", "Logistic Regression"))
    
    st.markdown("""
            <style>
            .custom-header {
                font-size: 26px;
                font-weight: bold;
                color: #333333; /* Dark Gray */
                background-color: #f0f0f0; /* Light Ash Gray */
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }
            .custom-text {
                font-size: 15px;
                color: #555555; /* Medium Gray */
                background-color: #f7f7f7; /* Slightly Lighter Gray */
                padding: 8px;
                border-radius: 5px;
                margin-top: 5px;
                text-align: center;
            }
            </style>
            """, unsafe_allow_html=True)
    if input == "Current Model":
        image = Image.open('pages/imagefiles/final_confusion_mat.png')  # Replace with your image path 
        # Display the image in Streamlit
        st.image(image, caption= 'Confusion Matrix', use_column_width=True)
        
    if input == "XGBoost":

        # Display the styled header and text
        st.markdown('<div class="custom-header">  XGBoost Algorithm Stats  </div>', unsafe_allow_html=True)
        st.markdown('<div class="custom-text">Accuracy: 80.48% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Precision: 80.4%</div>', unsafe_allow_html=True)
        st.markdown('<div class="custom-text">Recall: 80.3% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; F1 Score: 72.5%</div>', unsafe_allow_html=True)
        
        image = Image.open('pages/imagefiles/Model Metrics/XgBoost/training_confusion_matrix.png')  # Replace with your image path 
        # Display the image in Streamlit
        st.image(image, caption= 'Confusion Matrix', use_column_width=True)

    if input == "MLPClassifier":
        st.markdown('<div class="custom-header">MLPClassifier Algorithm Stats</div>', unsafe_allow_html=True)
        st.markdown('<div class="custom-text">Accuracy: 79.71% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Precision: 63.5%</div>', unsafe_allow_html=True)
        st.markdown('<div class="custom-text">Recall: 79.7% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; F1 Score: 70.7%</div>', unsafe_allow_html=True)
        
        image = Image.open('pages/imagefiles/Model Metrics/MLP Classifier/MLP training_confusion_matrix.png')  # Replace with your image path 
        # Display the image in Streamlit
        st.image(image, caption= 'Confusion Matrix', use_column_width=True)
    
    if input == "Gradient Boost":
        st.markdown('<div class="custom-header">Gradient Boost Algorithm Stats</div>', unsafe_allow_html=True)
        st.markdown('<div class="custom-text">Accuracy: 79.9% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Precision: 83.9%</div>', unsafe_allow_html=True)
        st.markdown('<div class="custom-text">Recall: 79.9% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; F1 Score: 72.5%</div>', unsafe_allow_html=True)
        
        image = Image.open('pages/imagefiles/Model Metrics/Gradient Boosting/training_confusion_matrix.png')  # Replace with your image path 
        # Display the image in Streamlit
        st.image(image, caption='Confusion Matrix', use_column_width=True)
        
    
    if input == "Logistic Regression":
        st.markdown('<div class="custom-header">Logistic Regression Algorithm Stats</div>', unsafe_allow_html=True)
        st.markdown('<div class="custom-text">Accuracy: 79.7% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Precision: 63.3%</div>', unsafe_allow_html=True)
        st.markdown('<div class="custom-text">Recall: 79.7% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; F1 Score: 72.5%</div>', unsafe_allow_html=True)
        
        image = Image.open('pages/imagefiles/Model Metrics/Logistic regression/training_confusion_matrix (1).png')  # Replace with your image path 
        # Display the image in Streamlit
        st.image(image, caption= 'Confusion Matrix', use_column_width=True)
        
    predicted_adherence = ""
    st.sidebar.title("Select Input Method")
    input_method = st.sidebar.radio("Method of Input:", ("Upload XLSX", "Manual form"))
    
    def add_bg_from_local(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
            }}
            .stApp p {{
            color: black;  /* Change paragraph colors to white */
            }}
          
            </style>
            """,
            unsafe_allow_html=True
        )

    main_bg_path = 'pages/imagefiles/pexels-pixabay-40568.jpg'
    
    # Add background images
    add_bg_from_local(main_bg_path)
    
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('model.joblib')
    le_y = joblib.load('label_encoder.joblib')

    
    st.markdown(
    """
    <div style="background-color: #b2b2b2; padding: 10px; border-radius: 5px; text-align: center;">
        <span style="color: #000000; font-weight: bold;">
            Provide the following patient details:
        </span>
    </div>
    """,
    unsafe_allow_html=True
    )
    if input_method == "Upload XLSX":
        df1 = ""
        uploaded_file = st.file_uploader("Choose a file", type = 'xlsx')
        if uploaded_file is not None:
            df1 = pd.read_excel(uploaded_file)
            s = {}
            for index, row in df1.iterrows():
                s = {}
                s = {
                    'Age': row['Age'],
                    'InsuranceType': row['InsuranceType'],
                    'MedianIncome': row['MedianIncome'],
                    'HospitalizationPriorYear': row['HospitalizationPriorYear'],
                    'MSRelatedHospitalization': row['MSRelatedHospitalization'],
                    'RelapsePriorYear': row['RelapsePriorYear'],
                    'Disease': row['Disease'],
                    'TherapeuticArea': row['TherapeuticArea'],
                    'SpecialtyPharma': row['SpecialtyPharma'],
                    'TrialLengthWeeks': row['TrialLengthWeeks'],
                    'MicroReimbursements': row['MicroReimbursements'],
                    'DoseLengthSeconds': row['DoseLengthSeconds'],
                    'DoseDelayHours': row['DoseDelayHours'],
                }
                
            
                user_df = pd.DataFrame([s])
                
                user_df_processed = preprocessor.transform(user_df)
                
                user_prediction = model.predict(user_df_processed)

                # Decode the prediction
                predicted_adherence = le_y.inverse_transform(user_prediction)[0]

                if(predicted_adherence == 'ADHERENT'):
                    st.markdown(
                        f"""
                        <div style="background-color: #b2b2b2; padding: 10px; border-radius: 5px; text-align: center;">
                            <span style="color: #000000; font-weight: bold;">
                                Predicted Adherence: {predicted_adherence} &nbsp;&nbsp; 😊
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                    f"""
                    <div style="background-color: #b2b2b2; padding: 10px; border-radius: 5px; text-align: center;">
                        <span style="color: #000000; font-weight: bold;">
                            Predicted Adherence: {predicted_adherence} &nbsp;&nbsp; 🙁
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                df1[index, 'Adherence'] = predicted_adherence
                if predicted_adherence == "NON-ADHERENT":
                    client = Client("yuva2110/vanilla-charbot")
                    result = client.predict(
                        message=f"""Provide a detailed set of recommendations for patients with {row['Disease']} in the therapeutic area of {row['TherapeuticArea']}, who are currently using the specialty pharma product {row['SpecialtyPharma']} and taking {row['MedicationName']}. The goal is to encourage patients to continue their medication to avoid future expenses and explain the risks associated with discontinuation in a detailed manner. Focus on retaining patients and offer practical advice and suggestions. Include the name {row['BrandName']} in the recommendations.

                            Include the following points:

                            Importance of Adherence:

                            Explain why it is crucial to consistently take {row['MedicationName']} as prescribed.
                            Risks of Discontinuation:

                            Detail the potential negative effects of stopping the medication, including worsening symptoms, increased healthcare costs, and additional health complications.
                            Management Tips:

                            Provide actionable tips for managing medication intake, such as setting reminders and tracking medication.
                            Financial Considerations:

                            Suggest ways to manage medication costs, including assistance programs and insurance coverage.
                            Educational Resources:

                            Encourage patients to educate themselves about {row['Disease']} and their treatment, and to consult healthcare professionals for any questions or concerns.
                            Support and Engagement:

                            Emphasize the importance of regular check-ups and open communication with healthcare providers.
                            End with:

                            A message of support, encouraging patients to stay committed to their treatment plan for better health outcomes.
                            Make the patient to stay with the particular brand name {row['BrandName']} and try to provide a non affective discount and goodies from the company sign below with particular brand name or company name. make the response brief and powerful.
                            limit the response to 200 words
                            """,
                        system_message="You are a friendly Chatbot.",
                        max_tokens=512,
                        temperature=0.7,
                        top_p=0.95,
                        api_name="/chat"
                    )
                else:
                    client = Client("yuva2110/vanilla-charbot")
                    result = client.predict(
                        message=f"give appreciation for patients with {row['Disease']} in therapeutic area of {row['TherapeuticArea']} taking speciality pharma of {row['SpecialtyPharma']} and medication {row['MedicationName']} to continue there medication to avoid future expenses and say them about the risks involved with discontinuation in detailed manner and make them retained and feel good about their progress and give name as {row['BrandName']} , give in points",
                        system_message="You are a friendly Chatbot.",
                        max_tokens=512,
                        temperature=0.7,
                        top_p=0.95,
                        api_name="/chat"
                    )
                df1[index, 'Recommendation'] = result
                save_path = "updated_file.xlsx"
                df1.to_excel(save_path, index=False, engine='openpyxl')
                st.success(f"File saved as {save_path}")

            # Provide a download link
            with open(save_path, "rb") as file:
                btn = st.download_button(
                    label="Download Updated Excel File",
                    data=file,
                    file_name=save_path,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    elif input_method == "Manual form":
        # Organize input fields into columns
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        col5, col6 = st.columns(2)
        col7, col8 = st.columns(2)
        col9, col10 = st.columns(2)
        col11, col12 = st.columns(2)
        col13, col14 = st.columns(2)

        with col1:
            age = st.number_input('Age', min_value=0, max_value=120, value=30)
        with col2:
            gender = st.selectbox("Gender", ("MALE", "FEMALE"))
        with col3:
            race = st.selectbox("Race", ("ASIAN", "WHITE", "BLACK OR AFRICAN AMERICAN", "OTHER"))
        with col4:
            insurance_type = st.selectbox("Insurance Type", ("COMMERCIAL", "NON-COMMERCIAL"))
        with col5:
            median_income = st.number_input('Median Income', min_value=0, value=50000)
        with col6:
            hospitalization_prior_year = st.selectbox("Hospitalization Prior Year", ("YES", "NO"))
        with col7:
            ms_related_hospitalization = st.selectbox("MS Related Hospitalization", ("YES", "NO"))
        with col8:
            relapse_prior_year = st.selectbox("Relapse Prior Year", ("YES", "NO"))
        with col9:
            disease = st.selectbox("Disease", ("BIPOLAR 1 DISORDER", "ASTHMA", "HYPERTENSION", "DIABETES MELLITUS"))
        with col10:
            therapeutic_area = st.selectbox("Therapeutic Area", ("PSYCHIATRY", "PULMONOLOGY", "CARDIOLOGY", "ENDOCRINOLOGY"))
        with col11:
            specialty_pharma = st.selectbox("Specialty Pharma", ("LITHIUM", "INHALED CORTICOSTEROIDS", "ACE INHIBITORS", "INSULIN"))
        with col12:
            trial_length_weeks = st.number_input('Trial Length (Weeks)', min_value=0, value=12)
        with col13:
            micro_reimbursements = st.selectbox('Micro Reimbursements', ['YES', 'NO'])
        with col14:
            dose_length_seconds = st.number_input('Dose Length (Seconds)', min_value=0, value=60)
        
        dose_delay_hours = st.number_input('Dose Delay (Hours)', min_value=0.00000, value=2.00000)

        medication_name = st.selectbox("Medication Name", ("DULERA", "BENICAR HCT", "BENICAR", "ALTACE",
                                                        "ADVAIR", "PULMICORT", "PRINZIDE", "ALBUTEROL", "NORVASC",
                                                        "SINGULAIR", "QVAR", "ALVESCO", "COREG", "AVAPRO", "SYMBICORT",
                                                        "ASMANEX", "ZESTORETIC", "FLOVENT", "MICARDIS HCT", "ZESTRIL",
                                                        "MICARDIS", "COZAAR", "DIOVAN", "UNIVASC", "HYZAAR", "LOTREL",
                                                        "PRINIVIL", "AVALIDE"))

        brand_name = st.selectbox("Brand Name", ("MOMETASONE FUROATE AND FORMOTEROL FUMARATE",
                                                "OLMESARTAN MEDOXOMIL AND HYDROCHLOROTHIAZIDE", "OLMESARTAN MEDOXOMIL",
                                                "RAMIPRIL", "FLUTICASONE PROPIONATE AND SALMETEROL", "BUDESONIDE",
                                                "LISINOPRIL/HYDROCHLOROTHIAZIDE", "ALBUTEROL SULFATE", "AMLODIPINE BESYLATE",
                                                "MONTELUKAST SODIUM", "BECLOMETHASONE DIPROPIONATE", "CICLESONIDE",
                                                "CARVEDILOL", "IRBESARTAN", "BUDESONIDE AND FORMOTEROL FUMARATE DIHYDRATE",
                                                "MOMETASONE FUROATE", "LISINOPRIL AND HYDROCHLOROTHIAZIDE",
                                                "FLUTICASONE PROPIONATE", "TELMISARTAN AND HYDROCHLOROTHIAZIDE",
                                                "LISINOPRIL", "TELMISARTAN", "LOSARTAN POTASSIUM", "VALSARTAN",
                                                "MOEXIPRIL HCL", "LOSARTAN POTASSIUM/HYDROCHLOROTHIAZIDE", 
                                                "AMLODIPINE BESYLATE AND BENAZEPRIL HCL", "IRBESARTAN AND HYDROCHLOROTHIAZIDE"))
        email_id = st.text_input("Enter Patient Email Address:", key ='email_input')
        
        def is_valid_email(email):
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return re.match(pattern, email) is not None
        
        
        input_data = {
            'Age': [age],
            'InsuranceType': [insurance_type],
            'MedianIncome': [median_income],
            'HospitalizationPriorYear': [hospitalization_prior_year],
            'MSRelatedHospitalization': [ms_related_hospitalization],
            'RelapsePriorYear': [relapse_prior_year],
            'Disease': [disease],
            'TherapeuticArea': [therapeutic_area],
            'SpecialtyPharma': [specialty_pharma],
            'TrialLengthWeeks': [trial_length_weeks],
            'MicroReimbursements': [micro_reimbursements],
            'DoseLengthSeconds': [dose_length_seconds],
            'DoseDelayHours': [dose_delay_hours],
        }

        if st.button('Predict'):
        
            user_df = pd.DataFrame(input_data)
                
            user_df_processed = preprocessor.transform(user_df)
                
            user_prediction = model.predict(user_df_processed)

            # Decode the prediction
            predicted_adherence = le_y.inverse_transform(user_prediction)[0]
            if(predicted_adherence == 'ADHERENT'):
                st.markdown(
                    f"""
                    <div style="background-color: #b2b2b2; padding: 10px; border-radius: 5px; text-align: center;">
                        <span style="color: #000000; font-weight: bold;">
                            Predicted Adherence: {predicted_adherence} &nbsp;&nbsp; 😊
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                    st.markdown(
                    f"""
                    <div style="background-color: #b2b2b2; padding: 10px; border-radius: 5px; text-align: center;">
                        <span style="color: #000000; font-weight: bold;">
                            Predicted Adherence: {predicted_adherence} &nbsp;&nbsp; 🙁
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if predicted_adherence == "NON-ADHERENT":
                client = Client("yuva2110/vanilla-charbot")
                result = client.predict(
                    message=f"""Provide a detailed set of recommendations for patients with {disease} in the therapeutic area of {therapeutic_area}, who are currently using the specialty pharma product {specialty_pharma} and taking {medication_name}. The goal is to encourage patients to continue their medication to avoid future expenses and explain the risks associated with discontinuation in a detailed manner. Focus on retaining patients and offer practical advice and suggestions. Include the name {brand_name} in the recommendations.

                        Include the following points:

                        Importance of Adherence:

                        Explain why it is crucial to consistently take {medication_name} as prescribed.
                        Risks of Discontinuation:

                        Detail the potential negative effects of stopping the medication, including worsening symptoms, increased healthcare costs, and additional health complications.
                        Management Tips:

                        Provide actionable tips for managing medication intake, such as setting reminders and tracking medication.
                        Financial Considerations:

                        Suggest ways to manage medication costs, including assistance programs and insurance coverage.
                        Educational Resources:

                        Encourage patients to educate themselves about {disease} and their treatment, and to consult healthcare professionals for any questions or concerns.
                        Support and Engagement:

                        Emphasize the importance of regular check-ups and open communication with healthcare providers.
                        End with:

                        A message of support, encouraging patients to stay committed to their treatment plan for better health outcomes.
                        Make the patient to stay with the particular brand name {brand_name} and try to provide a non affective discount and goodies from the company sign below with particular brand name or company name. make the response brief and powerful.
                        limit the response to 200 words
                        """,
                    system_message="You are a friendly Chatbot.",
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    api_name="/chat"
                )
                if(is_valid_email(email_id)):
                    send_email(email_id, 'Non - Adherent', result)
                else:
                    st.markdown(
                    f"""
                    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; text-align: center;">
                        <span style="color: #000000; font-weight: bold;">
                            Invalid email id ... Mail not sent !!
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
            else:
                client = Client("yuva2110/vanilla-charbot")
                result = client.predict(
                    message=f"give appreciation for patients with {disease} in therapeutic area of {therapeutic_area} taking speciality pharma of {specialty_pharma} and medication {medication_name} to continue there medication to avoid future expenses and say them about the risks involved with discontinuation in detailed manner and make them retained and feel good about their progress and give name as {brand_name} , give in points conclude with the given values",
                    system_message="You are a friendly Chatbot.",
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    api_name="/chat"
                )
                if(is_valid_email(email_id)):
                    send_email(email_id, 'Adherent', result)
                else:
                    st.markdown(
                    f"""
                    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; text-align: center;">
                        <span style="color: #000000; font-weight: bold;">
                            Invalid email id ... Mail not sent !!
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
            st.markdown(
                    f"""
                    <div style="background-color: #b2b2b2; padding: 10px; border-radius: 5px; text-align: center;">
                        <span style="color: #000000; font-weight: bold;">
                            {result} 
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

else:
    st.error("You should sign in")
    time.sleep(1)
    st.switch_page('pages/login.py')
