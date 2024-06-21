import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import base64
from io import BytesIO

# Load the trained model
with open('C:\\Users\\keshavk\\code\\crime analyser\\crime_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to convert image to base64
def image_to_base64(image_path):
    try:
        image = Image.open(image_path)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return ""

# Load images and convert to base64
logo_path = "C:\\Users\\keshavk\\code\\crime analyser\\Seal_of_the_Chicago_Police_Department.png"
banner_path = "C:\\Users\\keshavk\\code\\crime analyser\\CPD Logo.png"

logo_base64 = image_to_base64(logo_path)
banner_base64 = image_to_base64(banner_path)

# Set Streamlit page configuration
st.set_page_config(page_title="Chicago Police Department", page_icon=":police_car:", layout="wide")

# Custom CSS to set the background to white and position the images
st.markdown(
    """
    <style>
    body {
        background-color: white;
    }
    .container {
        display: flex;
        align-items: flex-start;
        padding-top: 0;
        margin-top: 0;
    }
    .logo {
        margin: 0;
        padding: 0;
    }
    .banner {
        margin-left: 100px;
        margin-top: 0;
        flex-grow: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create an HTML structure to position the images
html_content = f"""
<div class="container">
    <div class="logo">
        <img src="data:image/png;base64,{logo_base64}" height="150px">
    </div>
    <div class="banner">
        <img src="data:image/png;base64,{banner_base64}" height="150px">
    </div>
</div>
"""

# Community area options and mapping to numerical values
community_area_options = [
    "STREET", "APARTMENT", "RESIDENCE", "PARKING LOT / GARAGE(NON.RESID.)", 
    "RESIDENCE-GARAGE", "SMALL RETAIL STORE", "ALLEY", "OTHER (SPECIFY)", 
    "SIDEWALK", "RESTAURANT", "GAS STATION", "RESTAURANT DRIVE-IN",
    # Add more as needed from the graph
]

community_area_mapping = {name: i for i, name in enumerate(community_area_options)}

# Ward options and mapping to numerical values
ward_options = [
    "CONCEALED CARRY LICENSE VIOLATION", "INTERFERENCE WITH PUBLIC OFFICER", "NARCOTICS",
    "WEAPONS VIOLATION", "HOMICIDE", "CRIMINAL TRESPASS", "OTHER OFFENSE",
    "OFFENSE INVOLVING CHILDREN", "STALKING", "ROBBERY", "CRIMINAL SEXUAL ASSAULT",
    "BATTERY", "BURGLARY", "THEFT", "ASSAULT", "SEX OFFENSE", "MOTOR VEHICLE THEFT",
    "CRIMINAL DAMAGE", "DECEPTIVE PRACTICE", "ARSON", "PUBLIC PEACE VIOLATION",
    "PROSTITUTION", "KIDNAPPING"
    # Add more as needed from the graph
]

ward_mapping = {name: i for i, name in enumerate(ward_options)}

# Streamlit app
tab1, tab2, tab3 = st.tabs(["HOME", "PREDICTION", "ABOUT"])

with tab1:
    # Display the HTML content
    st.markdown(html_content, unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("# To serve our communities and protect the lives, rights, and property of all people in Chicago.")
    st.markdown("# VISION: All Chicagoans are safe, supported, and proud of the Chicago Police Department.")

with tab2:
    st.markdown(
        """
        <style>
        .main {
            background-color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Crime Prediction Input")

    # Create columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.number_input('Year', min_value=2001, max_value=2024, value=2024, help="Enter the year of the incident.")
        month = st.number_input('Month', min_value=1, max_value=12, value=1, help="Enter the month of the incident.")
        hour = st.number_input('Hour', min_value=0, max_value=23, value=0, help="Enter the hour of the incident (0-23).")

    with col2:
        district = st.number_input('District', min_value=1, max_value=25, value=1, help="Enter the police district number.")
        ward = st.selectbox('Ward', options=ward_options, help="Select the ward where the incident occurred.")

    with col3:
        community_area = st.selectbox('Community Area', options=community_area_options, help="Select the community area where the incident occurred.")

    if st.button('Predict'):
        with st.spinner('Making prediction...'):
            try:
                # Map the community area and ward to their numerical values
                community_area_value = community_area_mapping[community_area]
                ward_value = ward_mapping[ward]

                # Create a DataFrame for the input
                input_data = pd.DataFrame({
                    'Year': [year],
                    'Month': [month],
                    'Hour': [hour],
                    'District': [district],
                    'Ward': [ward_value],
                    'Community Area': [community_area_value],
                })

                # Prediction
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0][1]

                # Display prediction
                if prediction:
                    st.success(f'Arrest is likely with a probability of {prediction_proba:.2f}')
                else:
                    st.warning(f'Arrest is unlikely with a probability of {prediction_proba:.2f}')
            except Exception as e:
                st.error(f"Error making prediction: {e}")

with tab3:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("# TOPICS COVERED")
        st.subheader("Data Exploration")
        st.subheader("Data Visualization")
        st.subheader("Python Scripting")
        st.subheader("Machine Learning")
        st.subheader("Prediction")

    with col2:
        st.write("# **My Project GitHub link** ⬇️")
        st.write("#### [GitHub](https://github.com/keshavk1023/)")
