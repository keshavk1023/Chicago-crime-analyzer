# Importing necessary libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # Random forest model
from PIL import Image  # Image handling
import base64  # Encoding images to base64
from io import BytesIO  # Handling byte streams

# Load the trained model from a pickle file
with open('C:\\Users\\keshavk\\code\\crime analyser\\crime_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to convert image to base64 encoding
def image_to_base64(image_path):
    try:
        image = Image.open(image_path)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return ""

# Load and convert images to base64
logo_path = "C:\\Users\\keshavk\\code\\crime analyser\\Seal_of_the_Chicago_Police_Department.png"
banner_path = "C:\\Users\\keshavk\\code\\crime analyser\\CPD Logo.png"

logo_base64 = image_to_base64(logo_path)
banner_base64 = image_to_base64(banner_path)

# Set Streamlit page configuration
st.set_page_config(page_title="Chicago Police Department", page_icon=":police_car:", layout="wide")

# Custom CSS to style the Streamlit page
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

# HTML structure for positioning images
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

# Community area options and their numerical mappings
community_area_options = [
    "STREET", "APARTMENT", "RESIDENCE", "PARKING LOT / GARAGE(NON.RESID.)", 
    "RESIDENCE-GARAGE", "SMALL RETAIL STORE", "ALLEY", "OTHER (SPECIFY)", 
    "SIDEWALK", "RESTAURANT", "GAS STATION", "RESTAURANT DRIVE-IN",
    # Add more as needed
]
community_area_mapping = {name: i for i, name in enumerate(community_area_options)}

# Ward options and their numerical mappings
ward_options = [
    "CONCEALED CARRY LICENSE VIOLATION", "INTERFERENCE WITH PUBLIC OFFICER", "NARCOTICS",
    "WEAPONS VIOLATION", "HOMICIDE", "CRIMINAL TRESPASS", "OTHER OFFENSE",
    "OFFENSE INVOLVING CHILDREN", "STALKING", "ROBBERY", "CRIMINAL SEXUAL ASSAULT",
    "BATTERY", "BURGLARY", "THEFT", "ASSAULT", "SEX OFFENSE", "MOTOR VEHICLE THEFT",
    "CRIMINAL DAMAGE", "DECEPTIVE PRACTICE", "ARSON", "PUBLIC PEACE VIOLATION",
    "PROSTITUTION", "KIDNAPPING"
    # Add more as needed
]
ward_mapping = {name: i for i, name in enumerate(ward_options)}

# Streamlit app with three tabs: HOME, PREDICTION, ABOUT
tab1, tab2, tab3 = st.tabs(["HOME", "PREDICTION", "ABOUT"])

with tab1:
    # Display the HTML content with images
    st.markdown(html_content, unsafe_allow_html=True)
    st.write("")  # Add some space
    st.write("")  # Add some space
    st.write("")  # Add some space
    st.write("")  # Add some space
    st.markdown("# To serve our communities and protect the lives, rights, and property of all people in Chicago.")
    st.markdown("# VISION: All Chicagoans are safe, supported, and proud of the Chicago Police Department.")

with tab2:
    # Custom CSS for the prediction tab background
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

    # Create three columns for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        # Input for year, month, and hour
        year = st.number_input('Year', min_value=2001, max_value=2024, value=2024, help="Enter the year of the incident.")
        month = st.number_input('Month', min_value=1, max_value=12, value=1, help="Enter the month of the incident.")
        hour = st.number_input('Hour', min_value=0, max_value=23, value=0, help="Enter the hour of the incident (0-23).")

    with col2:
        # Input for district and ward
        district = st.number_input('District', min_value=1, max_value=25, value=1, help="Enter the police district number.")
        ward = st.selectbox('Ward', options=ward_options, help="Select the ward where the incident occurred.")

    with col3:
        # Input for community area
        community_area = st.selectbox('Community Area', options=community_area_options, help="Select the community area where the incident occurred.")

    if st.button('Predict'):
        with st.spinner('Making prediction...'):
            try:
                # Map community area and ward to their numerical values
                community_area_value = community_area_mapping[community_area]
                ward_value = ward_mapping[ward]

                # Create a DataFrame for the input data
                input_data = pd.DataFrame({
                    'Year': [year],
                    'Month': [month],
                    'Hour': [hour],
                    'District': [district],
                    'Ward': [ward_value],
                    'Community Area': [community_area_value],
                })

                # Make prediction using the loaded model
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0][1]

                # Display the prediction result
                if prediction:
                    st.success(f'Arrest is likely with a probability of {prediction_proba:.2f}')
                else:
                    st.warning(f'Arrest is unlikely with a probability of {prediction_proba:.2f}')
            except Exception as e:
                st.error(f"Error making prediction: {e}")

with tab3:
    # Display topics covered and project GitHub link
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
