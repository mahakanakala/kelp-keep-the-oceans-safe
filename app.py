import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import os
import time
from dotenv import load_dotenv
import streamlit.components.v1 as components

load_dotenv()
GA_ID = os.getenv('GOOGLE_ANALYTICS_ID')
# DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

def inject_ga():
    """Inject Google Analytics tracking code"""
    if not GA_ID:
        return
    
    try:
        with open("google_analytics.html", "r") as f:
            ga_html = f.read()
            
        ga_html = ga_html.replace('%%GOOGLE_ANALYTICS_ID%%', GA_ID)
        components.html(ga_html, height=0)
    except Exception as e:
        st.error(f"Failed to inject Google Analytics: {str(e)}")

inject_ga()

# mapping
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import branca.colormap as cm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# database
from src.database import get_engine, test_connection
from sqlalchemy import text

os.makedirs('uploads/oil_spills', exist_ok=True)
os.makedirs('uploads/garbage', exist_ok=True)

# Initialize connection to database
@st.cache_resource
def init_connection():
    try:
        if not test_connection():
            st.error("Could not connect to database. Please check your configuration.")
            return None
        return get_engine()
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return None

# Load data from database
@st.cache_data
def load_data():
    try:
        engine = init_connection()
        if engine is None:
            return None, None
        
        # Load garbage data
        garbage_df = pd.read_sql_query(
            "SELECT * FROM marine_microplastic_density",
            engine
        )
        
        # Load oil spill data
        oil_spill_df = pd.read_sql_query(
            "SELECT * FROM oil_spills_historical",
            engine
        )
        
        return garbage_df, oil_spill_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def save_incident_report(incident_type, date, location, description, photo_path):
    """Save an incident report to the database"""
    try:
        engine = init_connection()
        if engine is None:
            return False
            
        table_name = "reported_oil_spills" if incident_type == "oil" else "reported_garbage_incidents"
        
        with engine.connect() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {table_name} (date, location, description, photo_path)
                    VALUES (:date, :location, :description, :photo_path)
                """),
                {
                    "date": date,
                    "location": location,
                    "description": description,
                    "photo_path": photo_path
                }
            )
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving report: {str(e)}")
        return False

def get_recent_reports():
    """Get recent incident reports from both tables"""
    try:
        engine = init_connection()
        if engine is None:
            return None, None
            
        oil_reports = pd.read_sql_query(
            "SELECT * FROM reported_oil_spills ORDER BY created_at DESC LIMIT 10",
            engine
        )
        
        garbage_reports = pd.read_sql_query(
            "SELECT * FROM reported_garbage_incidents ORDER BY created_at DESC LIMIT 10",
            engine
        )
        
        return oil_reports, garbage_reports
    except Exception as e:
        st.error(f"Error loading reports: {str(e)}")
        return None, None

st.title("Kelp Keep the Oceans Safe")

# Load data
garbage_df, oil_spill_df = load_data()
if garbage_df is None or oil_spill_df is None:
    st.error("Failed to load data. Please check the database connection.")
    st.stop()

# Convert the Degrees Minutes Seconds format to Degrees for Folium plotting in the oil_spill_df
def dms_to_dd(dms_value):
    parts = dms_value.replace(',', '.').split('.')
    degrees = float(parts[0])
    minutes = 0
    seconds = 0
    if len(parts) > 1:
        minutes = float(parts[1])
    if len(parts) > 2:
        seconds = float(parts[2])
    dd_value = degrees + (minutes / 60) + (seconds / 3600)
    return dd_value

oil_spill_df['Latitude'] = oil_spill_df['Latitude'].apply(dms_to_dd)
oil_spill_df['Longitude'] = oil_spill_df['Longitude'].apply(dms_to_dd)

st.markdown('''
## Understanding Marine Environmental Impact

**Oil**, an age-old fossil fuel, plays a crucial role in heating our homes, generating electricity, and driving various sectors of our economy. However, accidental oil spills in the ocean pose *significant* challenges. These spills can wreak havoc on marine life, spoil beach outings, and render seafood unsafe for consumption. Similarly, 
**Plastic**, a ubiquitous material in our daily lives, has found its way into the heart of our oceans, creating a pervasive environmental crisis. **Oil spills** and **plastic pollution** represent urgent environmental challenges, disrupting marine ecosystems and endangering both wildlife and human communities. This interactive map provides a visual representation of these environmental challenges, using data from NOAA and advanced mapping techniques.
''')

oil_image = Image.open('./public/images/turtle_oil_spill_copy.png')
plastic_image = Image.open('./public/images/turtle_plastic.jpeg')

images_column, description_column = st.columns(2)

with images_column:
    st.subheader("Oil Spills: Impact on Wildlife")
    st.image(oil_image, caption="MSNBC showcases a photo of the sea affected by the BP's oil spill")
    barrels_spilled = oil_spill_df['Barrels'].sum()
    recorded_spills = oil_spill_df.shape[0]
    right_metric, left_metric = st.columns(2)
    with right_metric:
        st.metric(label= ":oil_drum: Number of Barrels of Oil Spilled", value=barrels_spilled)
    with left_metric:
        st.metric(label=":chart_with_upwards_trend: Number of Recorded Spills", value=recorded_spills)
    st.metric(label="Average Number of Increase in Oil Spills/Year", value="2", delta="22% every year from 1929-1981", delta_color='inverse')

with description_column:
    st.subheader("Garbage Patches: Impact on Wildlife")
    st.image(plastic_image, caption="WWF showcases a photo of a turtle affected by plastic pollution")
    recorded_spills = len(garbage_df)
    garbage_df['Total_Pieces_L'] = pd.to_numeric(garbage_df['Total_Pieces_L'], errors='coerce')
    plastic_density = garbage_df['Total_Pieces_L'].sum()
    right_metric, left_metric = st.columns(2)
    with right_metric:
        st.metric(label=":roll_of_paper: Plastic Pieces Found in Oceans", value=plastic_density)
    with left_metric:
        st.metric(label=":world_map: Number of Patches as big as the Pacific Garbage Circle", value=recorded_spills)
        
st.divider()

# Create a geom obj for plotting
def prepare_heatmap(df,adn):
    df[['Latitude','Longitude']]=df[['Latitude','Longitude']].astype(dtype=float)
    df[adn] = df[adn].astype(dtype=float)
    df['Geometry'] = pd.Series([(df.loc[i,'Latitude'],df.loc[i,'Longitude']) for i in range(len(df['Latitude']))])
    
def to_datetime(df,date_col='Date',frmt='%Y-%m-%d'):
    df[date_col] =pd.to_datetime(df[date_col],errors='coerce')
    df['year'] = df[date_col].dt.year

# Apply the get_geom function
prepare_heatmap(garbage_df,'Total_Pieces_L')
prepare_heatmap(oil_spill_df, "Barrels")

# Apply Date Time function
to_datetime(garbage_df)

# Find the ideal frame for loading the render window
start_loc_plastic= (np.mean(garbage_df['Latitude']),np.mean(garbage_df['Longitude']))
start_loc_oil= (np.mean(oil_spill_df['Latitude']),np.mean(oil_spill_df['Longitude']))

st.markdown('''
## Visualizing the Impact of Oil Spills and Garbage Patches
*Use the multi-select button to view the layers/attributes*
            ''')

# Add custom CSS to reduce spacing
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Garbage & Oil Heatmaps
m_1=folium.Map(location=start_loc_plastic,
              zoom_start=2,
              min_zoom=1.5,
              tiles='CartoDB positron',
              attr="<a href=https://endless-sky.github.io/>Endless Sky</a>")

col1, col2 = st.columns(2)
with col1:
    show_heatmap_garbage = st.checkbox("Show Garbage Heatmap")
    show_garbage_markers = st.checkbox("Show Garbage Markers")
with col2:
    show_heatmap_oil = st.checkbox("Show Oil Spill Heatmap")
    show_oil_markers = st.checkbox("Show Oil Spill Markers")
    
if show_heatmap_garbage:
    HeatMap(data=garbage_df[['Latitude', 'Longitude', 'Total_Pieces_L']].values, radius=10, blur=5,
            # gradient={.1: '#A1887F', .3: "#795548", 1: '#3E2723'}
            ).add_to(m_1)
    colormap = cm.LinearColormap(colors=['blue', 'lightgreen', 'yellow', 'orange', 'red' ],
                                  vmin=garbage_df['Total_Pieces_L'].min(),
                                  vmax=garbage_df['Total_Pieces_L'].max(),
                                  caption='Total Pieces of Garbage')

    # Add colormap legend to the map at the top left corner
    colormap.add_to(m_1)

if show_heatmap_oil:
    HeatMap(data=oil_spill_df[['Latitude', 'Longitude', 'Barrels']].values, 
            radius=10, 
            blur=5,
            gradient={'0.2': 'beige', '0.6': 'brown', '1.0': 'black'}).add_to(m_1)
    colormap = cm.LinearColormap(colors=['blue', 'lightgreen', 'yellow', 'orange', 'red' ],
                                  vmin=oil_spill_df['Barrels'].min(),
                                  vmax=oil_spill_df['Barrels'].max(),
                                  caption='Total Barrels of Oil Spilled')
    colormap.add_to(m_1)

if show_oil_markers:
    for _, row in oil_spill_df.iterrows():
        popup_content = f"""
        <h3>{row['Oil Spill Name']}</h3>
        <p><b>Location:</b> {row['Location']}</p>
        <p><b>Date:</b> {row['Date']}</p>
        <p><b>Number of Barrels:</b> {row['Barrels']}</p>
        <p><b>Impact Regions:</b> {row['Impact Regions']}</p>
        <p><b>Organisms Affected:</b> {row['Organisms Affected']}</p>
        """
        folium.Marker(location=[row['Latitude'], row['Longitude']],
                      tooltip=f"<b>{row['Oil Spill Name']}</b><br># of Barrels Spilled: {row['Barrels']}",
                      popup=folium.Popup(popup_content, max_width=800),
                      icon=folium.Icon(icon='tint', prefix='fa', color='black')
                      ).add_to(m_1)

if show_garbage_markers:
        # Add the Pacific Garbage Patch CircleMarker
    latitude, longitude = 38.5, -145
    radius = 70  # Approximately 0.62 million square miles in meters
    folium.CircleMarker(location=[latitude, longitude],
                        radius=radius,
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.2,
                        popup='Pacific Garbage Patch').add_to(m_1)

st_folium(m_1, width=1300)

# Report forms section
st.markdown("## Report Environmental Incidents")

# Collection forms
oil_col, garbage_col = st.columns(2)
with oil_col:
    st.subheader("Report Oil Spill Incident")
    oil_spill_date = st.date_input("Date of Incident")
    oil_spill_location = st.text_input("Location")
    oil_spill_description = st.text_area("Description")
    oil_spill_photo = st.file_uploader("Upload Photo of Incident", type=["jpg", "jpeg", "png"])

    if st.button("Report Oil Spill"):
        time.sleep(.5)
        st.toast('Reported Oil Spill Incident')
        if oil_spill_date and oil_spill_location and oil_spill_description and oil_spill_photo:
            # Save image locally
            image_path = f'uploads/oil_spills/{oil_spill_date}.jpg'
            with open(image_path, 'wb') as f:
                f.write(oil_spill_photo.getvalue())

            save_incident_report("oil", oil_spill_date, oil_spill_location, oil_spill_description, image_path)

with garbage_col:
    st.subheader("Report Garbage Incident")
    garbage_date = st.date_input("Date of Incident", key="garbage_date")
    garbage_location = st.text_input("Location",  key="garbage_location")
    garbage_description = st.text_area("Description",  key="garbage_description")
    garbage_photo = st.file_uploader("Upload Photo of Incident", type=["jpg", "jpeg", "png"],  key="garbage_image")

    if st.button("Report Garbage Incident"):
        time.sleep(.5)
        st.toast('Reported Garbage Incident')
        if garbage_date and garbage_location and garbage_description and garbage_photo:
            # Save image locally
            image_path = f'uploads/garbage/{garbage_date}.jpg'
            with open(image_path, 'wb') as f:
                f.write(garbage_photo.getvalue())

            save_incident_report("garbage", garbage_date, garbage_location, garbage_description, image_path)


# Sidebar with Chatbot
st.sidebar.title("Question Answering Chatbot using RoBERTa")

@st.cache_resource
def load_qa_model():
    model_name = "deepset/roberta-base-squad2"
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Create pipeline
        nlp = pipeline(
            'question-answering',
            model=model,
            tokenizer=tokenizer
        )
        return nlp
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return None

# Load the model
nlp = load_qa_model()

question = st.sidebar.text_input("Enter your question:")

context = '''
            Welcome to Seas the Day Chatbot! Ask any questions about our project.
            Seas the Day Chatbot is your virtual assistant dedicated to ocean conservation and environmental protection. Whether you have questions about oil spills, garbage patches, marine life, or eco-friendly initiatives, I'm here to help.

            Info about the project: It is hosted on Streamlit and it is a data visualization project of geospatial data (mapped using Folium) of oil spills and garbage patches.

            **How I Can Assist You:**
            - Provide information about recent oil spill incidents and their impact on marine life.
            - Share insights about garbage patches in the oceans and their environmental consequences.
            - Answer questions about eco-friendly practices and initiatives for ocean conservation.
            - Assist in identifying marine life and their habitats.
            - Offer guidance on reporting and responding to environmental incidents.

            - I'm here to provide information and assistance based on available data and knowledge up to my last update in November 2023.
            - For urgent or real-time environmental emergencies, please contact local authorities and environmental organizations.
            - Trained by Maha Kanakala
            Feel free to ask any questions related to ocean conservation, and I'll do my best to provide accurate and helpful answers!
'''

if question and nlp is not None:
    try:
        with st.spinner('Processing your question...'):
            answer = nlp(question=question, context=context)
            st.sidebar.write("Answer:", answer['answer'])
    except Exception as e:
        st.sidebar.error(f"Error processing question: {str(e)}")
elif question and nlp is None:
    st.sidebar.error("Chatbot is currently unavailable. Please try again later.")