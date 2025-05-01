import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import time
from dotenv import load_dotenv
import streamlit.components.v1 as components

# mapping
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import branca.colormap as cm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

from src.supabase_data import (
    get_marine_debris_data,
    get_oil_spills_data,
    save_incident_report,
    init_supabase
)

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

# database
from src.supabase_data import (
    get_marine_debris_data,
    get_oil_spills_data
)
from sqlalchemy import text

os.makedirs('uploads/oil_spills', exist_ok=True)
os.makedirs('uploads/garbage', exist_ok=True)

# init Supabase connection
conn = init_supabase()

# load data from Supabase
@st.cache_data
def load_data():
    try:
        marine_debris_data = get_marine_debris_data()
        garbage_df = pd.DataFrame(marine_debris_data)
        
        oil_spills_data = get_oil_spills_data()
        oil_spill_df = pd.DataFrame(oil_spills_data)
        
        return garbage_df, oil_spill_df
    except Exception as e:
        st.error(f"Error loading data from Supabase: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


def get_recent_reports():
    """Get recent incident reports from both tables"""
    try:
        conn = init_supabase()
        if conn is None:
            return None, None
            
        oil_reports = pd.read_sql_query(
            "SELECT * FROM reported_oil_spills ORDER BY created_at DESC LIMIT 10",
            conn
        )
        
        garbage_reports = pd.read_sql_query(
            "SELECT * FROM reported_garbage_incidents ORDER BY created_at DESC LIMIT 10",
            conn
        )
        
        return oil_reports, garbage_reports
    except Exception as e:
        st.error(f"Error loading reports: {str(e)}")
        return None, None

st.title("Kelp Keep the Oceans Safe")

# Load data
garbage_df, oil_spill_df = load_data()
if garbage_df.empty or oil_spill_df.empty:
    st.error("Failed to load data. Please check the Supabase connection.")
    st.stop()

# Convert the Degrees Minutes Seconds format to Degrees for Folium plotting
def dms_to_dd(dms_value):
    """Convert European format DMS (degrees, minutes) to decimal degrees"""
    if pd.isna(dms_value):
        return None
    try:
        parts = str(dms_value).replace(',', '.').split('.')
        degrees = float(parts[0])
        minutes = 0
        seconds = 0
        if len(parts) > 1:
            minutes = float(parts[1])
        if len(parts) > 2:
            seconds = float(parts[2])
        dd_value = degrees + (minutes / 60) + (seconds / 3600)
        return dd_value
    except (ValueError, TypeError, IndexError):
        return None

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
    
    # Safely handle oil spill metrics
    barrels_spilled = 0
    if not oil_spill_df.empty and 'barrels' in oil_spill_df.columns:
        barrels_spilled = oil_spill_df['barrels'].sum()
    recorded_spills = len(oil_spill_df)
    
    right_metric, left_metric = st.columns(2)
    with right_metric:
        st.metric(label= ":oil_drum: Number of Barrels of Oil Spilled", value=barrels_spilled)
    with left_metric:
        st.metric(label=":chart_with_upwards_trend: Number of Recorded Spills", value=recorded_spills)
    st.metric(label="Average Number of Increase in Oil Spills/Year", value="2", delta="22% every year from 1929-1981", delta_color='inverse')

with description_column:
    st.subheader("Garbage Patches: Impact on Wildlife")
    st.image(plastic_image, caption="WWF showcases a photo of a turtle affected by plastic pollution")
    recorded_garbage = len(garbage_df)
    
    plastic_density = 0
    if not garbage_df.empty and 'total_pieces_l' in garbage_df.columns:
        garbage_df['total_pieces_l'] = pd.to_numeric(garbage_df['total_pieces_l'], errors='coerce')
        plastic_density = garbage_df['total_pieces_l'].sum()
    
    right_metric, left_metric = st.columns(2)
    with right_metric:
        st.metric(label=":roll_of_paper: Plastic Pieces Found in Oceans", value=plastic_density)
    with left_metric:
        st.metric(label=":world_map: Number of Patches as big as the Pacific Garbage Circle", value=recorded_garbage)
        
st.divider()

def prepare_heatmap(df, value_column, is_oil_spill=False):
    """Prepare DataFrame for heatmap plotting"""
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.warning(f"Missing required columns for heatmap. Available columns: {list(df.columns)}")
        return
        
    df = df.copy()
    
    if is_oil_spill:
        # convert European format coordinates for oil spill data
        df['latitude'] = df['latitude'].apply(dms_to_dd)
        df['longitude'] = df['longitude'].apply(dms_to_dd)
    else:
        # for marine debris data, just convert to float
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # convert value column to float if it exists
    if value_column in df.columns:
        df[value_column] = pd.to_numeric(df[value_column].astype(str).str.replace(',', '.'), errors='coerce')
    
    # drop any rows where conversion failed (resulted in None)
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # create geometry column for mapping
    df['geometry'] = pd.Series([(lat, lon) 
                               for lat, lon in zip(df['latitude'], df['longitude'])])
    
    return df

garbage_df = prepare_heatmap(garbage_df, 'total_pieces_l', is_oil_spill=False)
oil_spill_df = prepare_heatmap(oil_spill_df, 'barrels', is_oil_spill=True)

def to_datetime(df, date_col='date', frmt=None):
    """Convert date column to datetime and extract year
    Args:
        df (pd.DataFrame): Input dataframe
        date_col (str): Name of the date column
        frmt (str): Format string for date parsing. If None, will try multiple formats
    Returns:
        pd.DataFrame: DataFrame with converted dates and added year column
    """
    if frmt is None:
        # Try different date formats based on the data source
        if 'oil_spill' in df.columns or 'barrels' in df.columns:
            # For oil spill data (DD.MM.YY format)
            df[date_col] = pd.to_datetime(df[date_col], format='%d.%m.%y', errors='coerce')
            # Fix years for dates before 2000
            df['year'] = df[date_col].dt.year.apply(lambda x: x + 1900 if x and x < 100 else x)
            # Update dates with corrected years
            df[date_col] = pd.to_datetime(df.apply(
                lambda row: row[date_col].replace(year=row['year']) if pd.notnull(row[date_col]) else row[date_col],
                axis=1
            ))
        else:
            # For other data (assuming ISO format YYYY-MM-DD)
            df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
            df['year'] = df[date_col].dt.year
    else:
        df[date_col] = pd.to_datetime(df[date_col], format=frmt, errors='coerce')
        df['year'] = df[date_col].dt.year
    
    return df

# Convert date columns to datetime
if not garbage_df.empty and 'date' in garbage_df.columns:
    garbage_df = to_datetime(garbage_df)

if not oil_spill_df.empty and 'date' in oil_spill_df.columns:
    oil_spill_df = to_datetime(oil_spill_df)

# Set default map location if DataFrames are empty
if garbage_df.empty:
    start_loc_plastic = (0, 0)
else:
    start_loc_plastic = (np.mean(garbage_df['latitude']), np.mean(garbage_df['longitude']))

if oil_spill_df.empty:
    start_loc_oil = (0, 0)
else:
    start_loc_oil = (np.mean(oil_spill_df['latitude']), np.mean(oil_spill_df['longitude']))

if start_loc_plastic == (0, 0) and start_loc_oil == (0, 0):
    map_center = (20, 0) 
else:
    map_center = start_loc_plastic if start_loc_plastic != (0, 0) else start_loc_oil

m_1 = folium.Map(location=map_center,
                 zoom_start=2,
                 min_zoom=1.5,
                 tiles='CartoDB positron',
                 attr="<a href=https://endless-sky.github.io/>Endless Sky</a>")

st.markdown('''
## Visualizing the Impact of Oil Spills and Garbage Patches
*Use the multi-select button to view the layers/attributes*
            ''')

# add custom CSS to reduce spacing
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
col1, col2 = st.columns(2)
with col1:
    show_heatmap_garbage = st.checkbox("Show Garbage Heatmap")
    show_garbage_markers = st.checkbox("Show Garbage Markers")
with col2:
    show_heatmap_oil = st.checkbox("Show Oil Spill Heatmap")
    show_oil_markers = st.checkbox("Show Oil Spill Markers")
    
if show_heatmap_garbage:
    if not garbage_df.empty and all(col in garbage_df.columns for col in ['latitude', 'longitude', 'total_pieces_l']):
        HeatMap(data=garbage_df[['latitude', 'longitude', 'total_pieces_l']].values, radius=10, blur=5).add_to(m_1)
        colormap = cm.LinearColormap(colors=['blue', 'lightgreen', 'yellow', 'orange', 'red'],
                                   vmin=garbage_df['total_pieces_l'].min(),
                                   vmax=garbage_df['total_pieces_l'].max())
        colormap.add_to(m_1)
    else:
        st.warning("Missing required columns for garbage heatmap")

if show_heatmap_oil:
    if not oil_spill_df.empty and all(col in oil_spill_df.columns for col in ['latitude', 'longitude', 'barrels']):
        HeatMap(data=oil_spill_df[['latitude', 'longitude', 'barrels']].values, radius=10, blur=5).add_to(m_1)
        colormap = cm.LinearColormap(colors=['blue', 'lightgreen', 'yellow', 'orange', 'red'],
                                   vmin=oil_spill_df['barrels'].min(),
                                   vmax=oil_spill_df['barrels'].max())
        colormap.add_to(m_1)
    else:
        st.warning("Missing required columns for oil spill heatmap")

if show_oil_markers:
    if not oil_spill_df.empty and all(col in oil_spill_df.columns for col in ['latitude', 'longitude', 'barrels', 'date']):
        for idx, row in oil_spill_df.iterrows():
            popup_text = f"""
            <b>Oil Spill</b><br>
            Date: {row['date']}<br>
            Amount Spilled: {row['barrels']} barrels<br>
            """
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=popup_text,
                color='red',
                fill=True
            ).add_to(m_1)
    else:
        st.warning("Missing required columns for oil spill markers")

if show_garbage_markers:
    if not garbage_df.empty and all(col in garbage_df.columns for col in ['latitude', 'longitude', 'total_pieces_l', 'date']):
        for idx, row in garbage_df.iterrows():
            popup_text = f"""
            <b>Marine Debris</b><br>
            Date: {row['date']}<br>
            Total Pieces: {row['total_pieces_l']}<br>
            """
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=popup_text,
                color='blue',
                fill=True
            ).add_to(m_1)
    else:
        st.warning("Missing required columns for garbage markers")

st_folium(m_1, width=1300)

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
        time.sleep(0.5)
        st.toast('Reported Oil Spill Incident')
        success = save_incident_report(
            "oil",
            oil_spill_date if oil_spill_date else None,
            oil_spill_location.strip() if oil_spill_location.strip() else None,
            oil_spill_description.strip() if oil_spill_description.strip() else None,
            oil_spill_photo,
        )
        if success:
            st.success("Oil spill incident reported successfully!")
        else:
            st.error("Failed to report oil spill incident.")

with garbage_col:
    st.subheader("Report Garbage Incident")
    garbage_date = st.date_input("Date of Incident", key="garbage_date")
    garbage_location = st.text_input("Location", key="garbage_location")
    garbage_description = st.text_area("Description", key="garbage_description")
    garbage_photo = st.file_uploader("Upload Photo of Incident", type=["jpg", "jpeg", "png"], key="garbage_image")

    if st.button("Report Garbage Incident"):
        time.sleep(0.5)
        st.toast('Reported Garbage Incident')
        success = save_incident_report(
            "garbage",
            garbage_date if garbage_date else None,
            garbage_location.strip() if garbage_location.strip() else None,
            garbage_description.strip() if garbage_description.strip() else None,
            garbage_photo,
        )
        if success:
            st.success("Garbage incident reported successfully!")
        else:
            st.error("Failed to report garbage incident.")

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