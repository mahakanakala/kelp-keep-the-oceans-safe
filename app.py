import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import gcsfs
import time

# mapping
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import branca.colormap as cm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from google.cloud import storage

fs = storage.Client()

st.set_page_config(page_title="Kelp Keep the Oceans Safe",
                   page_icon=':otter:', layout='wide')

st.title("Kelp Keep the Oceans Safe")

GCS_BUCKET_NAME = 'seas-the-day-streamlit'
fs = gcsfs.GCSFileSystem(project='seas-the-day-404205')

# Import data
garbage_df = pd.read_csv('./public/data/marine_microplastic_density.csv', encoding="latin-1")
oil_spill_df = pd.read_csv('./public/data/oilspills_1967-91.csv',
                            encoding='latin-1'
                           )

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

# Garbage & Oil Heatmaps
m_1=folium.Map(location=start_loc_plastic,
              zoom_start=2,
              min_zoom=1.5,
              tiles='Open Street Map')

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
    HeatMap(data=oil_spill_df[['Latitude', 'Longitude', 'Barrels']].values, radius=10, blur=5,
            gradient={.2: 'beige', .6: 'brown', 1: 'black'}).add_to(m_1)
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

st.divider()

# Create Collection dfs
garbage_df = pd.DataFrame(columns=['Date', 'Photo', 'Location', 'Description'])
oil_spill_collection_df = pd.DataFrame(columns=['Date', 'Photo', 'Location', 'Description'])

# Collecting Information & uploading to Google Cloud
oil_col, garbage_col = st.columns(2)
with oil_col:
    st.header("Report Oil Spill Incident")
    oil_spill_date = st.date_input("Date of Incident")
    oil_spill_location = st.text_input("Location")
    oil_spill_description = st.text_area("Description")
    oil_spill_photo = st.file_uploader("Upload Photo of Incident", type=["jpg", "jpeg", "png"])

    if st.button("Report Oil Spill"):
        time.sleep(.5)
        st.toast('Reported Oil Spill Incident')
        if oil_spill_date and oil_spill_location and oil_spill_description and oil_spill_photo:
            oil_spill_image_bytes = io.BytesIO(oil_spill_photo.read())

            with fs.open(f'{GCS_BUCKET_NAME}/oil_spill/{oil_spill_date}.jpg', 'wb') as f:
                f.write(oil_spill_image_bytes.getvalue())

            oil_spill_collection_df = oil_spill_collection_df.append({
                'Date': oil_spill_date,
                'Photo': f'gs://{GCS_BUCKET_NAME}/oil_spill_collection/{oil_spill_date}.jpg',
                'Location': oil_spill_location,
                'Description': oil_spill_description
            }, ignore_index=True)

with garbage_col:
    st.header("Report Garbage Incident")
    garbage_date = st.date_input("Date of Incident", key="garbage_date")
    garbage_location = st.text_input("Location",  key="garbage_location")
    garbage_description = st.text_area("Description",  key="garbage_description")
    garbage_photo = st.file_uploader("Upload Photo of Incident", type=["jpg", "jpeg", "png"],  key="garbage_image")

    if st.button("Report Garbage Incident"):
        time.sleep(.5)
        st.toast('Reported Garbage Incident')
        if garbage_date and garbage_location and garbage_description and garbage_photo:
            garbage_image_bytes = io.BytesIO(garbage_photo.read())

            with fs.open(f'{GCS_BUCKET_NAME}/garbage_collection/{garbage_date}.jpg', 'wb') as f:
                f.write(garbage_image_bytes.getvalue())

            garbage_df = garbage_df.append({
                'Date': garbage_date,
                'Photo': f'gs://{GCS_BUCKET_NAME}/garbage_collection/{garbage_date}.jpg',
                'Location': garbage_location,
                'Description': garbage_description
            }, ignore_index=True)


# Sidebar with Chatbot
st.sidebar.title("Question Answering Chatbot using RoBERTa")

# Load the model and tokenizer
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the pipeline
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Sidebar input for user's question
question = st.sidebar.text_input("Enter your question:")

# Contextual information
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

if question:
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    QA_input = {
        'question': question,
        'context': context
    }
    answer = nlp(QA_input)
    st.sidebar.write("Answer:", answer['answer'])