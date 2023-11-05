import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
from google.cloud import storage

# Streamlit APIs

# mapping
import geopandas as gpd
import folium
from streamlit_folium import st_folium, folium_static
from folium.plugins import MarkerCluster,HeatMap
import branca.colormap as colormap
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

st.set_page_config(page_title="Seas the Day with Donations",
                   page_icon=':world_map:', layout='wide')

st.title("Seas the Day with Donations")

# Import data
garbage_df = gpd.read_file('./public/data/marine_microplastic_density.csv')
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

**Oil**, an age-old fossil fuel, plays a crucial role in heating our homes, generating electricity, and driving various sectors of our economy. However, accidental oil spills in the ocean pose *significant* challenges. These spills can wreak havoc on marine life, spoil beach outings, and render seafood unsafe for consumption. Addressing these issues requires robust scientific efforts to clean up the oil, assess the pollution's impact, and aid the ocean in its recovery journey. This interactive map provides a visual representation of these environmental challenges, using data from NOAA and advanced mapping techniques.
''')

oil_image = Image.open('./public/images/turtle_oil_spill_copy.png')
plastic_image = Image.open('./public/images/turtle_plastic.jpeg')

images_column, description_column = st.columns(2)

with images_column:
    st.subheader("Oil Spills: Impact on Wildlife")
    st.image(oil_image, caption="MSNBC showcases a photo of the sea affected by the BP's oil spill")
    barrels_spilled = oil_spill_df['Barrels'].sum()
    recorded_spills = len(oil_spill_df)
    right_metric, left_metric = st.columns(2)
    with right_metric:
        st.metric(label="Number of Barrels of Oil Spilled", value=barrels_spilled, delta=-0.5)
    with left_metric:
        st.metric(label="Number of Recorded Spills", value=recorded_spills, delta=-0.5)

with description_column:
    st.subheader("Garbage Patches: Impact on Wildlife")
    st.image(plastic_image, caption="WWF showcases a photo of a turtle affected by plastic pollution")
    plastic_density = garbage_df['Total_Pieces_L'].sum()
    recorded_spills = len(oil_spill_df)
    right_metric, left_metric = st.columns(2)
    with right_metric:
        st.metric(label="Plastic Pieces Found in Oceans", value=plastic_density, delta=-0.5)
    with left_metric:
        st.metric(label="Number of Patches as big as the Pacific Garbage Circle", value=recorded_spills, delta=-0.5)

st.markdown('''

### Taking Action Together
Understanding the gravity of marine pollution is the first step towards conservation. By raising awareness and using innovative technologies, we can work towards a cleaner, healthier marine environment for future generations.

Let's make a difference. Explore the map, learn about the impact, and join the fight against marine pollution!
''')

# Create a geom obj for plotting
def get_geom(df,adn):
    '''get a geometry variable'''
    df[['Latitude','Longitude']]=df[['Latitude','Longitude']].astype(dtype=float)
    df[adn] = df[adn].astype(dtype=float)
    df['Geometry'] = pd.Series([(df.loc[i,'Latitude'],df.loc[i,'Longitude']) for i in range(len(df['Latitude']))])
    
def to_datetime(df,date_col='Date',frmt='%Y-%m-%d'):
    '''add_date col as datetime'''
    df[date_col] =pd.to_datetime(df[date_col],errors='coerce')
    df['year'] = df[date_col].dt.year

# Apply the get_geom function
get_geom(garbage_df,'Total_Pieces_L')
get_geom(oil_spill_df, "Barrels")

# Apply Date Time function
to_datetime(garbage_df)

# Find the ideal frame for loading the render window
start_loc_plastic= (np.mean(garbage_df['Latitude']),np.mean(garbage_df['Longitude']))
start_loc_oil= (np.mean(oil_spill_df['Latitude']),np.mean(oil_spill_df['Longitude']))

# ---
# Garbage & Oil Heatmaps
m_1=folium.Map(location=start_loc_plastic,
              zoom_start=2,
              min_zoom=1.5,
              tiles='Open Street Map')

HeatMap(data=garbage_df[['Latitude','Longitude','Total_Pieces_L']].values, 
        radius=10,
        blur=5).add_to(m_1)

m_2=folium.Map(location=start_loc_oil,
              tiles='Open Street Map',
              zoom_start=2,
              min_zoom=1.5)

HeatMap(data=oil_spill_df[['Latitude','Longitude','Barrels']].values, 
        # oil_spill_df[['Longtitude', 'Latitude', 'Density']],
        radius=10,
        blur=5).add_to(m_2)

m_2 = folium.Map(location=[oil_spill_df['Latitude'].mean(), oil_spill_df['Longitude'].mean()], zoom_start=2)

# Add markers for oil spills dynamically from the DataFrame
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
                  ).add_to(m_2)

latitude, longitude = 38.5, -145

# Define the radius of the circle in meters (adjust as needed)
radius = 75  

# Create a CircleMarker representing the Pacific Garbage Patch
folium.CircleMarker(location=[latitude, longitude],
                    radius=radius,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.2,
                    popup='Pacific Garbage Patch').add_to(m_1)

st_data = st_folium(m_1, width=1300, returned_objects=[])
st_data = st_folium(m_2, width=1300, returned_objects=[])


combined_map = folium.Map(location=start_loc_plastic, zoom_start=2, min_zoom=1.5, tiles='OpenStreetMap')

# Add the garbage heatmap to the combined map
HeatMap(data=garbage_df[['Latitude', 'Longitude', 'Total_Pieces_L']].values, radius=10, blur=5).add_to(combined_map)

# Add the oil spill markers and heatmap to the combined map
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
                  ).add_to(combined_map)

# Add the Pacific Garbage Patch CircleMarker
latitude, longitude = 38.5, -145
radius = 70  # Approximately 0.62 million square miles in meters
folium.CircleMarker(location=[latitude, longitude],
                    radius=radius,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.2,
                    popup='Pacific Garbage Patch').add_to(combined_map)

# Render the combined map
st_folium(combined_map, width=1300, height=600)









# Sidebar with Chatbot
st.sidebar.header("Question Answering Chatbot")
st.sidebar.markdown('''
                    Made using [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2?context=You+are+a+LLM+built+to+knowledgable+on+my+data+science+project+submission.+This+is+a+project+about+the+forecasting+model+of+global+temperatures.+The+data+set+is+from+GCAG.+You+are+hosted+on+a+Streamlit+app.&question=What+is+the+data?). 
                    Ask any questions about our project!
                    ''')
# User input box
question = st.sidebar.text_input("Enter your question:")

# Contextualization
context = '''
            You are an LLM built to be knowledgeable on a data science project submission to the NJIT Hackathon. This is a project about the data visualization. The data set is from GCAG, this is the website 
            that has more info on the data: https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/data-info. 
            You are hosted on a Streamlit app.
'''

# Model training
model_name = "deepset/roberta-base-squad2"

# Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Intializing deepset/roberta-base-squad2 model and getting answer instances from pipeline
if question:
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    QA_input = {
        'question': question,
        'context': context
    }
    answer = nlp(QA_input)
    st.sidebar.write("Answer:", answer['answer'])


# # Store the DataFrame as a CSV file
# scraped_data.to_csv('scraped_data.csv', index=False)

# # Upload the CSV file to Google Cloud Storage
# client = storage.Client()
# bucket = client.get_bucket('your-bucket-name')
# blob = bucket.blob('scraped_data.csv')
# blob.upload_from_filename('scraped_data.csv')