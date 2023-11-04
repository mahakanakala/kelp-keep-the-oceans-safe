import streamlit as st
import pandas as pd
import numpy as np

# mapping
import geopandas as gpd
import folium
from streamlit_folium import st_folium, folium_static
from folium.plugins import MarkerCluster,HeatMap,HeatMapWithTime
import branca.colormap as colormap
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

st.set_page_config(page_title="Seas the Day with Donations",
                   page_icon=':world_map:', layout='wide')

st.title("Seas the Day with Donations")

st.markdown('''
## Understanding Marine Environmental Impact

Marine pollution, caused by oil spills and garbage patches, poses a significant threat to marine life. This interactive map provides a visual representation of these environmental challenges, using data from NOAA and advanced mapping techniques.

### Exploring Oil Spills & Garbage Patches
> Dive into the world of marine pollution and explore the global distribution of oil spills and garbage patches. The map showcases real-time data, allowing users to visualize the impact of these pollutants on our oceans.

### Unveiling the Impact
Oil spills disrupt marine ecosystems, coating wildlife in oil and leading to devastating consequences for aquatic organisms. Garbage patches, consisting of accumulated plastic waste, pose a long-lasting threat to marine life, causing ingestion and entanglement.

### Taking Action Together
Understanding the gravity of marine pollution is the first step towards conservation. By raising awareness and using innovative technologies, we can work towards a cleaner, healthier marine environment for future generations.

Let's make a difference. Explore the map, learn about the impact, and join the fight against marine pollution!
''')

# st.image()

# Import data
garbage_df = gpd.read_file('./public/data/marine_microplastic_density.csv')
oil_spill_df = pd.read_csv('./public/data/oilspills_1967-91.csv', encoding='latin-1')

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
to_datetime(oil_spill_df)

# Find the ideal frame for loading the render window
start_loc_plastic= (np.mean(garbage_df['Latitude']),np.mean(garbage_df['Longitude']))
start_loc_oil= (np.mean(oil_spill_df['Latitude']),np.mean(oil_spill_df['Longitude']))


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

#heatmap:
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

# folium_static(m_1)
st_data = st_folium(m_1, width=1500, returned_objects=[])
st_data = st_folium(m_2, width=1500, returned_objects=[])
# folium_static(m_2)

map_center = [(garbage_df['Latitude'].mean() + oil_spill_df['Latitude'].mean()) /2 ,
              (garbage_df['Longitude'].mean() + oil_spill_df['Longitude'].mean()) /2]

# Create a folium map
m = folium.Map(location=map_center, zoom_start=2)

# Add garbage patches data as a heatmap layer
garbage_heatmap_data = [[row['Latitude'], row['Longitude']] for index, row in garbage_df.iterrows()]
HeatMap(garbage_heatmap_data, radius=10).add_to(m)

# Add oil spills data as a separate layer with markers
for index, row in oil_spill_df.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup=row['Oil Spill Name']).add_to(m)

# Render the map using streamlit-folium
st_data = st_folium(m, width=1500, returned_objects=[])

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
            You are an LLM built to be knowledgeable on a data science project submission to the NJIT Hackathon. This is a project about the forecasting. The data set is from GCAG, this is the website 
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