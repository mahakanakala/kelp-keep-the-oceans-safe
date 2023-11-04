import streamlit as st
import pandas as pd
import numpy as np
import os
import re

# mapping
import geopandas as gpd
import folium
from streamlit_folium import st_folium, folium_static
from folium.plugins import MarkerCluster,HeatMap,HeatMapWithTime
import branca.colormap as colormap
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

st.set_page_config(page_title="Seas the Day with Donations",
                   page_icon=':map:', layout='wide')

st.title("Seas the Day with Donations")

st.markdown('''
            ### Oil Spills & Garabage Patches
            > Explore the oil spills and garbahe patches around the world, with data from NOA and mapping using Folium and Streamlit!
            
            ''')

# Import data
# plotting_file = gpd.read_file('./sea_micro.csv')
garbage_df = gpd.read_file('./data/marine_microplastic_density.csv')
oil_spill_df = pd.read_csv('./data/oilspills_1967-91.csv', encoding='latin-1')

def dms_to_dd(dms_value):
    # Replace commas with periods and split the DMS value into parts
    parts = dms_value.replace(',', '.').split('.')
    degrees = float(parts[0])
    minutes = 0
    seconds = 0
    # Check if there are minutes and seconds parts
    if len(parts) > 1:
        minutes = float(parts[1])
    if len(parts) > 2:
        seconds = float(parts[2])
    # Calculate the decimal degrees
    dd_value = degrees + (minutes / 60) + (seconds / 3600)
    return dd_value

oil_spill_df['Latitude'] = oil_spill_df['Latitude'].apply(dms_to_dd)
oil_spill_df['Longitude'] = oil_spill_df['Longitude'].apply(dms_to_dd)

# Create a geom obj for plotting
def get_geom(df,adn):
    '''get a geometry variable'''
    df[['Latitude','Longitude']]=df[['Latitude','Longitude']].astype(dtype=float)
    df[adn] = df[adn].astype(dtype=float)
#     df[['Latitude','Longitude']]=df[['Latitude','Longitude']].astype('float16')
    df['Geometry'] = pd.Series([(df.loc[i,'Latitude'],df.loc[i,'Longitude']) for i in range(len(df['Latitude']))])
    
def to_datetime(df,date_col='Date',frmt='%Y-%m-%d'):
    '''add_date col as datetime'''
    df[date_col] =pd.to_datetime(df[date_col],errors='coerce')
    df['year'] = df[date_col].dt.year
    
# oil_spill_gdf = gpd.GeoDataFrame(oil_spill_df, 
#                                  geometry=gpd.points_from_xy(oil_spill_df['Longitude'], oil_spill_df['Latitude'],))

# oil_spill_gdf.crs = "EPSG:4326"

get_geom(garbage_df,'Total_Pieces_L')
# get_geom(oil_spill_df, "Barrels")

#set to datetime'
to_datetime(garbage_df)
# to_datetime(geomar)
# to_datetime(sea_micro)

# find loc for max plastic
max_plas = garbage_df['Total_Pieces_L'].max()
idx= garbage_df[garbage_df['Total_Pieces_L']==max_plas].index
loc1= garbage_df.iloc[idx][['Latitude','Longitude']].values

# find loc for max oil
# max_plas_oil = oil_spill_df['Barrels'].max()
# idx_oil= oil_spill_df[plotting_file['Barrels']==max_plas].index
# loc1_oil= oil_spill_df.iloc[idx][['Latitude','Longitude']].values

start_loc_plastic= (np.mean(garbage_df['Latitude']),np.mean(garbage_df['Longitude']))
start_loc_oil= (np.mean(oil_spill_df['Latitude']),np.mean(oil_spill_df['Longitude']))

# start_loc_oil = (np.mean(oil_spill_df['Latitude']),np.mean(oil_spill_df['Longitude']))
#map
m_1=folium.Map(location=start_loc_plastic,
              tiles='Open Street Map',
              zoom_start=2,
              min_zoom=1.5)

#heatmap:
HeatMap(data=garbage_df[['Latitude','Longitude','Total_Pieces_L']].values, 
        # oil_spill_df[['Longtitude', 'Latitude', 'Density']],
        radius=10,
        blur=5).add_to(m_1)

#add area of highest concentration
folium.CircleMarker(location= (loc1[0][0],loc1[0][1]),
                  tooltip="<b>max plastic density</b>",
                  color='black',
                  radius=15).add_to(m_1)

# Add markers for oil spill incidents
# for _, row in oil_spill_df.iterrows():
#     folium.Marker(location=(float(row['Latitude'].replace(',', '.')), float(row['Longitude'].replace(',', '.'))),
#                   tooltip="<b>Oil Spill</b>",
#                   icon=folium.Icon(color='red')).add_to(m_1)

# folium.GeoJson(oil_spill_df,
#                name='Oil Spill Data',
#                tooltip=folium.GeoJsonTooltip(fields=['Longitude', 'Latitude'])  # Specify the fields you want to show in the tooltip
#                ).add_to(m_1)

# m_2=folium.Map(location=start_loc_oil,
#               tiles='Open Street Map',
#               zoom_start=2,
#               min_zoom=1.5)

# #heatmap:
# HeatMap(data=oil_spill_df[['Latitude','Longitude','Barrels']].values, 
#         # oil_spill_df[['Longtitude', 'Latitude', 'Density']],
#         radius=10,
#         blur=5).add_to(m_2)

m_2 = folium.Map(location=[oil_spill_df['Latitude'].mean(), oil_spill_df['Longitude'].mean()], zoom_start=2)

# Add markers for oil spills dynamically from the DataFrame
for _, row in oil_spill_df.iterrows():
    folium.Marker(location=[row['Latitude'], row['Longitude']],
                  tooltip=f"<b>{row['Oil Spill Name']}</b><br>Barrels: {row['Barrels']}").add_to(m_2)

# Display the map using Streamlit
folium_static(m_2)


# HeatMap(oil_data).add_to(m_1)
folium_static(m_1)
# folium_static(m_2)

# st_data = st_folium(m_1, width=725)

st.sidebar.header("Question Answering Chatbot")
st.sidebar.markdown('''
                    Made using [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2?context=You+are+a+LLM+built+to+knowledgable+on+my+data+science+project+submission.+This+is+a+project+about+the+forecasting+model+of+global+temperatures.+The+data+set+is+from+GCAG.+You+are+hosted+on+a+Streamlit+app.&question=What+is+the+data?). 
                    Ask any questions about our project!
                    ''')
# User input box
question = st.sidebar.text_input("Enter your question:")

# Contextualization
context = '''
            You are an LLM built to be knowledgeable on a data science project submission to the Rutgers Data Science Fall 23 Datathon 
            submitted by Maha, Nikhila, and Nivedha. This is a project about the forecasting. The data set is from GCAG, this is the website 
            that has more info on the data: https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/data-info. 
            You are hosted on a Streamlit app. This project is about a forecasting model of global temperatures built using Prophet by Facebook. 
            The raw data has Year and Anomaly Column. The anamoly representsanomaly means a departure from a reference value or long-term average. 
            A positive anomaly indicates that the observed temperature was warmer than the reference value, while a negative anomaly indicates 
            that the observed temperature was cooler than the reference value. The visualizations are built plotly - an interactive python 
            plotting library. The github link to the streamlit is: https://github.com/mahakanakala/datathon23_streamlit.
            The github link to the model and training of this project is: https://github.com/mahakanakala/datathon23.
            The processed data has Year, Anomaly, Month, and Season Column.
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