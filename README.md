# Kelp Keep the Oceans Safe

 This project utilizes geospatial data visualization techniques to map and analyze incidents of oil spills and garbage patches, providing valuable insights into their locations, impacts, and trends.

## Table of Contents

- [Kelp Keep the Oceans Safe](#kelp-keep-the-oceans-safe)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Technologies Used](#technologies-used)
  - [How to Use](#how-to-use)

## Introduction

It is designed to inform and engage users by visualizing real-time data about oil spills and garbage patches across the world's oceans. The project provides an interactive map interface where users can explore incidents, learn about their impact on marine life, and discover eco-friendly initiatives.

## Features

- **Interactive Map:** Explore the global distribution of oil spills and garbage patches using an interactive map.
- **Incident Details:** Click on map markers to view detailed information about specific incidents, including dates, locations, and impact regions.
- **User Contributions:** Allow users to report new oil spills and garbage incidents by uploading photos and descriptions.
- **Question-Answering Chatbot:** Interact with a chatbot to ask questions related to ocean conservation and receive informative responses.

## Technologies Used

- **Streamlit:** Web framework for creating interactive and customizable dashboards.
- **Pandas:** Data manipulation library for handling and analyzing tabular data.
- **Folium:** Python wrapper for Leaflet.js, used for creating interactive maps.
- **Google Cloud Storage:** Cloud-based storage service for storing user-uploaded images and data.
- **Transformers:** Library for natural language processing tasks, used for the question-answering chatbot.
- **Vertex AI:** Google Cloud's machine learning platform for building, training, and deploying machine learning models.

## How to Use

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
   The application will be accessible at `http://localhost:8501`.

3. **Interact with the App:**
   - Explore the interactive map to view incidents of oil spills and garbage patches.
   - Click on map markers to view detailed information about each incident.
   - Use the chatbot to ask questions about ocean conservation and environmental initiatives.
   - Report new incidents by providing relevant details and uploading photos.

