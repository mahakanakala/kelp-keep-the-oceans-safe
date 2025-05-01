# Visit the [website](https://kelp-keep-the-oceans-safe.streamlit.app/)

This project utilizes geospatial data visualization techniques to map and analyze incidents of oil spills and garbage patches, providing valuable insights into their locations, impacts, and trends.

## 🏆 Winner of NJIT Hackathon Fall 23 (Best use of Google Cloud)
> Orignally used to use [GCP(Google Cloud Storage, Vertex AI)](https://github.com/mahakanakala/kelp-keep-the-oceans-safe/tree/7e6441f605ed990e71e0ab532afcd276d5320a1e) but switched to PostgreSQL & Supabase

## Table of Contents
- [Visit the website](#visit-the-website)
  - [🏆 Winner of NJIT Hackathon Fall 23 (Best use of Google Cloud)](#-winner-of-njit-hackathon-fall-23-best-use-of-google-cloud)
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
- **Question-Answering Chatbot:** Interact with a chatbot developed using VerteAI to ask questions related to ocean conservation and receive informative responses.

## Technologies Used
- **Streamlit:** Web framework for creating interactive and customizable dashboards.
- **Pandas:** Data manipulation library for handling and analyzing tabular data.
- **Folium:** Python wrapper for Leaflet.js, used for creating interactive maps.
- **PostgresSQL:** Database for storing incident reports, including details such as dates, locations, descriptions, and photo URLs.
- **Supabase:** Backend for managing both structured data and file storage with Row-Level Security (RLS) policies ensuring data integrity and access control.

## How to Use
**Interact with the App:**
   - Explore the interactive map to view incidents of oil spills and garbage patches.
   - Click on map markers to view detailed information about each incident.
   - Use the chatbot to ask questions about ocean conservation and environmental initiatives.
   - Report new incidents by providing relevant details and uploading photos.

