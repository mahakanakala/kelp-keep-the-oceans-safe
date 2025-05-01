import streamlit as st
from st_supabase_connection import SupabaseConnection, execute_query
import pytz
from datetime import datetime 
import urllib.parse
import uuid

def init_supabase():
    """Initialize Supabase connection"""
    if "supabase_client" not in st.session_state:
        st.session_state["supabase_client"] = st.connection(
            "supabase",
            type=SupabaseConnection,
            url=st.secrets["SUPABASE_URL"],
            key=st.secrets["SUPABASE_KEY"]
        )
    return st.session_state["supabase_client"]

def get_marine_debris_data():
    """Fetch marine microplastic debris data from Supabase"""
    conn = init_supabase()
    try:
        result = execute_query(
            conn.table("marine_microplastic_density").select("*"),
            ttl=0
        )
        return result.data
    except Exception as e:
        st.error(f"Error fetching marine debris data: {str(e)}")
        return []

def get_oil_spills_data():
    """Fetch historical oil spills data from Supabase"""
    conn = init_supabase()
    try:
        result = execute_query(
            conn.table("oilspills_1967_1991").select("*"),
            ttl=0
        )
        return result.data
    except Exception as e:
        st.error(f"Error fetching oil spills data: {str(e)}")
        return []

def get_table_info():
    """Get information about all tables"""
    conn = init_supabase()
    try:
        # Get info for each table using execute_query
        marine_result = execute_query(
            conn.table("marine_microplastic_density").select("*", count="exact"),
            ttl=0
        )
        oil_result = execute_query(
            conn.table("oilspills_1967_1991").select("*", count="exact"),
            ttl=0
        )
        
        table_info = {
            "marine_microplastic_density": {
                "count": marine_result.count if marine_result.count else 0,
                "sample": marine_result.data[:1] if marine_result.data else []
            },
            "oilspills_1967_1991": {
                "count": oil_result.count if oil_result.count else 0,
                "sample": oil_result.data[:1] if oil_result.data else []
            }
        }
        
        return table_info
    except Exception as e:
        st.error(f"Error getting table info: {str(e)}")
        return {}

def save_incident_report(incident_type, date, location, description, photo_file):
    """Save an incident report to Supabase using the initialized connection"""
    try:
        conn = init_supabase()
        UTC = pytz.utc
        current_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %z")

        bucket_id = "incident-photos"
        folder_name = "oil" if incident_type == "oil" else "garbage"

        # Sanitize the file name
        if photo_file:
            sanitized_file_name = f"{uuid.uuid4()}_{urllib.parse.quote(photo_file.name)}"
            destination_path = f"{folder_name}/{date}_{sanitized_file_name}"
        else:
            destination_path = None

        photo_url = None
        if photo_file:
            upload_result = conn.upload(
                bucket_id, "local", photo_file, destination_path
            )

            if not upload_result:
                raise Exception("Failed to upload photo to Supabase storage.")

            # Construct the public URL for the uploaded file
            photo_url = f"{st.secrets['SUPABASE_URL']}/storage/v1/object/public/{bucket_id}/{destination_path}"

        # Save the incident report to the database
        table_name = "reported_oil_spills" if incident_type == "oil" else "reported_garbage_incidents"
        data = {
            "date_time": current_time,
            "location": location if location else None,
            "description": description if description else None,
            "photo_path": photo_url,
        }

        result = conn.table(table_name).insert(data).execute()
        if not result.data:
            raise Exception("Failed to insert incident report into Supabase database.")

        return True
    except Exception as e:
        st.error(f"Error saving report: {str(e)}")
        return False