import streamlit as st
from st_supabase_connection import SupabaseConnection, execute_query

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

# def list_tables():
#     """List all available tables in the database"""
#     conn = init_supabase()
#     try:
#         st.write("Attempting to list all tables...")
#         # This query gets all table names from the Supabase information schema
#         result = conn.query("""
#             SELECT table_name 
#             FROM information_schema.tables 
#             WHERE table_schema = 'public'
#         """).execute()
#         st.write("Available tables:", [row['table_name'] for row in result.data])
#         return result.data
#     except Exception as e:
#         st.error(f"Error listing tables: {str(e)}")
#         return []

def test_connection():
    """Test the Supabase connection"""
    try:
        conn = init_supabase()
        # Try a simple query to verify connection
        result = execute_query(
            conn.table("marine_microplastic_density").select("*").limit(1),
            ttl=0
        )
            
        return True
    except Exception as e:
        st.error(f"❌ Connection test failed: {str(e)}")
        return False

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

def save_incident_report(incident_type, date, location, description, photo_path):
    """Save an incident report to Supabase"""
    conn = init_supabase()
    try:
        table_name = "reported_oil_spills" if incident_type == "oil" else "reported_garbage_incidents"
        
        data = {
            "date": str(date),
            "location": location,
            "description": description,
            "photo_path": photo_path
        }
        
        result = conn.table(table_name).insert(data).execute()
        return True if result.data else False
    except Exception as e:
        st.error(f"Error saving report: {str(e)}")
        return False

# def get_recent_reports():
#     """Get recent incident reports from both tables"""
#     conn = init_supabase()
#     try:
#         # Get recent oil spill reports
#         oil_reports = conn.table("reported_oil_spills") \
#                          .select("*") \
#                          .order("created_at", desc=True) \
#                          .limit(10) \
#                          .execute()
        
#         # Get recent garbage incident reports
#         garbage_reports = conn.table("reported_garbage_incidents") \
#                             .select("*") \
#                             .order("created_at", desc=True) \
#                             .limit(10) \
#                             .execute()
        
#         return oil_reports.data, garbage_reports.data
#     except Exception as e:
#         st.error(f"Error loading reports: {str(e)}")
#         return None, None 