import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, Float, String, Date, DateTime, ForeignKey
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from datetime import datetime

load_dotenv()

# Database connection
def get_database_url():
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    name = os.getenv('DB_NAME')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')

    # Debug
    print("Database connection parameters:")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Database: {name}")
    print(f"User: {user}")
    print(f"Password: {'*' * len(password) if password else 'Not set'}")

    if not all([host, port, name, user]):
        raise ValueError("Missing required database configuration. Please check your .env file.")

    try:
        port = int(port)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid port number: {port}. Must be a valid integer.")

    return f"postgresql://{user}:{password}@{host}:{port}/{name}"

# Create engine
def get_engine():
    try:
        url = get_database_url()
        return create_engine(url)
    except Exception as e:
        print(f"Error creating database engine: {str(e)}")
        raise

# Create tables
def create_tables():
    engine = get_engine()
    metadata = MetaData()

    # Marine Microplastic Density table
    Table('marine_microplastic_density', metadata,
          Column('id', Integer, primary_key=True),
          Column('latitude', Float),
          Column('longitude', Float),
          Column('total_pieces_l', Float),
          Column('year', Integer))

    # Historical Oil Spills table
    Table('oil_spills_historical', metadata,
          Column('id', Integer, primary_key=True),
          Column('date', Date),
          Column('latitude', Float),
          Column('longitude', Float),
          Column('location', String),
          Column('oil_spill_name', String),
          Column('barrels', Integer),
          Column('impact_regions', String),
          Column('organisms_affected', String))

    # Reported Oil Spills table
    Table('reported_oil_spills', metadata,
          Column('id', Integer, primary_key=True),
          Column('date', Date),
          Column('location', String),
          Column('description', String),
          Column('photo_path', String),
          Column('created_at', DateTime, default=datetime.utcnow))

    # Reported Garbage Incidents table
    Table('reported_garbage_incidents', metadata,
          Column('id', Integer, primary_key=True),
          Column('date', Date),
          Column('location', String),
          Column('description', String),
          Column('photo_path', String),
          Column('created_at', DateTime, default=datetime.utcnow))

    try:
        metadata.create_all(engine)
        return True
    except SQLAlchemyError as e:
        print(f"Error creating tables: {e}")
        return False

# Test database connection
def test_connection():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return False

# Import data from CSV files
def import_historical_data():
    try:
        engine = get_engine()
        
        # Import marine microplastic data
        df_plastic = pd.read_csv('./public/data/marine_microplastic_density.csv', encoding="latin-1")
        df_plastic.to_sql('marine_microplastic_density', engine, if_exists='replace', index=False)
        
        # Import oil spills data
        df_oil = pd.read_csv('./public/data/oilspills_1967-91.csv', encoding='latin-1')
        df_oil.to_sql('oil_spills_historical', engine, if_exists='replace', index=False)
        
        return True
    except Exception as e:
        print(f"Error importing historical data: {e}")
        return False

def get_table_info():
    """Get information about all tables in the database"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Get table names
            tables = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            
            print("\nDatabase Tables:")
            for table in tables:
                # Get row count for each table
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table[0]}")).scalar()
                print(f"- {table[0]}: {count} rows")
                
            return True
    except Exception as e:
        print(f"Error getting table info: {str(e)}")
        return False

def preview_tables():
    """Preview the contents of each table"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Preview historical data
            print("\nMarine Microplastic Density (First 3 rows):")
            result = conn.execute(text("SELECT * FROM marine_microplastic_density LIMIT 3"))
            for row in result:
                print(row)
            
            print("\nHistorical Oil Spills (First 3 rows):")
            result = conn.execute(text("SELECT * FROM oil_spills_historical LIMIT 3"))
            for row in result:
                print(row)
            
            # Check if there are any reported incidents
            print("\nReported Oil Spills:")
            result = conn.execute(text("SELECT COUNT(*) FROM reported_oil_spills"))
            count = result.scalar()
            print(f"Total reports: {count}")
            
            print("\nReported Garbage Incidents:")
            result = conn.execute(text("SELECT COUNT(*) FROM reported_garbage_incidents"))
            count = result.scalar()
            print(f"Total reports: {count}")
            
            return True
    except Exception as e:
        print(f"Error previewing tables: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing database connection...")
    # Test the database connection
    if test_connection():
        print("Database connection successful!")
        # Create tables
        if create_tables():
            print("Tables created successfully!")
            # Import historical data
            if import_historical_data():
                print("Historical data imported successfully!")
                # Show table information
                get_table_info()
                preview_tables()
            else:
                print("Failed to import historical data.")
        else:
            print("Failed to create tables.")
    else:
        print("Failed to connect to database. Please check your configuration.") 