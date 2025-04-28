from src.database import create_tables, import_historical_data
   
def migrate():
    """Run database migrations"""
    if create_tables():
        if import_historical_data():
            print("Migration successful!")
        else:
            print("Failed to import data")
    else:
        print("Failed to create tables")

if __name__ == "__main__":
    migrate()