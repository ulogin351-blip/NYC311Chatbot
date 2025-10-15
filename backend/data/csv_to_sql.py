import pandas as pd
import sqlite3
import os
from pathlib import Path

class CSVToSQL:
    def __init__(self, data_dir=None, db_path=None):
        """
        Initialize CSV to SQL converter
        
        Args:
            data_dir (str): Directory containing CSV files (defaults to script's directory)
            db_path (str): Path to SQLite database file (defaults to database.db in script's directory)
        """
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        
        # Set default paths relative to script location
        self.data_dir = Path(data_dir) if data_dir else script_dir
        self.db_path = db_path if db_path else script_dir / "database.db"
        self.connection = None
    
    def connect_db(self):
        """Create connection to SQLite database"""
        self.connection = sqlite3.connect(self.db_path)
        return self.connection
    
    def load_csv_to_sql(self, csv_filename, table_name=None):
        """
        Load a CSV file into SQLite database
        
        Args:
            csv_filename (str): Name of CSV file in data directory
            table_name (str): Name for the SQL table (defaults to filename without extension)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            csv_path = self.data_dir / csv_filename
            
            if not csv_path.exists():
                print(f"CSV file not found: {csv_path}")
                return False
            
            # Use filename as table name if not provided
            if table_name is None:
                table_name = csv_path.stem.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').lower()
                # Ensure table name doesn't start with a number (SQLite requirement)
                if table_name[0].isdigit():
                    table_name = 'table_' + table_name
            
            # Read CSV file (suppress mixed type warning)
            df = pd.read_csv(csv_path, low_memory=False)
            
            # Connect to database
            if self.connection is None:
                self.connect_db()
            
            # Load DataFrame to SQL table
            df.to_sql(table_name, self.connection, if_exists='replace', index=False)
            
            print(f"✅ Loaded {csv_filename} into table '{table_name}' ({len(df)} rows)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading {csv_filename}: {str(e)}")
            return False
    
    def load_all_csvs(self):
        """Load all CSV files from data directory into database"""
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            print("No CSV files found in data directory")
            return
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            self.load_csv_to_sql(csv_file.name)
    
    def query(self, sql_query):
        """
        Execute SQL query and return results
        
        Args:
            sql_query (str): SQL query to execute
        
        Returns:
            pandas.DataFrame: Query results
        """
        if self.connection is None:
            self.connect_db()
        
        try:
            result = pd.read_sql_query(sql_query, self.connection)
            return result
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
            return None
    
    def list_tables(self):
        """List all tables in the database"""
        if self.connection is None:
            self.connect_db()
        
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", 
            self.connection
        )
        return tables['name'].tolist()
    
    def describe_table(self, table_name):
        """Get column information for a table"""
        if self.connection is None:
            self.connect_db()
        
        try:
            columns = pd.read_sql_query(f"PRAGMA table_info({table_name})", self.connection)
            return columns[['name', 'type']]
        except Exception as e:
            print(f"❌ Error describing table {table_name}: {str(e)}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


if __name__ == "__main__":
    # Initialize converter
    converter = CSVToSQL()
    
    # Load all CSV files into database
    converter.load_all_csvs()
    
    # Show available tables
    print("\n📋 Available tables:")
    tables = converter.list_tables()
    for table in tables:
        print(f"  - {table}")
    
    # Close connection
    converter.close()
