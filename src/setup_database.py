import os
import pymysql
from pathlib import Path
from db_utils import get_db_connection
from dotenv import load_dotenv

load_dotenv()

def run_sql_file(cursor, filepath):
    """Execute SQL commands from a file"""
    print(f"Running {filepath.name}...")
    with open(filepath, 'r') as f:
        sql = f.read()
    
    # Remove comments and split by semicolons
    lines = []
    for line in sql.split('\n'):
        # Skip comment lines
        if line.strip().startswith('--'):
            continue
        lines.append(line)
    
    cleaned_sql = '\n'.join(lines)
    
    # Split and execute statements
    statements = cleaned_sql.split(';')
    for statement in statements:
        statement = statement.strip()
        if statement:
            try:
                cursor.execute(statement)
                print(f"  Executed statement")
            except Exception as e:
                print(f"  Error: {e}")

def main():
    db_name = os.getenv('DB_NAME')
    
    print("DATABASE SETUP (using .env credentials)")
    print(f"Database: {db_name}")
    print()
    
    conn = pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=int(os.getenv('DB_PORT', '3306'))
    )
    
    cursor = conn.cursor()
    
    print(f"Creating database {db_name}...")
    cursor.execute(f"""
        CREATE DATABASE IF NOT EXISTS {db_name}
        DEFAULT CHARACTER SET utf8mb4
        DEFAULT COLLATE utf8mb4_unicode_ci
    """)
    cursor.execute(f"USE {db_name}")
    conn.commit()
    print(f"Using database {db_name}")
    print()
    
    database_dir = Path(__file__).parent.parent / 'database'
    
    # Run setup scripts in order
    scripts = [
        'init_raw_contributions.sql',
        'init_contributors.sql',
        'init_contributions.sql',
        'init_candidates.sql',
        'init_committees.sql',
        'init_candidate_committee_links.sql'
    ]
    
    for script_name in scripts:
        script_path = database_dir / script_name
        if script_path.exists():
            run_sql_file(cursor, script_path)
            conn.commit()
        else:
            print(f"WARNING: {script_name} not found")
    
    print()
    
    print("DATABASE SETUP COMPLETE!")
    
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()

