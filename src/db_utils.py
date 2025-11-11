import os
from contextlib import contextmanager
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv

load_dotenv()


def get_db_connection(use_dict_cursor=False):
    """
    Create a new database connection
    
    Args:
        use_dict_cursor: If True, returns results as dictionaries instead of tuples
        
    Returns:
        pymysql.Connection object
    """
    cursor_class = DictCursor if use_dict_cursor else None
    
    return pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=int(os.getenv('DB_PORT', '3306')),
        database=os.getenv('DB_NAME'),
        cursorclass=cursor_class
    )


@contextmanager
def db_connection(use_dict_cursor=False):
    """
    Context manager for database connections (auto-closes)
    
    Usage:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
            # connection automatically closed when leaving block
    
    Args:
        use_dict_cursor: If True, returns results as dictionaries instead of tuples
    """
    conn = get_db_connection(use_dict_cursor=use_dict_cursor)
    try:
        yield conn
    finally:
        conn.close()

