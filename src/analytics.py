import sqlite3
import datetime
from pathlib import Path

# Analytics DB Path
DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "analytics.db"

def init_db():
    """Initializes the SQLite database for storing analytics logs."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Create the query_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_query TEXT,
            answer TEXT,
            source_type TEXT,
            retrieval_time_sec REAL,
            llm_time_sec REAL,
            total_time_sec REAL,
            input_tokens INTEGER,
            output_tokens INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

def log_query(user_query: str, answer: str, source_type: str, 
              retrieval_time_sec: float, llm_time_sec: float, 
              input_tokens: int, output_tokens: int):
    """
    Logs a single user query to the database.
    """
    total_time_sec = retrieval_time_sec + llm_time_sec
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO query_logs (
            timestamp, user_query, answer, source_type, 
            retrieval_time_sec, llm_time_sec, total_time_sec, 
            input_tokens, output_tokens
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user_query,
        answer,
        source_type,
        retrieval_time_sec,
        llm_time_sec,
        total_time_sec,
        input_tokens,
        output_tokens
    ))
    
    conn.commit()
    conn.close()

def get_analytics_data():
    """
    Retrieves all analytics data as a list of dictionaries.
    """
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM query_logs ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    
    conn.close()
    return [dict(row) for row in rows]

# Initialize database on import
init_db()
