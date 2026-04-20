import sqlite3

def create_table():

    conn = sqlite3.connect("churn.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenure INTEGER,
        monthly_charges REAL,
        total_charges REAL,
        prediction TEXT
    )
    """)

    conn.commit()
    conn.close()