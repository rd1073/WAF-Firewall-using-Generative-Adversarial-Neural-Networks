import sqlite3

DB_FILE = "firewall_logs.db"  # Update if the file name is different

def view_logs():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs")
    rows = cursor.fetchall()
    
    for row in rows:
        print(row)
    
    conn.close()

view_logs()
