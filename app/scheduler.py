import sqlite3

def init_db():
    conn = sqlite3.connect('appointments.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            time TEXT,
            doctor TEXT
        )
    ''')
    conn.commit()
    conn.close()

def schedule_appointment(name, date, time, doctor):
    conn = sqlite3.connect('appointments.db')
    c = conn.cursor()
    c.execute("INSERT INTO appointments (name, date, time, doctor) VALUES (?, ?, ?, ?)",
              (name, date, time, doctor))
    conn.commit()
    conn.close()
