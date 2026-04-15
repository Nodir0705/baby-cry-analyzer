import sqlite3
import json
from datetime import datetime
from config import settings


def init_db():
    conn = sqlite3.connect(settings.DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS events
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  cry_type TEXT,
                  synced INTEGER DEFAULT 0,
                  confidence REAL DEFAULT 0.0,
                  all_probs TEXT DEFAULT '{}')''')
    for col, dtype, default in [
        ('confidence', 'REAL', '0.0'),
        ('all_probs', 'TEXT', "'{}'"),
    ]:
        try:
            c.execute(f"ALTER TABLE events ADD COLUMN {col} {dtype} DEFAULT {default}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def log_event(cry_type, synced=False, confidence=0.0, all_probs=None):
    conn = sqlite3.connect(settings.DB_PATH)
    c = conn.cursor()
    local_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    probs_json = json.dumps(all_probs) if all_probs else '{}'
    c.execute(
        "INSERT INTO events (cry_type, synced, timestamp, confidence, all_probs) VALUES (?, ?, ?, ?, ?)",
        (cry_type, 1 if synced else 0, local_time, confidence, probs_json),
    )
    conn.commit()
    conn.close()
