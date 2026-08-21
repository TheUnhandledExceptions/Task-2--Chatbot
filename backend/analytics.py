import sqlite3
import json
import numpy as np
from typing import Dict, Any

DB_PATH = "latency_analytics.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS latencies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stt_latency REAL,
            retrieval_latency REAL,
            generation_latency REAL,
            total_latency REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_latency(timings: Dict[str, float]):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO latencies (stt_latency, retrieval_latency, generation_latency, total_latency)
        VALUES (?, ?, ?, ?)
    ''', (timings.get("stt", 0.0), timings.get("retrieval", 0.0), timings.get("generation", 0.0), timings.get("total", 0.0)))
    conn.commit()
    conn.close()

def get_analytics() -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT stt_latency, retrieval_latency, generation_latency, total_latency FROM latencies')
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return {"error": "No data"}
        
    totals = [r[3] for r in rows]
    
    return {
        "count": len(totals),
        "total_pipeline": {
            "p50": np.percentile(totals, 50),
            "p70": np.percentile(totals, 70),
            "p100": np.percentile(totals, 100),
        }
    }

init_db()
