import sqlite3
from typing import List, Tuple
import uuid

# 데이터베이스 초기화 및 테이블 생성
def initialize_database(db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # 사용자 테이블 생성
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    face_data BLOB NOT NULL,
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role TEXT DEFAULT 'user'
);

    ''')
    
    # 접근 로그 테이블 생성
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS access_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        result TEXT NOT NULL,
        reason TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    ''')
    
    # 경고 이벤트 테이블 생성
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alert_events (
        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        event_type TEXT NOT NULL,
        description TEXT
    );
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# 사용자 추가
def add_user(name: str, face_data: bytes, db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    user_id = str(uuid.uuid4())  # UUID 생성
    cursor.execute('INSERT INTO users (user_id, name, face_data) VALUES (?, ?, ?)', (user_id, name, face_data))
    conn.commit()
    conn.close()

# (수정) 모든 사용자 조회
def get_all_users(db_name: str = "face_access_control.db") -> List[Tuple]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()
    conn.close()
    return users

# (추가) 사용자 이름으로 조회
def find_user_by_name(name: str, db_name: str = "face_access_control.db") -> Tuple or None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE name = ?', (name,))
    user = cursor.fetchone()  # 한 명만 조회한다면 fetchone
    conn.close()
    return user

# (추가) 사용자 이름으로 삭제
def delete_user_by_name(name: str, db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE name = ?', (name,))
    conn.commit()
    conn.close()

# 접근 로그 추가
def log_access(user_id: int, result: str, reason: str = None, db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO access_logs (user_id, result, reason) VALUES (?, ?, ?)', (user_id, result, reason))
    conn.commit()
    conn.close()

# 접근 로그 조회
def get_access_logs(db_name: str = "face_access_control.db") -> List[Tuple]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM access_logs')
    logs = cursor.fetchall()
    conn.close()
    return logs

# 경고 이벤트 추가
def add_alert(event_type: str, description: str = None, db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO alert_events (event_type, description) VALUES (?, ?)', (event_type, description))
    conn.commit()
    conn.close()

# 경고 이벤트 조회
def get_alert_events(db_name: str = "face_access_control.db") -> List[Tuple]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM alert_events')
    events = cursor.fetchall()
    conn.close()
    return events

# (기존) 사용자 user_id로 삭제
def delete_user(user_id: int, db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()
