import sqlite3
import uuid
import pickle
from typing import List, Tuple, Optional

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
    
    # 접근 로그 테이블 생성 (user_id를 TEXT로 변경)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS access_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
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

# 사용자 추가 (직렬화 포함)
def add_user(name: str, face_data: List[Tuple[float, float, float]], db_name: str = "face_access_control.db", role: str = "user"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    user_id = str(uuid.uuid4())  # UUID 생성
    
    # face_data를 직렬화하여 BLOB로 저장
    pickled_face_data = pickle.dumps(face_data)
    
    cursor.execute('INSERT INTO users (user_id, name, face_data, role) VALUES (?, ?, ?, ?)', (user_id, name, pickled_face_data, role))
    conn.commit()
    conn.close()
    return user_id  # 생성된 user_id 반환 (필요 시 활용)

# 모든 사용자 조회 (역직렬화 포함)
def get_all_users(db_name: str = "face_access_control.db") -> List[Tuple[str, str, List[Tuple[float, float, float]], str]]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, name, face_data, role FROM users')
    users = cursor.fetchall()
    conn.close()
    
    # face_data 역직렬화
    deserialized_users = []
    for user in users:
        user_id, name, face_data_blob, role = user
        try:
            face_data = pickle.loads(face_data_blob)
        except (pickle.UnpicklingError, EOFError) as e:
            face_data = None  # 또는 적절한 기본값 설정
        deserialized_users.append((user_id, name, face_data, role))
    
    return deserialized_users

# 사용자 이름으로 조회 (역직렬화 포함)
def find_user_by_name(name: str, db_name: str = "face_access_control.db") -> Optional[Tuple[str, str, List[Tuple[float, float, float]], str]]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, name, face_data, role FROM users WHERE name = ?', (name,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        user_id, name, face_data_blob, role = row
        try:
            face_data = pickle.loads(face_data_blob)
        except (pickle.UnpicklingError, EOFError) as e:
            face_data = None  # 또는 적절한 기본값 설정
        return (user_id, name, face_data, role)
    else:
        return None

# 사용자 이름으로 삭제
def delete_user_by_name(name: str, db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE name = ?', (name,))
    conn.commit()
    conn.close()

# 접근 로그 추가
def log_access(user_id: str, result: str, reason: str = None, db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO access_logs (user_id, result, reason) VALUES (?, ?, ?)', (user_id, result, reason))
    conn.commit()
    conn.close()

# 접근 로그 조회
def get_access_logs(db_name: str = "face_access_control.db") -> List[Tuple[int, str, str, str, Optional[str]]]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT log_id, user_id, attempt_time, result, reason FROM access_logs')
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
def get_alert_events(db_name: str = "face_access_control.db") -> List[Tuple[int, str, str, Optional[str]]]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT alert_id, event_time, event_type, description FROM alert_events')
    events = cursor.fetchall()
    conn.close()
    return events

# 사용자 삭제 (user_id로)
def delete_user(user_id: str, db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()
