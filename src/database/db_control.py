import sqlite3
import uuid
import pickle
from typing import List, Tuple, Optional

def initialize_database(db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # 사용자 테이블 (user_id는 TEXT로 UUID 저장)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        face_data BLOB NOT NULL,
        gesture_data BLOB,  -- 손동작 제스처 데이터 추가
        registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        role TEXT DEFAULT 'user'
    );
    ''')

    # 접근 로그 테이블 (user_id도 TEXT)
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

    # 경고 이벤트 테이블
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
def add_user(name: str, 
             face_data: List[List[Tuple[float, float, float]]],  # 여러 장이면 List[List[...]]
             gesture_data: Optional[List[List[Tuple[float, float, float]]]] = None,  # 손동작 제스처 데이터 추가
             db_name: str = "face_access_control.db",
             role: str = "user") -> str:
    """
    name: 사용자 이름
    face_data: 여러 장의 이미지 랜드마크를 저장하려면 List[List[Tuple]]] 구조를 권장
    gesture_data: 여러 장의 손동작 제스처 데이터를 저장하려면 List[List[Tuple]]] 구조를 권장
    role: 기본값 user
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    user_id = str(uuid.uuid4())  # 랜덤 UUID 생성
    pickled_face_data = pickle.dumps(face_data)  # BLOB로 직렬화
    pickled_gesture_data = pickle.dumps(gesture_data) if gesture_data else None  # 손동작 제스처 데이터 직렬화

    cursor.execute('''
        INSERT INTO users (user_id, name, face_data, gesture_data, role)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, name, pickled_face_data, pickled_gesture_data, role))

    conn.commit()
    conn.close()
    return user_id

# 모든 사용자 조회 (역직렬화 포함)
def get_all_users(db_name: str = "face_access_control.db") -> List[Tuple[str, str, List[List[Tuple[float, float, float]]], Optional[List[List[Tuple[float, float, float]]]], str]]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, name, face_data, gesture_data, role FROM users')
    rows = cursor.fetchall()
    conn.close()

    # rows = [(user_id(str), name(str), face_data(BLOB), gesture_data(BLOB), role(str)), ...]
    results = []
    for row in rows:
        user_id, name, face_blob, gesture_blob, role = row
        try:
            face_data = pickle.loads(face_blob)  # 얼굴 랜드마크 역직렬화
        except Exception:
            face_data = None
        try:
            gesture_data = pickle.loads(gesture_blob) if gesture_blob else None  # 손동작 제스처 역직렬화
        except Exception:
            gesture_data = None
        results.append((user_id, name, face_data, gesture_data, role))
    return results

# 사용자 이름으로 조회 (역직렬화 포함)
def find_user_by_name(name: str, db_name: str = "face_access_control.db") -> Optional[Tuple[str, str, List[List[Tuple[float, float, float]]], Optional[List[List[Tuple[float, float, float]]]], str]]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, name, face_data, gesture_data, role FROM users WHERE name = ?', (name,))
    row = cursor.fetchone()
    conn.close()
    if row:
        user_id, name, face_blob, gesture_blob, role = row
        try:
            face_data = pickle.loads(face_blob)
        except Exception:
            face_data = None
        try:
            gesture_data = pickle.loads(gesture_blob) if gesture_blob else None
        except Exception:
            gesture_data = None
        return (user_id, name, face_data, gesture_data, role)
    else:
        return None

# 사용자 이름으로 삭제
def delete_user_by_name(name: str, db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE name = ?', (name,))
    conn.commit()
    conn.close()

# 사용자 삭제 (user_id로)
def delete_user(user_id: str, db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()

# 접근 로그 추가
def log_access(user_id: str, result: str, reason: str = None, db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO access_logs (user_id, result, reason) VALUES (?, ?, ?)',
                   (user_id, result, reason))
    conn.commit()
    conn.close()

# 접근 로그 조회
def get_access_logs(db_name: str = "face_access_control.db") -> List[Tuple[int, str, str, str, Optional[str]]]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT log_id, user_id, attempt_time, result, reason FROM access_logs')
    rows = cursor.fetchall()
    conn.close()
    return rows

# 경고 이벤트 추가
def add_alert(event_type: str, description: str = None, db_name: str = "face_access_control.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO alert_events (event_type, description) VALUES (?, ?)',
                   (event_type, description))
    conn.commit()
    conn.close()

# 경고 이벤트 조회
def get_alert_events(db_name: str = "face_access_control.db") -> List[Tuple[int, str, str, Optional[str]]]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT alert_id, event_time, event_type, description FROM alert_events')
    rows = cursor.fetchall()
    conn.close()
    return rows
