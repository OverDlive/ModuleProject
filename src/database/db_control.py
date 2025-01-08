import sqlite3
import uuid
import pickle
from typing import List, Tuple, Optional

def initialize_database(db_name: str = "face_access_control.db"):
    """
    데이터베이스를 초기화하고 필요한 테이블들을 생성합니다.
    
    Args:
        db_name (str): 사용할 데이터베이스 파일 이름. 기본값은 "face_access_control.db".
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # 사용자 테이블 생성 (face_data와 gesture_data를 BLOB으로 저장)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        face_data BLOB NOT NULL,
        gesture_data BLOB,  
        registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        role TEXT DEFAULT 'user'
    );
    ''')

    # 접근 로그 테이블 생성
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

    # 테이블 생성 후 변경 사항 저장
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def add_user(name: str, 
             face_data: List[List[Tuple[float, float, float]]],   # 여러 장의 얼굴 랜드마크 데이터
             gesture_data: Optional[List[List[Tuple[float, float, float]]]] = None,  # 여러 장의 손동작 랜드마크 데이터 (옵션)
             db_name: str = "face_access_control.db",
             role: str = "user") -> str:
    """
    새로운 사용자를 데이터베이스에 추가합니다.
    
    Args:
        name (str): 사용자 이름.
        face_data (List[List[Tuple[float, float, float]]]): 여러 장의 얼굴 랜드마크 데이터.
        gesture_data (Optional[List[List[Tuple[float, float, float]]]]): 여러 장의 손동작 랜드마크 데이터. 기본값은 None.
        db_name (str): 사용할 데이터베이스 파일 이름. 기본값은 "face_access_control.db".
        role (str): 사용자 역할. 기본값은 'user'.
    
    Returns:
        str: 생성된 사용자 ID(UUID).
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 고유한 사용자 ID 생성
    user_id = str(uuid.uuid4())

    # 얼굴 데이터 직렬화 (BLOB 저장)
    pickled_face_data = pickle.dumps(face_data)

    # 손동작 데이터 직렬화 (옵션)
    pickled_gesture_data = pickle.dumps(gesture_data) if gesture_data else None

    # 사용자 데이터 삽입
    cursor.execute('''
        INSERT INTO users (user_id, name, face_data, gesture_data, role)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, name, pickled_face_data, pickled_gesture_data, role))

    # 변경 사항 저장 후 연결 종료
    conn.commit()
    conn.close()
    return user_id

def get_all_users(db_name: str = "face_access_control.db") -> List[Tuple[str, str, List[List[Tuple[float, float, float]]], Optional[List[List[Tuple[float, float, float]]]], str]]:
    """
    모든 사용자를 조회하고, 얼굴 및 손동작 데이터를 역직렬화하여 반환합니다.
    
    Args:
        db_name (str): 사용할 데이터베이스 파일 이름. 기본값은 "face_access_control.db".
    
    Returns:
        List[Tuple[str, str, List[List[Tuple[float, float, float]]], Optional[List[List[Tuple[float, float, float]]]], str]]:
            사용자 ID, 이름, 얼굴 데이터, 손동작 데이터, 역할의 리스트.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 모든 사용자 데이터 조회
    cursor.execute('''
    SELECT user_id, name, face_data, gesture_data, role FROM users
    ''')

    users = []
    for row in cursor.fetchall():
        user_id = row[0]
        name = row[1]
        face_data = pickle.loads(row[2])  # 얼굴 데이터 역직렬화
        gesture_data = pickle.loads(row[3]) if row[3] else None  # 손동작 데이터 역직렬화
        role = row[4]
        users.append((user_id, name, face_data, gesture_data, role))

    conn.close()
    return users

def find_user_by_name(name: str, db_name: str = "face_access_control.db") -> Optional[Tuple[str, str, List[List[Tuple[float, float, float]]], Optional[List[List[Tuple[float, float, float]]]], str]]:
    """
    사용자 이름으로 사용자를 조회하고, 얼굴 및 손동작 데이터를 역직렬화하여 반환합니다.
    
    Args:
        name (str): 조회할 사용자 이름.
        db_name (str): 사용할 데이터베이스 파일 이름. 기본값은 "face_access_control.db".
    
    Returns:
        Optional[Tuple[str, str, List[List[Tuple[float, float, float]]], Optional[List[List[Tuple[float, float, float]]]], str]]:
            사용자 ID, 이름, 얼굴 데이터, 손동작 데이터, 역할의 튜플 또는 None.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 이름으로 사용자 조회
    cursor.execute('''
    SELECT user_id, name, face_data, gesture_data, role FROM users WHERE name = ?
    ''', (name,))

    row = cursor.fetchone()
    conn.close()
    if row:
        user_id = row[0]
        name = row[1]
        face_data = pickle.loads(row[2]) if row[2] else None  # 얼굴 데이터 역직렬화
        gesture_data = pickle.loads(row[3]) if row[3] else None  # 손동작 데이터 역직렬화
        role = row[4]
        return (user_id, name, face_data, gesture_data, role)
    return None

def delete_user_by_name(name: str, db_name: str = "face_access_control.db"):
    """
    사용자 이름으로 사용자를 삭제합니다.
    
    Args:
        name (str): 삭제할 사용자 이름.
        db_name (str): 사용할 데이터베이스 파일 이름. 기본값은 "face_access_control.db".
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 이름으로 사용자 삭제
    cursor.execute('DELETE FROM users WHERE name = ?', (name,))
    
    # 변경 사항 저장 후 연결 종료
    conn.commit()
    conn.close()

def delete_user(user_id: str, db_name: str = "face_access_control.db"):
    """
    사용자 ID로 사용자를 삭제합니다.
    
    Args:
        user_id (str): 삭제할 사용자 ID.
        db_name (str): 사용할 데이터베이스 파일 이름. 기본값은 "face_access_control.db".
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 사용자 ID로 사용자 삭제
    cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
    
    # 변경 사항 저장 후 연결 종료
    conn.commit()
    conn.close()

def log_access(user_id: str, result: str, reason: str = None, db_name: str = "face_access_control.db"):
    """
    사용자의 접근 시도를 로그에 기록합니다.
    
    Args:
        user_id (str): 접근한 사용자 ID.
        result (str): 접근 결과 (예: 'SUCCESS', 'FAIL').
        reason (str, optional): 접근 실패 사유. 기본값은 None.
        db_name (str): 사용할 데이터베이스 파일 이름. 기본값은 "face_access_control.db".
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 접근 로그 삽입
    cursor.execute('INSERT INTO access_logs (user_id, result, reason) VALUES (?, ?, ?)',
                   (user_id, result, reason))
    
    # 변경 사항 저장 후 연결 종료
    conn.commit()
    conn.close()

def get_access_logs(db_name: str = "face_access_control.db") -> List[Tuple[int, str, str, str, Optional[str]]]:
    """
    모든 접근 로그를 조회합니다.
    
    Args:
        db_name (str): 사용할 데이터베이스 파일 이름. 기본값은 "face_access_control.db".
    
    Returns:
        List[Tuple[int, str, str, str, Optional[str]]]:
            로그 ID, 사용자 ID, 시도 시간, 결과, 사유의 리스트.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 모든 접근 로그 조회
    cursor.execute('SELECT log_id, user_id, attempt_time, result, reason FROM access_logs')
    rows = cursor.fetchall()
    conn.close()
    return rows

def add_alert(event_type: str, description: str = None, db_name: str = "face_access_control.db"):
    """
    경고 이벤트를 데이터베이스에 추가합니다.
    
    Args:
        event_type (str): 이벤트 유형 (예: 'UNAUTHORIZED_ACCESS').
        description (str, optional): 이벤트 설명. 기본값은 None.
        db_name (str): 사용할 데이터베이스 파일 이름. 기본값은 "face_access_control.db".
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 경고 이벤트 삽입
    cursor.execute('INSERT INTO alert_events (event_type, description) VALUES (?, ?)',
                   (event_type, description))
    
    # 변경 사항 저장 후 연결 종료
    conn.commit()
    conn.close()

def get_alert_events(db_name: str = "face_access_control.db") -> List[Tuple[int, str, str, Optional[str]]]:
    """
    모든 경고 이벤트를 조회합니다.
    
    Args:
        db_name (str): 사용할 데이터베이스 파일 이름. 기본값은 "face_access_control.db".
    
    Returns:
        List[Tuple[int, str, str, Optional[str]]]:
            경고 ID, 이벤트 시간, 유형, 설명의 리스트.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 모든 경고 이벤트 조회
    cursor.execute('SELECT alert_id, event_time, event_type, description FROM alert_events')
    rows = cursor.fetchall()
    conn.close()
    return rows