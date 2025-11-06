import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from config import DATABASE_CONFIG

class DatabaseManager:    
    def __init__(self, config=None):
        self.config = config or DATABASE_CONFIG
        self.db_type = self.config.get('type', 'sqlite')
        
        if self.db_type == 'sqlite':
            self.db_path = Path(self.config.get('path', 'database/face_gallery.db'))
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()
        else:
            raise NotImplementedError("PostgreSQL support requires additional setup")
    
    def _init_sqlite(self):
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS identities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    embedding BLOB NOT NULL,
                    num_samples INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_id INTEGER NOT NULL,
                    image_hash TEXT UNIQUE,
                    metadata JSON,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (identity_id) REFERENCES identities(id)
                )
            ''')
            
            conn.commit()
            conn.close()
            print(f"✓ SQLite database initialized at {self.db_path}")
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def add_identity(self, name: str, embedding: np.ndarray, num_samples: int = 1):
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Convert embedding to BLOB
            embedding_blob = embedding.tobytes()
            
            cursor.execute('''
                INSERT OR REPLACE INTO identities (name, embedding, num_samples, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (name, embedding_blob, num_samples, datetime.utcnow()))
            
            conn.commit()
            conn.close()
            print(f"✓ Added identity: {name}")
        except Exception as e:
            print(f"Error adding identity: {e}")
    
    def get_identity(self, name: str) -> Dict:
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, embedding, num_samples FROM identities WHERE name = ?
            ''', (name,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            id_, name, embedding_blob, num_samples = row
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            
            return {
                'id': id_,
                'name': name,
                'embedding': embedding,
                'num_samples': num_samples
            }
        except Exception as e:
            print(f"Error retrieving identity: {e}")
            return None
    
    def load_gallery(self) -> Dict:
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT name, embedding FROM identities ORDER BY id
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            embeddings = []
            identities = []
            
            for name, embedding_blob in rows:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                embeddings.append(embedding)
                identities.append(name)
            
            embeddings = np.array(embeddings) if embeddings else np.empty((0, 512), dtype=np.float32)
            
            return {
                'embeddings': embeddings,
                'identities': identities
            }
        except Exception as e:
            print(f"Error loading gallery: {e}")
            return {
                'embeddings': np.empty((0, 512), dtype=np.float32),
                'identities': []
            }
    
    def delete_identity(self, name: str):
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT id FROM identities WHERE name = ?', (name,))
            result = cursor.fetchone()
            
            if result is None:
                print(f"Identity {name} not found")
                return
            
            identity_id = result[0]
            
            # Delete face records
            cursor.execute('DELETE FROM face_records WHERE identity_id = ?', (identity_id,))
            
            # Delete identity
            cursor.execute('DELETE FROM identities WHERE id = ?', (identity_id,))
            
            conn.commit()
            conn.close()
            print(f"✓ Deleted identity: {name}")
        except Exception as e:
            print(f"Error deleting identity: {e}")
    
    def list_identities(self) -> List[Dict]:
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, num_samples, created_at, updated_at
                FROM identities ORDER BY name
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            identities = []
            for id_, name, num_samples, created_at, updated_at in rows:
                identities.append({
                    'id': id_,
                    'name': name,
                    'num_samples': num_samples,
                    'created_at': created_at,
                    'updated_at': updated_at
                })
            
            return identities
        except Exception as e:
            print(f"Error listing identities: {e}")
            return []
    
    def add_face_record(self, identity_name: str, image_hash: str, metadata: Dict = None):
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get identity ID
            cursor.execute('SELECT id FROM identities WHERE name = ?', (identity_name,))
            result = cursor.fetchone()
            
            if result is None:
                print(f"Identity {identity_name} not found")
                conn.close()
                return
            
            identity_id = result[0]
            metadata_json = json.dumps(metadata or {})
            
            cursor.execute('''
                INSERT INTO face_records (identity_id, image_hash, metadata)
                VALUES (?, ?, ?)
            ''', (identity_id, image_hash, metadata_json))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error adding face record: {e}")
    
    def get_statistics(self) -> Dict:
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM identities')
            num_identities = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM face_records')
            num_records = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(num_samples) FROM identities')
            total_samples = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'num_identities': num_identities,
                'num_face_records': num_records,
                'total_samples': total_samples,
                'db_size_mb': self.db_path.stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}