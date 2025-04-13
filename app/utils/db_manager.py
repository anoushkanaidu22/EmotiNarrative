import sqlite3
import os
import json
from datetime import datetime

class DBManager:
    def __init__(self, db_path):
        #init database manager with path to SQLite database.
        #check directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        
    def get_connection(self):
        #Get a database connection
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  #return rows as dictionaries
        return conn
    
    def create_tables(self):
        #create database tables if they don't exist
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # media table: stores uploaded content information
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            path TEXT NOT NULL,
            upload_date TIMESTAMP NOT NULL
        )
        ''')
        
        # analysis table: stores emotion analysis results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            emotion_data TEXT NOT NULL,
            analysis_date TIMESTAMP NOT NULL,
            FOREIGN KEY (media_id) REFERENCES media (id)
        )
        ''')
        
        # validation table: stores user validation of analysis results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS validation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            validated_emotions TEXT NOT NULL,
            validation_date TIMESTAMP NOT NULL,
            FOREIGN KEY (analysis_id) REFERENCES analysis (id)
        )
        ''')
        
        # model version history: tracks model version changes
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT NOT NULL,
            version TEXT NOT NULL,
            created_date TIMESTAMP NOT NULL,
            accuracy REAL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_media(self, media_type, file_path):
        #add a new media entry and return its ID
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO media (type, path, upload_date) VALUES (?, ?, ?)",
            (media_type, file_path, datetime.now())
        )
        
        media_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return media_id
    
    def add_analysis(self, media_id, model_version, emotion_data):
        #add a new analysis entry and return its ID.
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO analysis (media_id, model_version, emotion_data, analysis_date) VALUES (?, ?, ?, ?)",
            (media_id, model_version, emotion_data, datetime.now())
        )
        
        analysis_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return analysis_id
    
    def add_validation(self, analysis_id, validated_emotions):
        #add a new validation entry and return its ID
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO validation (analysis_id, validated_emotions, validation_date) VALUES (?, ?, ?)",
            (analysis_id, validated_emotions, datetime.now())
        )
        
        validation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return validation_id
    
    def get_media(self, media_id):
        #Get media information by ID
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM media WHERE id = ?", (media_id,))
        media = cursor.fetchone()
        
        conn.close()
        return dict(media) if media else None
    
    def get_analysis(self, analysis_id):
        #get analysis information by ID
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM analysis WHERE id = ?", (analysis_id,))
        analysis = cursor.fetchone()
        
        conn.close()
        return dict(analysis) if analysis else None
    
    def add_model_version(self, model_type, version, accuracy=None):
        #add a new model version entry.
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO model_versions (model_type, version, created_date, accuracy) VALUES (?, ?, ?, ?)",
            (model_type, version, datetime.now(), accuracy)
        )
        
        conn.commit()
        conn.close()
    
    def get_pending_validations(self, model_type):
        #get validations that haven't been used for model improvement yet.
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT v.id, a.emotion_data, v.validated_emotions, m.type, m.path
        FROM validation v
        JOIN analysis a ON v.analysis_id = a.id
        JOIN media m ON a.media_id = m.id
        WHERE m.type = ? 
        """
        
        cursor.execute(query, (model_type,))
        validations = cursor.fetchall()
        
        conn.close()
        return [dict(v) for v in validations]
    
    def get_statistics(self):
        #get statistics for the dashboard
        conn = self.get_connection()
        cursor = conn.cursor()
        
        #total counts
        cursor.execute("SELECT COUNT(*) as total FROM media")
        total_media = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM media WHERE type = 'text'")
        total_text = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM media WHERE type = 'image'")
        total_image = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM validation")
        total_validations = cursor.fetchone()['total']
        
        #calculate average accuracy (agreement between model and validation)
        cursor.execute("""
        SELECT a.id, a.emotion_data, v.validated_emotions, m.type
        FROM analysis a
        JOIN validation v ON a.id = v.analysis_id
        JOIN media m ON a.media_id = m.id
        """)
        
        agreements = []
        for row in cursor.fetchall():
            try:
                model_emotions = json.loads(row['emotion_data'])
                validated_emotions = json.loads(row['validated_emotions'])
                media_type = row['type']
                
                #calculate agreement score based on media type
                if media_type == 'text':
                    #text: directly compare emotion dictionaries
                    if isinstance(model_emotions, dict) and isinstance(validated_emotions, dict):
                        #calculate agreement score aka how similar are the top emotions
                        model_top = max(model_emotions.items(), key=lambda x: x[1])
                        validated_top = max(validated_emotions.items(), key=lambda x: x[1])
                        
                        agreement = 1 if model_top[0] == validated_top[0] else 0
                        agreements.append(agreement)
                
                elif media_type == 'image':
                    #image: handle list of faces structure from DeepFace
                    if isinstance(model_emotions, list) and len(model_emotions) > 0:
                        #extract emotions from first face
                        face_emotions = None
                        if 'emotion' in model_emotions[0]:
                            face_emotions = model_emotions[0]['emotion']
                        else:
                            face_emotions = model_emotions[0]  # Direct emotion mapping
                            
                        if isinstance(face_emotions, dict):
                            #calculate agreement score
                            model_top = max(face_emotions.items(), key=lambda x: x[1])
                            validated_top = max(validated_emotions.items(), key=lambda x: x[1])
                            
                            agreement = 1 if model_top[0] == validated_top[0] else 0
                            agreements.append(agreement)
                    elif isinstance(model_emotions, dict):
                        #sometimes image emotions might be stored directly as a dictionary (?)
                        model_top = max(model_emotions.items(), key=lambda x: x[1])
                        validated_top = max(validated_emotions.items(), key=lambda x: x[1])
                        
                        agreement = 1 if model_top[0] == validated_top[0] else 0
                        agreements.append(agreement)
            
            except Exception as e:
                print(f"Error calculating agreement for analysis {row['id']}: {e}")
                #skip this entry if there's an error
        
        avg_accuracy = sum(agreements) / len(agreements) if agreements else 0
        
        #model vers history
        cursor.execute("""
        SELECT model_type, version, created_date, accuracy 
        FROM model_versions 
        ORDER BY created_date DESC
        """)
        
        model_versions = [dict(v) for v in cursor.fetchall()]
        
        conn.close()
        
        return {
            'total_media': total_media,
            'total_text': total_text,
            'total_image': total_image,
            'total_validations': total_validations,
            'avg_accuracy': avg_accuracy,
            'model_versions': model_versions
        }
        
    def mark_validations_used(self, validation_ids):
        #Mark validations as used for model improvement.
        #this imp doesn't actually delete validations
        #need to add a 'used' flag instead maybe idk
        pass