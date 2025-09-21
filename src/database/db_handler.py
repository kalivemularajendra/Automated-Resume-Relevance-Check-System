import sqlite3
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

class DatabaseHandler:
    def __init__(self, db_path: str = "resume_evaluations.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def initialize_db(self):
        """Create necessary tables"""
        self.connect()
        
        # Job Descriptions table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                role TEXT NOT NULL,
                location TEXT,
                description TEXT,
                parsed_data TEXT,
                min_exp INTEGER,
                max_exp INTEGER,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')
        
        # Evaluations table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                jd_id INTEGER,
                candidate_name TEXT,
                candidate_email TEXT,
                candidate_phone TEXT,
                resume_text TEXT,
                relevance_score REAL,
                hard_match_score REAL,
                semantic_match_score REAL,
                verdict TEXT,
                missing_skills TEXT,
                matched_skills TEXT,
                recommendations TEXT,
                experience_match TEXT,
                evaluated_at TIMESTAMP,
                FOREIGN KEY (jd_id) REFERENCES job_descriptions (id)
            )
        ''')
        
        # Analytics table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                metadata TEXT,
                recorded_at TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def save_jd(self, jd_data: Dict[str, Any]) -> int:
        """Save job description to database"""
        self.connect()
        
        self.cursor.execute('''
            INSERT INTO job_descriptions 
            (company, role, location, description, parsed_data, min_exp, max_exp, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            jd_data['company'],
            jd_data['role'],
            jd_data.get('location', ''),
            jd_data['description'],
            json.dumps(jd_data.get('parsed_data', {})),
            jd_data.get('min_exp', 0),
            jd_data.get('max_exp', 10),
            datetime.now(),
            datetime.now()
        ))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def save_evaluation(self, eval_data: Dict[str, Any]) -> int:
        """Save evaluation results to database"""
        self.connect()
        
        self.cursor.execute('''
            INSERT INTO evaluations
            (jd_id, candidate_name, candidate_email, candidate_phone, resume_text,
             relevance_score, hard_match_score, semantic_match_score, verdict,
             missing_skills, matched_skills, recommendations, experience_match, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            eval_data.get('jd_id'),
            eval_data['candidate_name'],
            eval_data.get('candidate_email', ''),
            eval_data.get('candidate_phone', ''),
            eval_data.get('resume_text', '')[:1000],  # Store first 1000 chars
            eval_data['relevance_score'],
            eval_data['hard_match_score'],
            eval_data['semantic_match_score'],
            eval_data['verdict'],
            json.dumps(eval_data.get('missing_skills', [])),
            json.dumps(eval_data.get('matched_skills', [])),
            json.dumps(eval_data.get('recommendations', [])),
            eval_data.get('experience_match', ''),
            datetime.now()
        ))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_all_jds(self) -> List[Dict[str, Any]]:
        """Get all job descriptions"""
        self.connect()
        
        self.cursor.execute('''
            SELECT * FROM job_descriptions
            ORDER BY created_at DESC
        ''')
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_evaluations_for_jd(self, jd_id: int) -> List[Dict[str, Any]]:
        """Get all evaluations for a specific JD"""
        self.connect()
        
        self.cursor.execute('''
            SELECT * FROM evaluations
            WHERE jd_id = ?
            ORDER BY relevance_score DESC
        ''', (jd_id,))
        
        rows = self.cursor.fetchall()
        results = []
        
        for row in rows:
            eval_dict = dict(row)
            # Parse JSON fields
            eval_dict['missing_skills'] = json.loads(eval_dict.get('missing_skills', '[]'))
            eval_dict['matched_skills'] = json.loads(eval_dict.get('matched_skills', '[]'))
            eval_dict['recommendations'] = json.loads(eval_dict.get('recommendations', '[]'))
            results.append(eval_dict)
        
        return results
    
    def get_evaluations_by_email(self, email: str) -> List[Dict[str, Any]]:
        """Get evaluations for a specific candidate by email"""
        self.connect()
        
        self.cursor.execute('''
            SELECT e.*, j.company, j.role 
            FROM evaluations e
            JOIN job_descriptions j ON e.jd_id = j.id
            WHERE e.candidate_email = ?
            ORDER BY e.evaluated_at DESC
        ''', (email,))
        
        rows = self.cursor.fetchall()
        results = []
        
        for row in rows:
            eval_dict = dict(row)
            # Parse JSON fields
            eval_dict['missing_skills'] = json.loads(eval_dict.get('missing_skills', '[]'))
            eval_dict['matched_skills'] = json.loads(eval_dict.get('matched_skills', '[]'))
            eval_dict['recommendations'] = json.loads(eval_dict.get('recommendations', '[]'))
            results.append(eval_dict)
        
        return results
    
    def get_all_evaluations(self) -> List[Dict[str, Any]]:
        """Get all evaluations"""
        self.connect()
        
        self.cursor.execute('''
            SELECT e.*, j.company, j.role
            FROM evaluations e
            LEFT JOIN job_descriptions j ON e.jd_id = j.id
            ORDER BY e.evaluated_at DESC
        ''')
        
        rows = self.cursor.fetchall()
        results = []
        
        for row in rows:
            eval_dict = dict(row)
            # Parse JSON fields
            eval_dict['missing_skills'] = json.loads(eval_dict.get('missing_skills', '[]'))
            eval_dict['matched_skills'] = json.loads(eval_dict.get('matched_skills', '[]'))
            eval_dict['recommendations'] = json.loads(eval_dict.get('recommendations', '[]'))
            results.append(eval_dict)
        
        return results
    
    def get_jd_count(self) -> int:
        """Get total number of JDs"""
        self.connect()
        self.cursor.execute('SELECT COUNT(*) FROM job_descriptions')
        return self.cursor.fetchone()[0]
    
    def get_resume_count(self) -> int:
        """Get total number of evaluated resumes"""
        self.connect()
        self.cursor.execute('SELECT COUNT(*) FROM evaluations')
        return self.cursor.fetchone()[0]
    
    def save_analytics(self, metric_name: str, metric_value: float, metadata: Dict = None):
        """Save analytics metrics"""
        self.connect()
        
        self.cursor.execute('''
            INSERT INTO analytics (metric_name, metric_value, metadata, recorded_at)
            VALUES (?, ?, ?, ?)
        ''', (
            metric_name,
            metric_value,
            json.dumps(metadata or {}),
            datetime.now()
        ))
        
        self.conn.commit()
    
    def __del__(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()