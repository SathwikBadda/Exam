import sqlite3
import pandas as pd
import json
from datetime import datetime
import os

class DatabaseManager:
    def __init__(self, db_path="nutrition_exercise.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                age INTEGER,
                gender TEXT,
                height REAL,
                weight REAL,
                activity_level TEXT,
                health_goals TEXT,
                dietary_preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Food logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS food_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                date TEXT,
                meal_type TEXT,
                food_item TEXT,
                quantity REAL,
                calories REAL,
                protein REAL,
                carbs REAL,
                fat REAL,
                fiber REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Exercise logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exercise_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                date TEXT,
                exercise_type TEXT,
                duration INTEGER,
                intensity TEXT,
                calories_burned REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                date TEXT,
                nutrition_plan TEXT,
                exercise_plan TEXT,
                goals TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Progress tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS progress_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                date TEXT,
                weight REAL,
                body_fat_percentage REAL,
                muscle_mass REAL,
                energy_level INTEGER,
                sleep_hours REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, username, age, gender, height, weight, activity_level, health_goals, dietary_preferences):
        """Create a new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, age, gender, height, weight, activity_level, health_goals, dietary_preferences)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (username, age, gender, height, weight, activity_level, json.dumps(health_goals), json.dumps(dietary_preferences)))
            
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()
    
    def get_user(self, username):
        """Get user by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'age': user[2],
                'gender': user[3],
                'height': user[4],
                'weight': user[5],
                'activity_level': user[6],
                'health_goals': json.loads(user[7]) if user[7] else [],
                'dietary_preferences': json.loads(user[8]) if user[8] else []
            }
        return None
    
    def add_food_log(self, user_id, date, meal_type, food_item, quantity, calories, protein, carbs, fat, fiber):
        """Add food log entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO food_logs (user_id, date, meal_type, food_item, quantity, calories, protein, carbs, fat, fiber)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, date, meal_type, food_item, quantity, calories, protein, carbs, fat, fiber))
        
        conn.commit()
        conn.close()
    
    def add_exercise_log(self, user_id, date, exercise_type, duration, intensity, calories_burned):
        """Add exercise log entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO exercise_logs (user_id, date, exercise_type, duration, intensity, calories_burned)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, date, exercise_type, duration, intensity, calories_burned))
        
        conn.commit()
        conn.close()
    
    def save_recommendation(self, user_id, date, nutrition_plan, exercise_plan, goals):
        """Save recommendation plans"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO recommendations (user_id, date, nutrition_plan, exercise_plan, goals)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, date, json.dumps(nutrition_plan), json.dumps(exercise_plan), json.dumps(goals)))
        
        conn.commit()
        conn.close()
    
    def get_user_food_logs(self, user_id, days=30):
        """Get user's food logs for the last n days"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM food_logs 
            WHERE user_id = ? 
            ORDER BY date DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=[user_id, days * 4])  # Assuming 4 meals per day
        conn.close()
        return df
    
    def get_user_exercise_logs(self, user_id, days=30):
        """Get user's exercise logs for the last n days"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM exercise_logs 
            WHERE user_id = ? 
            ORDER BY date DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=[user_id, days])
        conn.close()
        return df
    
    def get_user_recommendations(self, user_id, limit=10):
        """Get user's recommendations"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM recommendations 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=[user_id, limit])
        conn.close()
        return df
    
    def add_progress_entry(self, user_id, date, weight, body_fat_percentage, muscle_mass, energy_level, sleep_hours):
        """Add progress tracking entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO progress_tracking (user_id, date, weight, body_fat_percentage, muscle_mass, energy_level, sleep_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, date, weight, body_fat_percentage, muscle_mass, energy_level, sleep_hours))
        
        conn.commit()
        conn.close()
    
    def get_user_progress(self, user_id):
        """Get user's progress data"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM progress_tracking 
            WHERE user_id = ? 
            ORDER BY date DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=[user_id])
        conn.close()
        return df