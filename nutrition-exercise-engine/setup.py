"""
Setup script for Nutrition & Exercise Recommendation Engine
Run this script to initialize the project and create sample data
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_directory_structure():
    """Create necessary directories"""
    directories = ['data', 'models', 'exports', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def create_sample_nutrition_data():
    """Create comprehensive sample nutrition data"""
    nutrition_data = {
        'food_item': [
            # Fruits
            'Apple', 'Banana', 'Orange', 'Strawberries', 'Blueberries', 'Grapes', 'Mango', 'Pineapple',
            'Watermelon', 'Cantaloupe', 'Kiwi', 'Papaya', 'Pomegranate', 'Avocado',
            
            # Vegetables
            'Broccoli', 'Spinach', 'Carrots', 'Tomato', 'Cucumber', 'Bell Pepper', 'Onion', 'Garlic',
            'Sweet Potato', 'Potato', 'Cauliflower', 'Brussels Sprouts', 'Asparagus', 'Zucchini',
            
            # Proteins
            'Chicken Breast', 'Salmon', 'Tuna', 'Turkey', 'Beef', 'Pork', 'Eggs', 'Tofu',
            'Greek Yogurt', 'Cottage Cheese', 'Milk', 'Cheese', 'Beans', 'Lentils', 'Chickpeas',
            
            # Grains
            'Rice', 'Quinoa', 'Oats', 'Bread', 'Pasta', 'Barley', 'Bulgur', 'Cornmeal',
            
            # Nuts and Seeds
            'Almonds', 'Walnuts', 'Cashews', 'Peanuts', 'Sunflower Seeds', 'Chia Seeds', 'Flaxseeds',
            
            # Others
            'Olive Oil', 'Coconut Oil', 'Butter', 'Honey', 'Dark Chocolate'
        ],
        'calories_per_100g': [
            # Fruits
            52, 89, 47, 32, 57, 69, 60, 50, 30, 34, 61, 43, 83, 160,
            
            # Vegetables
            34, 23, 41, 18, 15, 31, 40, 149, 86, 77, 25, 43, 20, 17,
            
            # Proteins
            165, 208, 144, 104, 250, 242, 155, 76, 59, 98, 42, 113, 127, 116, 164,
            
            # Grains
            130, 368, 389, 265, 131, 354, 342, 365,
            
            # Nuts and Seeds
            579, 654, 553, 567, 584, 486, 534,
            
            # Others
            884, 862, 717, 304, 546
        ],
        'protein_g': [
            # Fruits
            0.3, 1.1, 0.9, 0.7, 0.7, 0.6, 0.8, 0.5, 0.6, 0.8, 1.1, 0.5, 1.7, 2.0,
            
            # Vegetables
            2.8, 2.9, 0.9, 0.9, 0.7, 1.0, 1.1, 6.4, 2.0, 2.0, 1.9, 3.4, 2.2, 1.2,
            
            # Proteins
            31.0, 25.4, 30.0, 29.0, 26.0, 26.0, 13.0, 8.1, 10.0, 11.1, 3.4, 25.0, 9.0, 9.0, 8.9,
            
            # Grains
            2.7, 14.1, 16.9, 9.0, 5.0, 12.5, 12.3, 8.1,
            
            # Nuts and Seeds
            21.2, 15.2, 18.2, 26.0, 20.8, 16.5, 18.3,
            
            # Others
            0.0, 0.0, 0.9, 0.3, 7.8
        ],
        'carbs_g': [
            # Fruits
            14.0, 23.0, 12.0, 8.0, 14.0, 17.0, 15.0, 13.0, 8.0, 8.0, 15.0, 11.0, 19.0, 9.0,
            
            # Vegetables
            7.0, 3.6, 10.0, 3.9, 3.6, 7.3, 9.3, 33.1, 20.0, 17.0, 5.0, 9.0, 3.9, 3.1,
            
            # Proteins
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 1.9, 3.6, 3.4, 5.0, 1.3, 23.0, 20.0, 27.4,
            
            # Grains
            28.0, 64.0, 66.0, 49.0, 25.0, 73.5, 76.0, 74.3,
            
            # Nuts and Seeds
            22.0, 14.0, 30.0, 16.0, 20.0, 42.1, 29.0,
            
            # Others
            0.0, 15.2, 0.1, 82.4, 46.4
        ],
        'fat_g': [
            # Fruits
            0.2, 0.3, 0.1, 0.3, 0.3, 0.2, 0.4, 0.1, 0.2, 0.2, 0.5, 0.3, 1.2, 15.0,
            
            # Vegetables
            0.4, 0.4, 0.2, 0.2, 0.1, 0.3, 0.1, 0.5, 0.1, 0.1, 0.3, 0.3, 0.1, 0.3,
            
            # Proteins
            3.6, 13.4, 1.0, 1.0, 15.0, 14.0, 11.0, 4.8, 0.4, 4.3, 1.0, 33.0, 0.5, 0.4, 2.6,
            
            # Grains
            0.3, 6.1, 6.9, 3.2, 1.1, 2.3, 1.3, 3.9,
            
            # Nuts and Seeds
            49.9, 65.2, 44.0, 49.2, 51.5, 30.7, 42.2,
            
            # Others
            100.0, 99.1, 81.1, 0.0, 30.0
        ],
        'fiber_g': [
            # Fruits
            2.4, 2.6, 2.4, 2.0, 2.4, 0.9, 1.6, 1.4, 0.4, 0.9, 3.0, 1.7, 4.0, 7.0,
            
            # Vegetables
            2.6, 2.2, 2.8, 1.2, 0.5, 2.5, 1.7, 2.1, 3.0, 2.2, 2.0, 3.8, 2.1, 1.0,
            
            # Proteins
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 6.4, 7.9, 12.2,
            
            # Grains
            0.4, 7.0, 10.6, 2.7, 1.8, 17.3, 18.3, 7.3,
            
            # Nuts and Seeds
            12.5, 6.7, 3.3, 8.5, 8.6, 34.4, 27.3,
            
            # Others
            0.0, 16.8, 0.0, 0.2, 11.0
        ],
        'category': [
            # Fruits
            'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit',
            'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit',
            
            # Vegetables
            'Vegetable', 'Vegetable', 'Vegetable', 'Vegetable', 'Vegetable', 'Vegetable', 
            'Vegetable', 'Vegetable', 'Vegetable', 'Vegetable', 'Vegetable', 'Vegetable', 
            'Vegetable', 'Vegetable',
            
            # Proteins
            'Protein', 'Protein', 'Protein', 'Protein', 'Protein', 'Protein', 'Protein', 
            'Protein', 'Dairy', 'Dairy', 'Dairy', 'Dairy', 'Legume', 'Legume', 'Legume',
            
            # Grains
            'Grain', 'Grain', 'Grain', 'Grain', 'Grain', 'Grain', 'Grain', 'Grain',
            
            # Nuts and Seeds
            'Nuts', 'Nuts', 'Nuts', 'Nuts', 'Seeds', 'Seeds', 'Seeds',
            
            # Others
            'Fat', 'Fat', 'Fat', 'Sweetener', 'Treat'
        ]
    }
    
    df = pd.DataFrame(nutrition_data)
    df.to_csv('data/food_nutrition.csv', index=False)
    print("Created comprehensive nutrition data: data/food_nutrition.csv")
    return df

def create_sample_exercise_data():
    """Create comprehensive sample exercise data"""
    exercise_data = {
        'exercise_type': [
            # Cardio
            'Walking', 'Running', 'Jogging', 'Cycling', 'Swimming', 'Rowing', 'Elliptical',
            'Stationary Bike', 'Treadmill', 'Jumping Rope', 'Dancing', 'Aerobics',
            
            # Strength Training
            'Weight Training', 'Bodyweight Exercises', 'Push-ups', 'Pull-ups', 'Squats',
            'Deadlifts', 'Bench Press', 'Resistance Training', 'Kettlebell Training',
            
            # Sports
            'Basketball', 'Tennis', 'Soccer', 'Volleyball', 'Badminton', 'Table Tennis',
            'Golf', 'Baseball', 'Football', 'Hockey',
            
            # Flexibility & Mind-Body
            'Yoga', 'Pilates', 'Stretching', 'Tai Chi', 'Meditation',
            
            # Outdoor Activities
            'Hiking', 'Rock Climbing', 'Mountain Biking', 'Kayaking', 'Surfing',
            
            # Combat Sports
            'Boxing', 'Martial Arts', 'Kickboxing', 'Wrestling',
            
            # Recreational
            'Bowling', 'Skateboarding', 'Rollerblading', 'Frisbee'
        ],
        'met_value': [
            # Cardio
            3.5, 8.0, 6.0, 6.8, 6.0, 8.5, 5.0, 4.0, 5.0, 10.0, 4.8, 6.0,
            
            # Strength Training
            6.0, 3.8, 3.8, 8.0, 5.0, 6.0, 6.0, 6.0, 8.0,
            
            # Sports
            6.5, 5.0, 7.0, 4.0, 4.5, 4.0, 3.5, 5.0, 8.0, 8.0,
            
            # Flexibility & Mind-Body
            2.5, 3.0, 2.3, 3.0, 1.0,
            
            # Outdoor Activities
            6.0, 8.0, 8.5, 5.0, 3.0,
            
            # Combat Sports
            8.0, 5.0, 8.0, 6.0,
            
            # Recreational
            3.0, 5.0, 7.0, 3.0
        ],
        'category': [
            # Cardio
            'Cardio', 'Cardio', 'Cardio', 'Cardio', 'Cardio', 'Cardio', 'Cardio',
            'Cardio', 'Cardio', 'Cardio', 'Cardio', 'Cardio',
            
            # Strength Training
            'Strength', 'Strength', 'Strength', 'Strength', 'Strength',
            'Strength', 'Strength', 'Strength', 'Strength',
            
            # Sports
            'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports',
            'Sports', 'Sports', 'Sports', 'Sports',
            
            # Flexibility & Mind-Body
            'Flexibility', 'Flexibility', 'Flexibility', 'Flexibility', 'Flexibility',
            
            # Outdoor Activities
            'Outdoor', 'Outdoor', 'Outdoor', 'Outdoor', 'Outdoor',
            
            # Combat Sports
            'Combat', 'Combat', 'Combat', 'Combat',
            
            # Recreational
            'Recreation', 'Recreation', 'Recreation', 'Recreation'
        ],
        'intensity': [
            # Cardio
            'Low', 'High', 'Moderate', 'Moderate', 'Moderate', 'High', 'Moderate',
            'Low', 'Moderate', 'High', 'Moderate', 'Moderate',
            
            # Strength Training
            'High', 'Moderate', 'Moderate', 'High', 'Moderate',
            'High', 'High', 'Moderate', 'High',
            
            # Sports
            'Moderate', 'Moderate', 'High', 'Moderate', 'Moderate', 'Moderate',
            'Low', 'Moderate', 'High', 'High',
            
            # Flexibility & Mind-Body
            'Low', 'Low', 'Low', 'Low', 'Low',
            
            # Outdoor Activities
            'Moderate', 'High', 'High', 'Moderate', 'Low',
            
            # Combat Sports
            'High', 'Moderate', 'High', 'High',
            
            # Recreational
            'Low', 'Moderate', 'Moderate', 'Low'
        ],
        'equipment_needed': [
            # Cardio
            'None', 'None', 'None', 'Bicycle', 'Pool', 'Rowing Machine', 'Elliptical Machine',
            'Stationary Bike', 'Treadmill', 'Jump Rope', 'None', 'None',
            
            # Strength Training
            'Weights', 'None', 'None', 'Pull-up Bar', 'None', 'Barbell', 'Bench', 'Resistance Bands', 'Kettlebell',
            
            # Sports
            'Ball', 'Racket', 'Ball', 'Ball', 'Racket', 'Paddle', 'Golf Clubs', 'Bat', 'Ball', 'Stick',
            
            # Flexibility & Mind-Body
            'Mat', 'Mat', 'Mat', 'None', 'None',
            
            # Outdoor Activities
            'None', 'Climbing Gear', 'Mountain Bike', 'Kayak', 'Surfboard',
            
            # Combat Sports
            'Gloves', 'None', 'Gloves', 'None',
            
            # Recreational
            'Bowling Ball', 'Skateboard', 'Rollerblades', 'Frisbee'
        ]
    }
    
    df = pd.DataFrame(exercise_data)
    df.to_csv('data/exercise_data.csv', index=False)
    print("Created comprehensive exercise data: data/exercise_data.csv")
    return df

def create_sample_users():
    """Create sample users for testing"""
    from database_manager import DatabaseManager
    
    db_manager = DatabaseManager()
    
    sample_users = [
        {
            'username': 'demo_user',
            'age': 28,
            'gender': 'Male',
            'height': 175,
            'weight': 70,
            'activity_level': 'Moderate',
            'health_goals': ['fitness_improvement', 'muscle_gain'],
            'dietary_preferences': ['vegetarian']
        },
        {
            'username': 'test_user',
            'age': 32,
            'gender': 'Female',
            'height': 165,
            'weight': 60,
            'activity_level': 'High',
            'health_goals': ['weight_loss', 'heart_health'],
            'dietary_preferences': ['gluten_free']
        }
    ]
    
    created_users = []
    for user in sample_users:
        user_id = db_manager.create_user(
            user['username'], user['age'], user['gender'], 
            user['height'], user['weight'], user['activity_level'],
            user['health_goals'], user['dietary_preferences']
        )
        if user_id:
            created_users.append((user_id, user['username']))
            print(f"Created sample user: {user['username']} (ID: {user_id})")
    
    return created_users

def create_sample_logs(created_users):
    """Create sample food and exercise logs for testing"""
    from database_manager import DatabaseManager
    
    db_manager = DatabaseManager()
    
    # Sample food items with nutrition info
    sample_foods = [
        {'food': 'Apple', 'calories': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2, 'fiber': 2.4},
        {'food': 'Chicken Breast', 'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'fiber': 0},
        {'food': 'Rice', 'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'fiber': 0.4},
        {'food': 'Broccoli', 'calories': 34, 'protein': 2.8, 'carbs': 7, 'fat': 0.4, 'fiber': 2.6},
        {'food': 'Salmon', 'calories': 208, 'protein': 25.4, 'carbs': 0, 'fat': 13.4, 'fiber': 0},
        {'food': 'Oats', 'calories': 389, 'protein': 16.9, 'carbs': 66, 'fat': 6.9, 'fiber': 10.6},
        {'food': 'Greek Yogurt', 'calories': 59, 'protein': 10, 'carbs': 3.6, 'fat': 0.4, 'fiber': 0},
        {'food': 'Quinoa', 'calories': 368, 'protein': 14.1, 'carbs': 64, 'fat': 6.1, 'fiber': 7}
    ]
    
    meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
    
    # Sample exercises
    sample_exercises = [
        {'exercise': 'Running', 'intensity': 'High'},
        {'exercise': 'Weight Training', 'intensity': 'High'},
        {'exercise': 'Walking', 'intensity': 'Low'},
        {'exercise': 'Yoga', 'intensity': 'Low'},
        {'exercise': 'Cycling', 'intensity': 'Moderate'},
        {'exercise': 'Swimming', 'intensity': 'Moderate'}
    ]
    
    # Create logs for the past 14 days
    for user_id, username in created_users:
        print(f"Creating sample logs for {username}...")
        
        for day in range(14):
            log_date = (datetime.now() - timedelta(days=day)).strftime('%Y-%m-%d')
            
            # Add 3-4 food logs per day
            daily_foods = random.sample(sample_foods, random.randint(3, 4))
            for i, food_data in enumerate(daily_foods):
                meal_type = meal_types[i % len(meal_types)]
                quantity = random.randint(50, 200)
                
                # Scale nutrition data by quantity
                scale_factor = quantity / 100
                
                db_manager.add_food_log(
                    user_id, log_date, meal_type, food_data['food'],
                    quantity,
                    food_data['calories'] * scale_factor,
                    food_data['protein'] * scale_factor,
                    food_data['carbs'] * scale_factor,
                    food_data['fat'] * scale_factor,
                    food_data['fiber'] * scale_factor
                )
            
            # Add 1-2 exercise logs per day (not every day)
            if random.random() > 0.3:  # 70% chance of exercise
                num_exercises = random.randint(1, 2)
                daily_exercises = random.sample(sample_exercises, num_exercises)
                
                for exercise_data in daily_exercises:
                    duration = random.randint(20, 60)
                    # Simple calorie calculation (5 calories per minute average)
                    calories_burned = duration * random.uniform(4, 8)
                    
                    db_manager.add_exercise_log(
                        user_id, log_date, exercise_data['exercise'],
                        duration, exercise_data['intensity'], calories_burned
                    )
        
        # Add some progress tracking data
        for week in range(2):  # Last 2 weeks
            progress_date = (datetime.now() - timedelta(weeks=week)).strftime('%Y-%m-%d')
            
            # Get user data for realistic progress
            user_data = db_manager.get_user(username)
            base_weight = user_data['weight']
            
            # Simulate slight weight changes
            weight_change = random.uniform(-0.5, 0.5)
            current_weight = base_weight + weight_change
            
            db_manager.add_progress_entry(
                user_id, progress_date, current_weight,
                random.uniform(15, 25),  # body fat percentage
                random.uniform(30, 45),  # muscle mass percentage
                random.randint(6, 9),    # energy level
                random.uniform(6, 9)     # sleep hours
            )
    
    print("Sample logs created successfully!")

def create_readme():
    """Create README file with setup and usage instructions"""
    readme_content = """# Nutrition & Exercise Recommendation Engine

## Overview
An AI-powered system that provides personalized nutrition and exercise recommendations based on user data, dietary habits, and physical activity patterns.

## Features
- **Personalized Analysis**: AI-powered analysis of dietary habits and physical activity
- **Smart Recommendations**: Machine learning algorithms for optimal nutrition and exercise plans
- **Progress Tracking**: Monitor health metrics and predict future progress
- **Cultural Awareness**: Recommendations consider cultural dietary preferences
- **Goal-Oriented**: Plans tailored to specific health goals
- **Data Export**: Download personalized plans and data

## Installation

### Requirements
- Python 3.8+
- pip package manager

### Setup Steps

1. **Clone/Download the project**
   ```bash
   # If using git
   git clone <repository-url>
   cd nutrition-exercise-engine
   
   # Or extract the downloaded files
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup script**
   ```bash
   python setup.py
   ```

4. **Start the application**
   ```bash
   streamlit run main_app.py
   ```

5. **Access the application**
   - Open your web browser
   - Go to `http://localhost:8501`

## Usage Guide

### 1. Registration
- Create a new account with your personal information
- Set your health goals and dietary preferences
- Specify your activity level and any restrictions

### 2. Data Logging
- **Food Logging**: Record your daily meals with nutritional information
- **Exercise Logging**: Log your workouts, duration, and intensity
- **Progress Tracking**: Monitor weight, energy levels, and other metrics

### 3. Get Recommendations
- Generate personalized nutrition and exercise recommendations
- View 7-day meal plans and weekly exercise routines
- See progress predictions based on your current plan

### 4. Download Plans
- Export your personalized nutrition and exercise plans
- Download complete plans in text format
- Save progress data for external analysis

## Sample Users
The setup creates two demo users for testing:
- **Username**: `demo_user` (Male, 28, vegetarian, fitness goals)
- **Username**: `test_user` (Female, 32, gluten-free, weight loss goals)

## Project Structure
```
nutrition-exercise-engine/
├── main_app.py                 # Main Streamlit application
├── database_manager.py         # Database operations
├── nutrition_analyzer.py       # Nutrition analysis module
├── activity_tracker.py         # Exercise analysis module
├── clustering_engine.py        # User clustering for recommendations
├── recommendation_engine.py    # Recommendation generation
├── progress_predictor.py       # Progress prediction models
├── visualization_utils.py      # Data visualization utilities
├── setup.py                   # Setup and initialization script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/                      # Data files
│   ├── food_nutrition.csv     # Nutrition database
│   └── exercise_data.csv      # Exercise database
├── models/                    # Trained ML models (auto-generated)
├── exports/                   # User data exports
└── logs/                      # Application logs
```

## Technology Stack
- **Backend**: Python, SQLite, Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

## Features Detail

### Machine Learning Components
1. **Clustering Engine**: Groups users with similar patterns using K-means
2. **Recommendation Engine**: Content-based and collaborative filtering
3. **Progress Predictor**: Linear regression and ensemble models for progress prediction
4. **Nutrition Analyzer**: Pattern recognition in eating habits
5. **Activity Tracker**: Exercise pattern analysis and recommendation

### Supported Goals
- Weight loss
- Muscle gain
- Heart health improvement
- Diabetes management
- Energy boost
- General fitness improvement

### Dietary Preferences
- Vegetarian
- Vegan
- Diabetic-friendly
- Hypertension-friendly
- Gluten-free
- Cultural preferences (Indian, Mediterranean, Asian, Western)

## Data Privacy
- All data is stored locally in SQLite database
- No external data sharing
- User controls all data exports and deletions
- Privacy-preserving machine learning algorithms

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **Database errors**: Delete `nutrition_exercise.db` and run setup again
3. **Port conflicts**: Change port in Streamlit with `streamlit run main_app.py --server.port 8502`

### Getting Help
- Check the application logs in the `logs/` directory
- Ensure Python 3.8+ is installed
- Verify all required packages are installed
- Check data files exist in the `data/` directory

## Development
To extend or modify the system:
1. Follow the modular architecture in the codebase
2. Add new recommendation algorithms in `recommendation_engine.py`
3. Extend nutrition data in `data/food_nutrition.csv`
4. Add new exercise types in `data/exercise_data.csv`

## License
This project is for educational and personal use.
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("Created README.md with setup instructions")
