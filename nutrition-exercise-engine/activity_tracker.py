import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ActivityTracker:
    def __init__(self, exercise_data_path="data/exercise_data.csv"):
        self.exercise_data_path = exercise_data_path
        self.exercise_data = self.load_exercise_data()
        self.met_values = self.load_met_values()
    
    def load_exercise_data(self):
        """Load exercise dataset"""
        try:
            df = pd.read_csv(self.exercise_data_path)
            return df
        except FileNotFoundError:
            # Create sample exercise data if file doesn't exist
            return self.create_sample_exercise_data()
    
    def create_sample_exercise_data(self):
        """Create sample exercise data for demonstration"""
        sample_data = {
            'exercise_type': [
                'Walking', 'Running', 'Cycling', 'Swimming', 'Weight Training', 'Yoga', 'Pilates',
                'Basketball', 'Tennis', 'Soccer', 'Dancing', 'Hiking', 'Rowing', 'Elliptical',
                'Jumping Rope', 'Rock Climbing', 'Martial Arts', 'Boxing', 'Aerobics', 'Stretching',
                'Volleyball', 'Badminton', 'Golf', 'Bowling', 'Skateboarding'
            ],
            'met_value': [
                3.5, 8.0, 6.8, 6.0, 6.0, 2.5, 3.0, 6.5, 5.0, 7.0, 4.8, 6.0, 8.5, 5.0,
                10.0, 8.0, 5.0, 8.0, 6.0, 2.3, 4.0, 4.5, 3.5, 3.0, 5.0
            ],
            'category': [
                'Cardio', 'Cardio', 'Cardio', 'Cardio', 'Strength', 'Flexibility', 'Flexibility',
                'Sports', 'Sports', 'Sports', 'Recreation', 'Outdoor', 'Cardio', 'Cardio',
                'Cardio', 'Outdoor', 'Combat', 'Combat', 'Cardio', 'Flexibility',
                'Sports', 'Sports', 'Recreation', 'Recreation', 'Recreation'
            ],
            'intensity': [
                'Low', 'High', 'Moderate', 'Moderate', 'High', 'Low', 'Low', 'Moderate', 'Moderate',
                'High', 'Moderate', 'Moderate', 'High', 'Moderate', 'High', 'High', 'Moderate',
                'High', 'Moderate', 'Low', 'Moderate', 'Moderate', 'Low', 'Low', 'Moderate'
            ],
            'equipment_needed': [
                'None', 'None', 'Bicycle', 'Pool', 'Weights', 'Mat', 'Mat', 'Ball', 'Racket',
                'Ball', 'None', 'None', 'Machine', 'Machine', 'Rope', 'Gear', 'None', 'Gloves',
                'None', 'Mat', 'Ball', 'Racket', 'Clubs', 'None', 'Board'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv('data/exercise_data.csv', index=False)
        return df
    
    def load_met_values(self):
        """Load MET values for calculating calories burned"""
        return dict(zip(self.exercise_data['exercise_type'], self.exercise_data['met_value']))
    
    def analyze_activity_logs(self, exercise_logs_df, user_weight=70):
        """Analyze user's exercise logs and identify activity patterns"""
        if exercise_logs_df.empty:
            return self.get_default_analysis()
        
        # Calculate calories burned
        exercise_logs_df = self.calculate_calories_burned(exercise_logs_df, user_weight)
        
        # Analyze activity patterns
        activity_patterns = self.analyze_activity_patterns(exercise_logs_df)
        
        # Identify fitness gaps
        fitness_gaps = self.identify_fitness_gaps(exercise_logs_df)
        
        # Generate exercise recommendations
        recommendations = self.generate_exercise_recommendations(activity_patterns, fitness_gaps)
        
        return {
            'activity_patterns': activity_patterns,
            'fitness_gaps': fitness_gaps,
            'recommendations': recommendations,
            'total_calories_burned': exercise_logs_df['calories_burned'].sum()
        }
    
    def calculate_calories_burned(self, exercise_logs_df, user_weight):
        """Calculate calories burned using MET values"""
        def calc_calories(row):
            met_value = self.met_values.get(row['exercise_type'], 5.0)  # Default MET value
            duration_hours = row['duration'] / 60  # Convert minutes to hours
            calories = met_value * user_weight * duration_hours
            return calories
        
        if 'calories_burned' not in exercise_logs_df.columns or exercise_logs_df['calories_burned'].isna().all():
            exercise_logs_df['calories_burned'] = exercise_logs_df.apply(calc_calories, axis=1)
        
        return exercise_logs_df
    
    def analyze_activity_patterns(self, exercise_logs_df):
        """Analyze activity patterns and habits"""
        patterns = {}
        
        # Exercise frequency
        exercise_frequency = len(exercise_logs_df) / 30 if len(exercise_logs_df) > 0 else 0  # Assuming 30 days
        patterns['exercises_per_week'] = exercise_frequency * 7
        
        # Most common exercises
        common_exercises = exercise_logs_df['exercise_type'].value_counts().head(5).to_dict()
        patterns['common_exercises'] = common_exercises
        
        # Average duration
        avg_duration = exercise_logs_df['duration'].mean() if not exercise_logs_df.empty else 0
        patterns['avg_duration_minutes'] = avg_duration
        
        # Total weekly duration
        total_duration = exercise_logs_df['duration'].sum()
        patterns['total_weekly_duration'] = total_duration
        
        # Intensity distribution
        intensity_distribution = exercise_logs_df['intensity'].value_counts(normalize=True).to_dict()
        patterns['intensity_distribution'] = intensity_distribution
        
        # Exercise category analysis
        if not exercise_logs_df.empty:
            exercise_logs_df = exercise_logs_df.merge(
                self.exercise_data[['exercise_type', 'category']], 
                on='exercise_type', 
                how='left'
            )
            category_distribution = exercise_logs_df['category'].value_counts(normalize=True).to_dict()
            patterns['category_distribution'] = category_distribution
        
        return patterns
    
    def identify_fitness_gaps(self, exercise_logs_df):
        """Identify fitness gaps based on recommended guidelines"""
        gaps = {}
        
        # WHO recommendations: 150 minutes moderate or 75 minutes vigorous per week
        total_duration = exercise_logs_df['duration'].sum()
        
        if total_duration < 150:  # Less than 150 minutes per week
            gaps['cardio_duration'] = {
                'current': total_duration,
                'recommended': 150,
                'deficit': 150 - total_duration
            }
        
        # Check for exercise variety
        unique_exercises = exercise_logs_df['exercise_type'].nunique()
        if unique_exercises < 3:
            gaps['exercise_variety'] = {
                'current': unique_exercises,
                'recommended': 3,
                'deficit': 3 - unique_exercises
            }
        
        # Check for strength training
        strength_exercises = exercise_logs_df.merge(
            self.exercise_data[['exercise_type', 'category']], 
            on='exercise_type', 
            how='left'
        )
        strength_sessions = len(strength_exercises[strength_exercises['category'] == 'Strength'])
        
        if strength_sessions < 2:  # Less than 2 strength sessions per week
            gaps['strength_training'] = {
                'current': strength_sessions,
                'recommended': 2,
                'deficit': 2 - strength_sessions
            }
        
        # Check for flexibility/recovery
        flexibility_exercises = len(strength_exercises[strength_exercises['category'] == 'Flexibility'])
        if flexibility_exercises < 1:
            gaps['flexibility'] = {
                'current': flexibility_exercises,
                'recommended': 2,
                'deficit': 2 - flexibility_exercises
            }
        
        return gaps
    
    def generate_exercise_recommendations(self, activity_patterns, fitness_gaps):
        """Generate personalized exercise recommendations"""
        recommendations = []
        
        # Address fitness gaps
        for gap_type, gap_info in fitness_gaps.items():
            if gap_type == 'cardio_duration':
                recommendations.append(f"Increase cardio by {gap_info['deficit']} minutes per week. Try walking, cycling, or swimming.")
            elif gap_type == 'strength_training':
                recommendations.append(f"Add {gap_info['deficit']} strength training sessions per week. Focus on major muscle groups.")
            elif gap_type == 'exercise_variety':
                recommendations.append("Try new activities to increase exercise variety and prevent boredom.")
            elif gap_type == 'flexibility':
                recommendations.append("Include yoga or stretching sessions for flexibility and recovery.")
        
        # Activity frequency recommendations
        exercises_per_week = activity_patterns.get('exercises_per_week', 0)
        if exercises_per_week < 3:
            recommendations.append("Aim for at least 3-4 exercise sessions per week for optimal health benefits.")
        
        # Duration recommendations
        avg_duration = activity_patterns.get('avg_duration_minutes', 0)
        if avg_duration < 30:
            recommendations.append("Try to exercise for at least 30 minutes per session for better results.")
        
        return recommendations
    
    def get_default_analysis(self):
        """Return default analysis when no exercise logs available"""
        return {
            'activity_patterns': {
                'exercises_per_week': 0,
                'common_exercises': {},
                'avg_duration_minutes': 0,
                'total_weekly_duration': 0,
                'intensity_distribution': {},
                'category_distribution': {}
            },
            'fitness_gaps': {
                'cardio_duration': {'current': 0, 'recommended': 150, 'deficit': 150},
                'strength_training': {'current': 0, 'recommended': 2, 'deficit': 2},
                'flexibility': {'current': 0, 'recommended': 2, 'deficit': 2}
            },
            'recommendations': [
                "Start with 150 minutes of moderate cardio per week",
                "Include 2 strength training sessions weekly",
                "Add flexibility exercises like yoga or stretching",
                "Begin with activities you enjoy to build consistency"
            ],
            'total_calories_burned': 0
        }
    
    def suggest_exercises_for_goals(self, goals, user_preferences=None):
        """Suggest exercises based on user goals and preferences"""
        suggestions = {}
        
        goal_mapping = {
            'weight_loss': ['Running', 'Cycling', 'Swimming', 'Jumping Rope'],
            'muscle_gain': ['Weight Training', 'Push-ups', 'Pull-ups', 'Resistance Training'],
            'endurance': ['Running', 'Cycling', 'Swimming', 'Rowing'],
            'flexibility': ['Yoga', 'Pilates', 'Stretching'],
            'stress_relief': ['Yoga', 'Walking', 'Swimming', 'Tai Chi'],
            'strength': ['Weight Training', 'Rock Climbing', 'Martial Arts', 'Boxing']
        }
        
        for goal in goals:
            if goal.lower() in goal_mapping:
                exercise_list = goal_mapping[goal.lower()]
                exercise_details = []
                
                for exercise in exercise_list:
                    exercise_info = self.exercise_data[self.exercise_data['exercise_type'] == exercise]
                    if not exercise_info.empty:
                        exercise_details.append({
                            'exercise': exercise,
                            'met_value': exercise_info.iloc[0]['met_value'],
                            'category': exercise_info.iloc[0]['category'],
                            'intensity': exercise_info.iloc[0]['intensity']
                        })
                
                suggestions[goal] = exercise_details
        
        return suggestions
    
    def create_weekly_plan(self, user_goals, fitness_level='beginner', available_days=3):
        """Create a weekly exercise plan based on goals and fitness level"""
        plan = {}
        
        # Adjust intensity based on fitness level
        intensity_multiplier = {
            'beginner': 0.7,
            'intermediate': 1.0,
            'advanced': 1.3
        }
        
        multiplier = intensity_multiplier.get(fitness_level, 1.0)
        
        # Base plan structure
        if available_days >= 5:
            plan_structure = ['Cardio', 'Strength', 'Cardio', 'Strength', 'Flexibility']
        elif available_days >= 3:
            plan_structure = ['Cardio', 'Strength', 'Cardio']
        else:
            plan_structure = ['Full Body Workout']
        
        for i, day_type in enumerate(plan_structure[:available_days]):
            day_key = f'Day_{i+1}'
            
            if day_type == 'Cardio':
                cardio_exercises = self.exercise_data[self.exercise_data['category'] == 'Cardio'].sample(2)
                plan[day_key] = {
                    'type': 'Cardio',
                    'exercises': [
                        {
                            'exercise': row['exercise_type'],
                            'duration': int(30 * multiplier),
                            'intensity': row['intensity']
                        } for _, row in cardio_exercises.iterrows()
                    ]
                }
            elif day_type == 'Strength':
                strength_exercises = self.exercise_data[self.exercise_data['category'] == 'Strength'].sample(1)
                plan[day_key] = {
                    'type': 'Strength',
                    'exercises': [
                        {
                            'exercise': row['exercise_type'],
                            'duration': int(45 * multiplier),
                            'intensity': row['intensity']
                        } for _, row in strength_exercises.iterrows()
                    ]
                }
            elif day_type == 'Flexibility':
                flexibility_exercises = self.exercise_data[self.exercise_data['category'] == 'Flexibility'].sample(1)
                plan[day_key] = {
                    'type': 'Flexibility',
                    'exercises': [
                        {
                            'exercise': row['exercise_type'],
                            'duration': int(30 * multiplier),
                            'intensity': row['intensity']
                        } for _, row in flexibility_exercises.iterrows()
                    ]
                }
            else:  # Full Body Workout
                mixed_exercises = self.exercise_data.sample(3)
                plan[day_key] = {
                    'type': 'Full Body',
                    'exercises': [
                        {
                            'exercise': row['exercise_type'],
                            'duration': int(20 * multiplier),
                            'intensity': row['intensity']
                        } for _, row in mixed_exercises.iterrows()
                    ]
                }
        
        return plan