import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProgressPredictor:
    def __init__(self):
        self.weight_model = LinearRegression()
        self.bmi_model = LinearRegression()
        self.energy_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.calorie_model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_training_data(self, users_data, food_logs_data, exercise_logs_data, progress_data):
        """Prepare training data for prediction models"""
        training_data = []
        
        for user_id, user_info in users_data.items():
            # Get user's historical data
            user_food_logs = food_logs_data[food_logs_data['user_id'] == user_id]
            user_exercise_logs = exercise_logs_data[exercise_logs_data['user_id'] == user_id]
            user_progress = progress_data[progress_data['user_id'] == user_id]
            
            if len(user_progress) < 2:  # Need at least 2 data points for trends
                continue
            
            # Sort by date
            user_progress = user_progress.sort_values('date')
            
            # Create features for each progress entry (except the last one, which is target)
            for i in range(len(user_progress) - 1):
                current_date = user_progress.iloc[i]['date']
                next_date = user_progress.iloc[i + 1]['date']
                
                # Current state features
                current_weight = user_progress.iloc[i]['weight']
                current_energy = user_progress.iloc[i]['energy_level']
                
                # Calculate features from food and exercise logs between dates
                period_food_logs = user_food_logs[
                    (user_food_logs['date'] >= current_date) & 
                    (user_food_logs['date'] < next_date)
                ]
                period_exercise_logs = user_exercise_logs[
                    (user_exercise_logs['date'] >= current_date) & 
                    (user_exercise_logs['date'] < next_date)
                ]
                
                # Feature extraction
                features = self.extract_features(
                    user_info, period_food_logs, period_exercise_logs, 
                    current_weight, current_energy
                )
                
                # Target values
                next_weight = user_progress.iloc[i + 1]['weight']
                next_energy = user_progress.iloc[i + 1]['energy_level']
                
                features.update({
                    'target_weight': next_weight,
                    'target_energy': next_energy,
                    'target_bmi': next_weight / ((user_info['height']/100) ** 2)
                })
                
                training_data.append(features)
        
        return pd.DataFrame(training_data)
    
    def extract_features(self, user_info, food_logs, exercise_logs, current_weight, current_energy):
        """Extract features for prediction"""
        features = {}
        
        # User demographic features
        features['age'] = user_info.get('age', 30)
        features['gender'] = 1 if user_info.get('gender') == 'Male' else 0
        features['height'] = user_info.get('height', 170)
        features['current_weight'] = current_weight
        features['current_bmi'] = current_weight / ((user_info['height']/100) ** 2)
        features['current_energy'] = current_energy
        
        # Activity level
        activity_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
        features['activity_level'] = activity_mapping.get(user_info.get('activity_level', 'Moderate'), 2)
        
        # Nutrition features from period
        if not food_logs.empty:
            features['avg_daily_calories'] = food_logs['calories'].sum() / max(food_logs['date'].nunique(), 1)
            features['avg_daily_protein'] = food_logs['protein'].sum() / max(food_logs['date'].nunique(), 1)
            features['avg_daily_carbs'] = food_logs['carbs'].sum() / max(food_logs['date'].nunique(), 1)
            features['avg_daily_fat'] = food_logs['fat'].sum() / max(food_logs['date'].nunique(), 1)
            features['avg_daily_fiber'] = food_logs['fiber'].sum() / max(food_logs['date'].nunique(), 1)
        else:
            features.update({
                'avg_daily_calories': 0, 'avg_daily_protein': 0, 'avg_daily_carbs': 0,
                'avg_daily_fat': 0, 'avg_daily_fiber': 0
            })
        
        # Exercise features from period
        if not exercise_logs.empty:
            features['total_exercise_duration'] = exercise_logs['duration'].sum()
            features['avg_calories_burned'] = exercise_logs['calories_burned'].mean()
            features['exercise_frequency'] = len(exercise_logs)
        else:
            features.update({
                'total_exercise_duration': 0, 'avg_calories_burned': 0, 'exercise_frequency': 0
            })
        
        # Caloric balance (simplified)
        bmr = self.calculate_bmr(user_info['age'], current_weight, user_info['height'], user_info['gender'])
        daily_expenditure = bmr * activity_mapping.get(user_info.get('activity_level', 'Moderate'), 2) * 0.5
        features['caloric_balance'] = features['avg_daily_calories'] - daily_expenditure
        
        return features
    
    def calculate_bmr(self, age, weight, height, gender):
        """Calculate Basal Metabolic Rate"""
        if gender.lower() == 'male':
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        return bmr
    
    def train_models(self, training_df):
        """Train prediction models"""
        if len(training_df) < 10:  # Need sufficient data for training
            print("Insufficient data for training. Using default models.")
            self.is_trained = False
            return
        
        # Prepare features and targets
        feature_columns = [
            'age', 'gender', 'height', 'current_weight', 'current_bmi', 'current_energy',
            'activity_level', 'avg_daily_calories', 'avg_daily_protein', 'avg_daily_carbs',
            'avg_daily_fat', 'avg_daily_fiber', 'total_exercise_duration', 
            'avg_calories_burned', 'exercise_frequency', 'caloric_balance'
        ]
        
        X = training_df[feature_columns].fillna(0)
        y_weight = training_df['target_weight']
        y_bmi = training_df['target_bmi']
        y_energy = training_df['target_energy']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        try:
            # Weight prediction model
            self.weight_model.fit(X_scaled, y_weight)
            
            # BMI prediction model
            self.bmi_model.fit(X_scaled, y_bmi)
            
            # Energy level prediction model
            self.energy_model.fit(X_scaled, y_energy)
            
            self.is_trained = True
            print("Models trained successfully!")
            
        except Exception as e:
            print(f"Error training models: {e}")
            self.is_trained = False
    
    def predict_progress(self, user_data, current_nutrition_plan, current_exercise_plan, weeks_ahead=4):
        """Predict user progress for specified weeks ahead"""
        if not self.is_trained:
            return self.get_default_predictions(user_data, weeks_ahead)
        
        predictions = {}
        current_weight = user_data.get('weight', 70)
        current_energy = 5  # Default energy level (1-10 scale)
        
        for week in range(1, weeks_ahead + 1):
            # Prepare features for prediction
            features = self.prepare_prediction_features(
                user_data, current_nutrition_plan, current_exercise_plan, 
                current_weight, current_energy
            )
            
            try:
                # Scale features
                features_scaled = self.scaler.transform([features])
                
                # Make predictions
                predicted_weight = self.weight_model.predict(features_scaled)[0]
                predicted_bmi = self.bmi_model.predict(features_scaled)[0]
                predicted_energy = self.energy_model.predict(features_scaled)[0]
                
                # Ensure realistic predictions
                predicted_weight = max(30, min(200, predicted_weight))  # Reasonable weight range
                predicted_bmi = max(15, min(50, predicted_bmi))  # Reasonable BMI range
                predicted_energy = max(1, min(10, predicted_energy))  # Energy scale 1-10
                
                predictions[f'Week_{week}'] = {
                    'weight': round(predicted_weight, 1),
                    'bmi': round(predicted_bmi, 1),
                    'energy_level': round(predicted_energy, 1),
                    'weight_change': round(predicted_weight - current_weight, 1),
                    'date': (datetime.now() + timedelta(weeks=week)).strftime('%Y-%m-%d')
                }
                
                # Update current values for next iteration
                current_weight = predicted_weight
                current_energy = predicted_energy
                
            except Exception as e:
                print(f"Error predicting week {week}: {e}")
                predictions[f'Week_{week}'] = self.get_default_week_prediction(current_weight, week)
        
        return predictions
    
    def prepare_prediction_features(self, user_data, nutrition_plan, exercise_plan, current_weight, current_energy):
        """Prepare features for prediction"""
        features = []
        
        # User demographic features
        features.extend([
            user_data.get('age', 30),
            1 if user_data.get('gender') == 'Male' else 0,
            user_data.get('height', 170),
            current_weight,
            current_weight / ((user_data.get('height', 170)/100) ** 2),  # current BMI
            current_energy
        ])
        
        # Activity level
        activity_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
        features.append(activity_mapping.get(user_data.get('activity_level', 'Moderate'), 2))
        
        # Estimated nutrition features from plan (simplified)
        estimated_calories = self.estimate_calories_from_plan(nutrition_plan)
        features.extend([
            estimated_calories,  # avg_daily_calories
            estimated_calories * 0.15 / 4,  # avg_daily_protein (15% of calories)
            estimated_calories * 0.55 / 4,  # avg_daily_carbs (55% of calories)
            estimated_calories * 0.30 / 9,  # avg_daily_fat (30% of calories)
            25  # avg_daily_fiber (recommended amount)
        ])
        
        # Estimated exercise features from plan
        exercise_duration, calories_burned, frequency = self.estimate_exercise_from_plan(exercise_plan)
        features.extend([
            exercise_duration,  # total_exercise_duration
            calories_burned,   # avg_calories_burned
            frequency          # exercise_frequency
        ])
        
        # Caloric balance
        bmr = self.calculate_bmr(
            user_data.get('age', 30), current_weight, 
            user_data.get('height', 170), user_data.get('gender', 'Male')
        )
        daily_expenditure = bmr * activity_mapping.get(user_data.get('activity_level', 'Moderate'), 2) * 0.5
        caloric_balance = estimated_calories - daily_expenditure
        features.append(caloric_balance)
        
        return features
    
    def estimate_calories_from_plan(self, nutrition_plan):
        """Estimate daily calories from nutrition plan"""
        # This is a simplified estimation
        # In a real application, you'd have detailed caloric information for each meal
        base_calories = 2000  # Default
        
        # Count meals in plan
        total_meals = 0
        for day_plan in nutrition_plan.values():
            if isinstance(day_plan, dict):
                total_meals += len([meal for meal in day_plan.values() if meal])
        
        if total_meals > 0:
            avg_meals_per_day = total_meals / len(nutrition_plan)
            # Estimate calories based on meal frequency
            estimated_calories = avg_meals_per_day * 500  # Rough estimate
            return min(max(estimated_calories, 1200), 3000)  # Reasonable range
        
        return base_calories
    
    def estimate_exercise_from_plan(self, exercise_plan):
        """Estimate exercise metrics from exercise plan"""
        total_duration = 0
        total_calories = 0
        frequency = 0
        
        for day_plan in exercise_plan.values():
            if isinstance(day_plan, dict) and 'exercises' in day_plan:
                frequency += 1
                for exercise in day_plan['exercises']:
                    duration = exercise.get('duration', 30)
                    total_duration += duration
                    # Rough calorie estimation (5 calories per minute on average)
                    total_calories += duration * 5
        
        avg_calories_burned = total_calories / max(frequency, 1) if frequency > 0 else 0
        
        return total_duration, avg_calories_burned, frequency
    
    def get_default_predictions(self, user_data, weeks_ahead):
        """Generate default predictions when models aren't trained"""
        predictions = {}
        current_weight = user_data.get('weight', 70)
        
        # Simple linear prediction based on typical healthy weight loss/gain
        weekly_change = 0.2  # 0.2 kg per week (conservative estimate)
        
        for week in range(1, weeks_ahead + 1):
            predicted_weight = current_weight + (weekly_change * week)
            predicted_bmi = predicted_weight / ((user_data.get('height', 170)/100) ** 2)
            
            predictions[f'Week_{week}'] = {
                'weight': round(predicted_weight, 1),
                'bmi': round(predicted_bmi, 1),
                'energy_level': 6.0,  # Moderate energy level
                'weight_change': round(weekly_change * week, 1),
                'date': (datetime.now() + timedelta(weeks=week)).strftime('%Y-%m-%d')
            }
        
        return predictions
    
    def get_default_week_prediction(self, current_weight, week):
        """Get default prediction for a single week"""
        return {
            'weight': round(current_weight + 0.1, 1),  # Minimal change
            'bmi': round(current_weight / (1.7 ** 2), 1),  # Assume average height
            'energy_level': 6.0,
            'weight_change': 0.1,
            'date': (datetime.now() + timedelta(weeks=week)).strftime('%Y-%m-%d')
        }
    
    def analyze_goal_achievement(self, predictions, user_goals):
        """Analyze likelihood of achieving user goals"""
        analysis = {}
        
        for goal in user_goals:
            if goal.lower() == 'weight_loss':
                final_week = f'Week_{len(predictions)}'
                if final_week in predictions:
                    weight_change = predictions[final_week]['weight_change']
                    if weight_change < -1:
                        analysis[goal] = "On track - predicted weight loss"
                    elif weight_change < 0:
                        analysis[goal] = "Slow progress - minor weight loss predicted"
                    else:
                        analysis[goal] = "Not on track - no weight loss predicted"
            
            elif goal.lower() == 'muscle_gain':
                # Simplified analysis
                analysis[goal] = "Moderate progress expected with consistent training"
            
            elif goal.lower() == 'fitness_improvement':
                final_week = f'Week_{len(predictions)}'
                if final_week in predictions:
                    energy_level = predictions[final_week]['energy_level']
                    if energy_level > 7:
                        analysis[goal] = "Good progress - high energy levels predicted"
                    elif energy_level > 5:
                        analysis[goal] = "Moderate progress - stable energy levels"
                    else:
                        analysis[goal] = "Consider adjusting plan - low energy predicted"
        
        return analysis
    
    def get_model_performance(self):
        """Get information about model performance"""
        if not self.is_trained:
            return "Models not trained yet. Using default predictions."
        
        return {
            'status': 'Trained',
            'weight_model': type(self.weight_model).__name__,
            'bmi_model': type(self.bmi_model).__name__,
            'energy_model': type(self.energy_model).__name__
        }