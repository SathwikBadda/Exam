import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RecommendationEngine:
    def __init__(self, nutrition_analyzer, activity_tracker, clustering_engine):
        self.nutrition_analyzer = nutrition_analyzer
        self.activity_tracker = activity_tracker
        self.clustering_engine = clustering_engine
        self.cultural_preferences = self.load_cultural_preferences()
        self.dietary_restrictions = self.load_dietary_restrictions()
    
    def load_cultural_preferences(self):
        """Load cultural dietary preferences"""
        return {
            'Indian': {
                'preferred_foods': ['Rice', 'Lentils', 'Vegetables', 'Yogurt', 'Spices'],
                'cooking_methods': ['Steaming', 'Boiling', 'Sautéing'],
                'meal_patterns': ['3 main meals', 'Evening snack']
            },
            'Mediterranean': {
                'preferred_foods': ['Olive Oil', 'Fish', 'Vegetables', 'Whole Grains'],
                'cooking_methods': ['Grilling', 'Roasting', 'Raw'],
                'meal_patterns': ['3 main meals', 'Light dinner']
            },
            'Asian': {
                'preferred_foods': ['Rice', 'Fish', 'Vegetables', 'Tofu', 'Green Tea'],
                'cooking_methods': ['Stir-frying', 'Steaming', 'Boiling'],
                'meal_patterns': ['3 main meals', 'Frequent small meals']
            },
            'Western': {
                'preferred_foods': ['Meat', 'Dairy', 'Bread', 'Vegetables'],
                'cooking_methods': ['Grilling', 'Baking', 'Roasting'],
                'meal_patterns': ['3 main meals', 'Snacks']
            }
        }
    
    def load_dietary_restrictions(self):
        """Load dietary restriction guidelines"""
        return {
            'vegetarian': {
                'avoid': ['Chicken', 'Beef', 'Pork', 'Fish', 'Turkey'],
                'emphasize': ['Legumes', 'Nuts', 'Seeds', 'Dairy', 'Eggs']
            },
            'vegan': {
                'avoid': ['Chicken', 'Beef', 'Pork', 'Fish', 'Dairy', 'Eggs'],
                'emphasize': ['Legumes', 'Nuts', 'Seeds', 'Plant-based proteins']
            },
            'diabetic': {
                'avoid': ['High sugar foods', 'Refined carbs', 'Sugary drinks'],
                'emphasize': ['Complex carbs', 'Fiber-rich foods', 'Lean proteins']
            },
            'hypertension': {
                'avoid': ['High sodium foods', 'Processed foods'],
                'emphasize': ['Potassium-rich foods', 'Whole grains', 'Lean proteins']
            },
            'gluten_free': {
                'avoid': ['Wheat', 'Barley', 'Rye', 'Bread', 'Pasta'],
                'emphasize': ['Rice', 'Quinoa', 'Corn', 'Naturally gluten-free foods']
            }
        }
    
    def generate_personalized_recommendations(self, user_id, user_data, nutrition_analysis, activity_analysis):
        """Generate comprehensive personalized recommendations"""
        
        # Get user cluster for collaborative filtering
        user_cluster = self.clustering_engine.get_user_cluster(user_id)
        similar_users = self.clustering_engine.get_similar_users(user_id)
        
        # Generate nutrition recommendations
        nutrition_recommendations = self.generate_nutrition_recommendations(
            user_data, nutrition_analysis, user_cluster
        )
        
        # Generate exercise recommendations
        exercise_recommendations = self.generate_exercise_recommendations(
            user_data, activity_analysis, user_cluster
        )
        
        # Create meal plans
        meal_plans = self.create_meal_plans(user_data, nutrition_analysis)
        
        # Create exercise plans
        exercise_plans = self.create_exercise_plans(user_data, activity_analysis)
        
        # Generate lifestyle recommendations
        lifestyle_recommendations = self.generate_lifestyle_recommendations(
            user_data, nutrition_analysis, activity_analysis
        )
        
        return {
            'nutrition_recommendations': nutrition_recommendations,
            'exercise_recommendations': exercise_recommendations,
            'meal_plans': meal_plans,
            'exercise_plans': exercise_plans,
            'lifestyle_recommendations': lifestyle_recommendations,
            'similar_users_count': len(similar_users),
            'user_cluster': user_cluster
        }
    
    def generate_nutrition_recommendations(self, user_data, nutrition_analysis, user_cluster):
        """Generate personalized nutrition recommendations"""
        recommendations = []
        
        # Base recommendations from nutrition analysis
        base_recommendations = nutrition_analysis.get('recommendations', [])
        recommendations.extend(base_recommendations)
        
        # Add cultural preferences
        cultural_background = user_data.get('cultural_background', 'Western')
        if cultural_background in self.cultural_preferences:
            cultural_foods = self.cultural_preferences[cultural_background]['preferred_foods']
            recommendations.append(f"Include culturally familiar foods: {', '.join(cultural_foods[:3])}")
        
        # Add dietary restriction considerations
        dietary_preferences = user_data.get('dietary_preferences', [])
        for restriction in dietary_preferences:
            if restriction.lower() in self.dietary_restrictions:
                restriction_info = self.dietary_restrictions[restriction.lower()]
                recommendations.append(f"For {restriction} diet: emphasize {', '.join(restriction_info['emphasize'][:2])}")
        
        # Add cluster-based recommendations
        if user_cluster != -1:
            cluster_recommendations = self.clustering_engine.get_cluster_recommendations(user_cluster)
            recommendations.extend(cluster_recommendations[:2])
        
        # Add goal-specific recommendations
        health_goals = user_data.get('health_goals', [])
        for goal in health_goals:
            goal_recommendations = self.get_nutrition_recommendations_for_goal(goal)
            recommendations.extend(goal_recommendations)
        
        return list(set(recommendations))  # Remove duplicates
    
    def generate_exercise_recommendations(self, user_data, activity_analysis, user_cluster):
        """Generate personalized exercise recommendations"""
        recommendations = []
        
        # Base recommendations from activity analysis
        base_recommendations = activity_analysis.get('recommendations', [])
        recommendations.extend(base_recommendations)
        
        # Add goal-specific exercise recommendations
        health_goals = user_data.get('health_goals', [])
        for goal in health_goals:
            goal_recommendations = self.get_exercise_recommendations_for_goal(goal)
            recommendations.extend(goal_recommendations)
        
        # Add fitness level considerations
        activity_level = user_data.get('activity_level', 'Moderate')
        if activity_level == 'Low':
            recommendations.append("Start with low-impact exercises and gradually increase intensity")
        elif activity_level == 'High':
            recommendations.append("Challenge yourself with advanced training techniques")
        
        # Add age-specific recommendations
        age = user_data.get('age', 30)
        if age > 50:
            recommendations.append("Include balance and flexibility exercises for healthy aging")
        elif age < 25:
            recommendations.append("Take advantage of high recovery capacity with varied training")
        
        return list(set(recommendations))
    
    def create_meal_plans(self, user_data, nutrition_analysis):
        """Create personalized meal plans"""
        meal_plans = {}
        
        # Get nutritional gaps
        nutritional_gaps = nutrition_analysis.get('nutritional_gaps', {})
        
        # Get dietary preferences
        dietary_preferences = user_data.get('dietary_preferences', [])
        cultural_background = user_data.get('cultural_background', 'Western')
        
        # Create daily meal plan for 7 days
        for day in range(1, 8):
            daily_plan = {
                'breakfast': self.suggest_meal('breakfast', nutritional_gaps, dietary_preferences, cultural_background),
                'lunch': self.suggest_meal('lunch', nutritional_gaps, dietary_preferences, cultural_background),
                'dinner': self.suggest_meal('dinner', nutritional_gaps, dietary_preferences, cultural_background),
                'snacks': self.suggest_meal('snacks', nutritional_gaps, dietary_preferences, cultural_background)
            }
            meal_plans[f'Day_{day}'] = daily_plan
        
        return meal_plans
    
    def suggest_meal(self, meal_type, nutritional_gaps, dietary_preferences, cultural_background):
        """Suggest specific meals based on requirements"""
        meal_suggestions = {
            'breakfast': {
                'Indian': ['Oats with nuts and fruits', 'Vegetable upma', 'Whole wheat paratha with yogurt'],
                'Mediterranean': ['Greek yogurt with berries', 'Whole grain toast with avocado', 'Oatmeal with nuts'],
                'Asian': ['Congee with vegetables', 'Steamed vegetables with rice', 'Miso soup with tofu'],
                'Western': ['Oatmeal with fruits', 'Scrambled eggs with vegetables', 'Whole grain cereal']
            },
            'lunch': {
                'Indian': ['Dal with rice and vegetables', 'Quinoa pulao', 'Mixed vegetable curry with roti'],
                'Mediterranean': ['Grilled fish with vegetables', 'Quinoa salad', 'Lentil soup with bread'],
                'Asian': ['Stir-fried vegetables with brown rice', 'Miso soup with salmon', 'Tofu with steamed vegetables'],
                'Western': ['Grilled chicken salad', 'Quinoa bowl with vegetables', 'Lean meat with sweet potato']
            },
            'dinner': {
                'Indian': ['Khichdi with vegetables', 'Grilled paneer with salad', 'Vegetable soup with roti'],
                'Mediterranean': ['Grilled fish with quinoa', 'Vegetable stew', 'Lentil salad'],
                'Asian': ['Steamed fish with vegetables', 'Vegetable stir-fry', 'Miso soup with tofu'],
                'Western': ['Grilled salmon with broccoli', 'Chicken breast with quinoa', 'Vegetable soup']
            },
            'snacks': {
                'Indian': ['Mixed nuts', 'Fruit salad', 'Roasted chickpeas'],
                'Mediterranean': ['Hummus with vegetables', 'Greek yogurt', 'Mixed olives and nuts'],
                'Asian': ['Green tea with almonds', 'Edamame', 'Fresh fruit'],
                'Western': ['Apple with nut butter', 'Greek yogurt', 'Mixed berries']
            }
        }
        
        # Get base suggestions for cultural background
        base_suggestions = meal_suggestions.get(meal_type, {}).get(cultural_background, [])
        
        # Filter based on dietary preferences
        filtered_suggestions = []
        for suggestion in base_suggestions:
            is_suitable = True
            
            for preference in dietary_preferences:
                if preference.lower() in self.dietary_restrictions:
                    avoid_foods = self.dietary_restrictions[preference.lower()]['avoid']
                    for avoid_food in avoid_foods:
                        if avoid_food.lower() in suggestion.lower():
                            is_suitable = False
                            break
                if not is_suitable:
                    break
            
            if is_suitable:
                filtered_suggestions.append(suggestion)
        
        # If no suitable suggestions, provide generic healthy options
        if not filtered_suggestions:
            generic_options = {
                'breakfast': ['Oatmeal with fruits', 'Smoothie with vegetables'],
                'lunch': ['Quinoa salad', 'Vegetable soup'],
                'dinner': ['Grilled vegetables', 'Lentil curry'],
                'snacks': ['Fresh fruit', 'Mixed nuts']
            }
            filtered_suggestions = generic_options.get(meal_type, ['Healthy balanced meal'])
        
        # Add nutritional gap considerations
        if 'protein' in nutritional_gaps:
            if meal_type in ['lunch', 'dinner']:
                filtered_suggestions = [s + ' (add extra protein)' for s in filtered_suggestions]
        
        return filtered_suggestions[:3]  # Return top 3 suggestions
    
    def create_exercise_plans(self, user_data, activity_analysis):
        """Create personalized exercise plans"""
        fitness_level = user_data.get('activity_level', 'Moderate').lower()
        health_goals = user_data.get('health_goals', [])
        available_days = user_data.get('available_days', 3)
        
        # Use activity tracker to create weekly plan
        weekly_plan = self.activity_tracker.create_weekly_plan(
            health_goals, fitness_level, available_days
        )
        
        return weekly_plan
    
    def generate_lifestyle_recommendations(self, user_data, nutrition_analysis, activity_analysis):
        """Generate comprehensive lifestyle recommendations"""
        recommendations = []
        
        # Sleep recommendations
        recommendations.append("Aim for 7-9 hours of quality sleep per night")
        
        # Hydration recommendations
        weight = user_data.get('weight', 70)
        water_intake = round(weight * 0.035, 1)  # 35ml per kg body weight
        recommendations.append(f"Drink at least {water_intake} liters of water daily")
        
        # Stress management
        recommendations.append("Practice stress management techniques like meditation or deep breathing")
        
        # Meal timing
        eating_patterns = nutrition_analysis.get('eating_patterns', {})
        meals_per_day = eating_patterns.get('avg_meals_per_day', 0)
        if meals_per_day < 3:
            recommendations.append("Establish regular meal times with at least 3 meals per day")
        
        # Recovery recommendations
        exercises_per_week = activity_analysis.get('activity_patterns', {}).get('exercises_per_week', 0)
        if exercises_per_week > 5:
            recommendations.append("Include rest days for proper recovery and muscle repair")
        
        # Health monitoring
        recommendations.append("Monitor your progress weekly and adjust plans as needed")
        
        return recommendations
    
    def get_nutrition_recommendations_for_goal(self, goal):
        """Get nutrition recommendations for specific health goals"""
        goal_nutrition = {
            'weight_loss': [
                'Create a moderate caloric deficit through portion control',
                'Increase protein intake to preserve muscle mass',
                'Focus on high-fiber foods for satiety'
            ],
            'muscle_gain': [
                'Increase protein intake to 1.6-2.2g per kg body weight',
                'Ensure adequate caloric intake to support muscle growth',
                'Include post-workout protein within 30 minutes'
            ],
            'heart_health': [
                'Reduce sodium intake and increase potassium-rich foods',
                'Include omega-3 fatty acids from fish or plant sources',
                'Limit saturated fats and trans fats'
            ],
            'diabetes_management': [
                'Focus on complex carbohydrates and fiber',
                'Monitor portion sizes and meal timing',
                'Include chromium and magnesium-rich foods'
            ],
            'energy_boost': [
                'Ensure adequate iron and B-vitamin intake',
                'Include complex carbohydrates for sustained energy',
                'Stay well-hydrated throughout the day'
            ]
        }
        
        return goal_nutrition.get(goal.lower(), [])
    
    def get_exercise_recommendations_for_goal(self, goal):
        """Get exercise recommendations for specific health goals"""
        goal_exercise = {
            'weight_loss': [
                'Include both cardio and strength training',
                'Try high-intensity interval training (HIIT)',
                'Aim for 150+ minutes of moderate cardio weekly'
            ],
            'muscle_gain': [
                'Focus on progressive resistance training',
                'Include compound movements like squats and deadlifts',
                'Allow adequate rest between strength sessions'
            ],
            'heart_health': [
                'Prioritize cardiovascular exercises',
                'Include activities like swimming, cycling, or walking',
                'Monitor heart rate during exercise'
            ],
            'flexibility': [
                'Include daily stretching or yoga',
                'Focus on major muscle groups',
                'Hold stretches for 15-30 seconds'
            ],
            'stress_relief': [
                'Try mind-body exercises like yoga or tai chi',
                'Include outdoor activities when possible',
                'Focus on rhythmic, meditative movements'
            ]
        }
        
        return goal_exercise.get(goal.lower(), [])
    
    def calculate_caloric_needs(self, user_data):
        """Calculate daily caloric needs using Harris-Benedict equation"""
        age = user_data.get('age', 30)
        weight = user_data.get('weight', 70)
        height = user_data.get('height', 170)
        gender = user_data.get('gender', 'Male')
        activity_level = user_data.get('activity_level', 'Moderate')
        
        # Calculate BMR (Basal Metabolic Rate)
        if gender.lower() == 'male':
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        
        # Activity multipliers
        activity_multipliers = {
            'Low': 1.2,
            'Moderate': 1.55,
            'High': 1.725
        }
        
        multiplier = activity_multipliers.get(activity_level, 1.55)
        daily_calories = bmr * multiplier
        
        return round(daily_calories)
    
    def generate_shopping_list(self, meal_plans, days=7):
        """Generate shopping list based on meal plans"""
        shopping_list = {
            'Proteins': set(),
            'Vegetables': set(),
            'Fruits': set(),
            'Grains': set(),
            'Dairy': set(),
            'Others': set()
        }
        
        # Extract ingredients from meal plans
        for day, meals in meal_plans.items():
            if day.startswith('Day_') and int(day.split('_')[1]) <= days:
                for meal_type, suggestions in meals.items():
                    for suggestion in suggestions:
                        # Simple ingredient extraction (can be enhanced)
                        if 'chicken' in suggestion.lower():
                            shopping_list['Proteins'].add('Chicken breast')
                        if 'fish' in suggestion.lower() or 'salmon' in suggestion.lower():
                            shopping_list['Proteins'].add('Fish/Salmon')
                        if 'eggs' in suggestion.lower():
                            shopping_list['Proteins'].add('Eggs')
                        if 'yogurt' in suggestion.lower():
                            shopping_list['Dairy'].add('Greek yogurt')
                        if 'quinoa' in suggestion.lower():
                            shopping_list['Grains'].add('Quinoa')
                        if 'rice' in suggestion.lower():
                            shopping_list['Grains'].add('Brown rice')
                        if 'vegetables' in suggestion.lower():
                            shopping_list['Vegetables'].add('Mixed vegetables')
                        if 'fruits' in suggestion.lower() or 'berries' in suggestion.lower():
                            shopping_list['Fruits'].add('Fresh fruits/berries')
                        if 'nuts' in suggestion.lower():
                            shopping_list['Others'].add('Mixed nuts')
                        if 'oats' in suggestion.lower():
                            shopping_list['Grains'].add('Oats')
        
        # Convert sets to lists
        for category in shopping_list:
            shopping_list[category] = list(shopping_list[category])
        
        return shopping_list
    
    def export_recommendations_to_dict(self, recommendations):
        """Export recommendations in a structured format for saving"""
        return {
            'timestamp': datetime.now().isoformat(),
            'nutrition_recommendations': recommendations['nutrition_recommendations'],
            'exercise_recommendations': recommendations['exercise_recommendations'],
            'meal_plans': recommendations['meal_plans'],
            'exercise_plans': recommendations['exercise_plans'],
            'lifestyle_recommendations': recommendations['lifestyle_recommendations'],
            'user_cluster': recommendations['user_cluster'],
            'similar_users_count': recommendations['similar_users_count']
        }