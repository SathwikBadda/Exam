import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class NutritionAnalyzer:
    def __init__(self, nutrition_data_path="data/food_nutrition.csv"):
        self.nutrition_data_path = nutrition_data_path
        self.nutrition_data = self.load_nutrition_data()
        self.imputer = KNNImputer(n_neighbors=5)
        self.scaler = StandardScaler()
    
    def load_nutrition_data(self):
        """Load nutrition dataset"""
        try:
            df = pd.read_csv(self.nutrition_data_path)
            return df
        except FileNotFoundError:
            # Create sample nutrition data if file doesn't exist
            return self.create_sample_nutrition_data()
    
    def create_sample_nutrition_data(self):
        """Create sample nutrition data for demonstration"""
        sample_data = {
            'food_item': [
                'Apple', 'Banana', 'Chicken Breast', 'Salmon', 'Broccoli', 'Rice', 'Oats', 'Eggs',
                'Spinach', 'Sweet Potato', 'Almonds', 'Greek Yogurt', 'Quinoa', 'Avocado', 'Tomato',
                'Lentils', 'Milk', 'Cheese', 'Bread', 'Pasta', 'Olive Oil', 'Carrots', 'Beans',
                'Tuna', 'Turkey', 'Orange', 'Strawberries', 'Blueberries', 'Beef', 'Pork'
            ],
            'calories_per_100g': [
                52, 89, 165, 208, 34, 130, 389, 155, 23, 86, 579, 59, 368, 160, 18,
                116, 42, 113, 265, 131, 884, 41, 127, 144, 104, 47, 32, 57, 250, 242
            ],
            'protein_g': [
                0.3, 1.1, 31.0, 25.4, 2.8, 2.7, 16.9, 13.0, 2.9, 2.0, 21.2, 10.0, 14.1, 2.0, 0.9,
                9.0, 3.4, 25.0, 9.0, 5.0, 0.0, 0.9, 9.0, 30.0, 29.0, 0.9, 0.7, 0.7, 26.0, 26.0
            ],
            'carbs_g': [
                14.0, 23.0, 0.0, 0.0, 7.0, 28.0, 66.0, 1.1, 3.6, 20.0, 22.0, 3.6, 64.0, 9.0, 3.9,
                20.0, 5.0, 1.3, 49.0, 25.0, 0.0, 10.0, 23.0, 0.0, 0.0, 12.0, 8.0, 14.0, 0.0, 0.0
            ],
            'fat_g': [
                0.2, 0.3, 3.6, 13.4, 0.4, 0.3, 6.9, 11.0, 0.4, 0.1, 49.9, 0.4, 6.1, 15.0, 0.2,
                0.4, 1.0, 33.0, 3.2, 1.1, 100.0, 0.2, 0.5, 1.0, 1.0, 0.1, 0.3, 0.3, 15.0, 14.0
            ],
            'fiber_g': [
                2.4, 2.6, 0.0, 0.0, 2.6, 0.4, 10.6, 0.0, 2.2, 3.0, 12.5, 0.0, 7.0, 7.0, 1.2,
                7.9, 0.0, 0.0, 2.7, 1.8, 0.0, 2.8, 6.4, 0.0, 0.0, 2.4, 2.0, 2.4, 0.0, 0.0
            ],
            'category': [
                'Fruit', 'Fruit', 'Protein', 'Protein', 'Vegetable', 'Grain', 'Grain', 'Protein',
                'Vegetable', 'Vegetable', 'Nuts', 'Dairy', 'Grain', 'Fruit', 'Vegetable',
                'Legume', 'Dairy', 'Dairy', 'Grain', 'Grain', 'Fat', 'Vegetable', 'Legume',
                'Protein', 'Protein', 'Fruit', 'Fruit', 'Fruit', 'Protein', 'Protein'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv('data/food_nutrition.csv', index=False)
        return df
    
    def analyze_food_logs(self, food_logs_df):
        """Analyze user's food logs and identify nutritional patterns"""
        if food_logs_df.empty:
            return self.get_default_analysis()
        
        # Handle missing values
        food_logs_df = self.handle_missing_values(food_logs_df)
        
        # Calculate daily nutritional intake
        daily_intake = self.calculate_daily_intake(food_logs_df)
        
        # Identify nutritional gaps
        nutritional_gaps = self.identify_nutritional_gaps(daily_intake)
        
        # Analyze eating patterns
        eating_patterns = self.analyze_eating_patterns(food_logs_df)
        
        # Generate recommendations
        recommendations = self.generate_nutrition_recommendations(nutritional_gaps, eating_patterns)
        
        return {
            'daily_intake': daily_intake,
            'nutritional_gaps': nutritional_gaps,
            'eating_patterns': eating_patterns,
            'recommendations': recommendations
        }
    
    def handle_missing_values(self, df):
        """Handle missing values in food logs"""
        # Fill missing nutritional values based on similar foods
        numeric_columns = ['calories', 'protein', 'carbs', 'fat', 'fiber']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df.groupby('food_item')[col].transform('mean'))
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def calculate_daily_intake(self, food_logs_df):
        """Calculate daily nutritional intake"""
        daily_intake = food_logs_df.groupby('date').agg({
            'calories': 'sum',
            'protein': 'sum',
            'carbs': 'sum',
            'fat': 'sum',
            'fiber': 'sum'
        }).reset_index()
        
        # Calculate average daily intake
        avg_daily_intake = {
            'calories': daily_intake['calories'].mean(),
            'protein': daily_intake['protein'].mean(),
            'carbs': daily_intake['carbs'].mean(),
            'fat': daily_intake['fat'].mean(),
            'fiber': daily_intake['fiber'].mean()
        }
        
        return avg_daily_intake
    
    def identify_nutritional_gaps(self, daily_intake):
        """Identify nutritional gaps based on recommended daily values"""
        # Standard recommendations (can be personalized based on user data)
        recommended_values = {
            'calories': 2000,  # Average adult
            'protein': 50,     # grams
            'carbs': 225,      # grams
            'fat': 65,         # grams
            'fiber': 25        # grams
        }
        
        gaps = {}
        for nutrient, current_intake in daily_intake.items():
            recommended = recommended_values.get(nutrient, 0)
            if current_intake < recommended * 0.8:  # Less than 80% of recommended
                gaps[nutrient] = {
                    'current': current_intake,
                    'recommended': recommended,
                    'deficit': recommended - current_intake
                }
        
        return gaps
    
    def analyze_eating_patterns(self, food_logs_df):
        """Analyze eating patterns and habits"""
        patterns = {}
        
        # Meal frequency
        meal_frequency = food_logs_df.groupby('date')['meal_type'].count().mean()
        patterns['avg_meals_per_day'] = meal_frequency
        
        # Most common foods
        common_foods = food_logs_df['food_item'].value_counts().head(10).to_dict()
        patterns['common_foods'] = common_foods
        
        # Meal distribution
        meal_distribution = food_logs_df['meal_type'].value_counts(normalize=True).to_dict()
        patterns['meal_distribution'] = meal_distribution
        
        # Food category analysis
        if 'category' in food_logs_df.columns:
            category_distribution = food_logs_df['category'].value_counts(normalize=True).to_dict()
            patterns['category_distribution'] = category_distribution
        
        return patterns
    
    def generate_nutrition_recommendations(self, nutritional_gaps, eating_patterns):
        """Generate personalized nutrition recommendations"""
        recommendations = []
        
        # Address nutritional gaps
        for nutrient, gap_info in nutritional_gaps.items():
            if nutrient == 'protein':
                recommendations.append(f"Increase protein intake by {gap_info['deficit']:.1f}g daily. Consider adding lean meats, eggs, or legumes.")
            elif nutrient == 'fiber':
                recommendations.append(f"Add {gap_info['deficit']:.1f}g more fiber daily. Include more fruits, vegetables, and whole grains.")
            elif nutrient == 'calories':
                if gap_info['deficit'] > 200:
                    recommendations.append("Consider increasing caloric intake with nutrient-dense foods.")
                else:
                    recommendations.append("Maintain current caloric intake but focus on nutrient quality.")
        
        # Meal frequency recommendations
        if eating_patterns.get('avg_meals_per_day', 0) < 3:
            recommendations.append("Aim for at least 3 balanced meals per day for better nutrient distribution.")
        
        # Food variety recommendations
        common_foods = eating_patterns.get('common_foods', {})
        if len(common_foods) < 5:
            recommendations.append("Increase food variety to ensure diverse nutrient intake.")
        
        return recommendations
    
    def get_default_analysis(self):
        """Return default analysis when no food logs available"""
        return {
            'daily_intake': {
                'calories': 0,
                'protein': 0,
                'carbs': 0,
                'fat': 0,
                'fiber': 0
            },
            'nutritional_gaps': {
                'calories': {'current': 0, 'recommended': 2000, 'deficit': 2000},
                'protein': {'current': 0, 'recommended': 50, 'deficit': 50},
                'fiber': {'current': 0, 'recommended': 25, 'deficit': 25}
            },
            'eating_patterns': {
                'avg_meals_per_day': 0,
                'common_foods': {},
                'meal_distribution': {}
            },
            'recommendations': [
                "Start logging your food intake to get personalized recommendations",
                "Aim for balanced meals with protein, carbs, and healthy fats",
                "Include plenty of fruits and vegetables for fiber and micronutrients"
            ]
        }
    
    def suggest_foods_for_nutrients(self, target_nutrients):
        """Suggest foods to meet specific nutritional targets"""
        suggestions = {}
        
        for nutrient in target_nutrients:
            if nutrient == 'protein':
                high_protein_foods = self.nutrition_data[self.nutrition_data['protein_g'] > 15].sort_values('protein_g', ascending=False)
                suggestions[nutrient] = high_protein_foods[['food_item', 'protein_g']].head(5).to_dict('records')
            elif nutrient == 'fiber':
                high_fiber_foods = self.nutrition_data[self.nutrition_data['fiber_g'] > 5].sort_values('fiber_g', ascending=False)
                suggestions[nutrient] = high_fiber_foods[['food_item', 'fiber_g']].head(5).to_dict('records')
            elif nutrient == 'calories':
                high_calorie_foods = self.nutrition_data[self.nutrition_data['calories_per_100g'] > 200].sort_values('calories_per_100g', ascending=False)
                suggestions[nutrient] = high_calorie_foods[['food_item', 'calories_per_100g']].head(5).to_dict('records')
        
        return suggestions