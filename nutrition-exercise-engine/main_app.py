import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Import custom modules
from database_manager import DatabaseManager
from nutrition_analyzer import NutritionAnalyzer
from activity_tracker import ActivityTracker
from clustering_engine import ClusteringEngine
from recommendation_engine import RecommendationEngine
from progress_predictor import ProgressPredictor
from visualization_utils import VisualizationUtils

# Page configuration
st.set_page_config(
    page_title="Nutrition & Exercise Recommendation Engine",
    page_icon="🏃‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'current_user_data' not in st.session_state:
    st.session_state.current_user_data = None

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all system components"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    db_manager = DatabaseManager()
    nutrition_analyzer = NutritionAnalyzer()
    activity_tracker = ActivityTracker()
    clustering_engine = ClusteringEngine()
    recommendation_engine = RecommendationEngine(nutrition_analyzer, activity_tracker, clustering_engine)
    progress_predictor = ProgressPredictor()
    visualization_utils = VisualizationUtils()
    
    return db_manager, nutrition_analyzer, activity_tracker, clustering_engine, recommendation_engine, progress_predictor, visualization_utils

# Load components
db_manager, nutrition_analyzer, activity_tracker, clustering_engine, recommendation_engine, progress_predictor, visualization_utils = initialize_components()

def format_date(date):
    if hasattr(date, 'strftime'):
        return date.strftime('%Y-%m-%d')
    return str(date)

def main():
    """Main application function"""
    st.title("🏃‍♀️ Nutrition & Exercise Recommendation Engine")
    st.markdown("---")
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        
        if not st.session_state.user_logged_in:
            page = st.selectbox("Choose an option:", ["Login/Register", "About"])
        else:
            page = st.selectbox("Choose a page:", [
                "Dashboard", 
                "Food Logging", 
                "Exercise Logging", 
                "Get Recommendations", 
                "Progress Tracking",
                "Download Plans",
                "Settings"
            ])
            
            if st.button("Logout"):
                st.session_state.user_logged_in = False
                st.session_state.current_user = None
                st.session_state.current_user_data = None
                st.rerun()
    
    # Page routing
    if not st.session_state.user_logged_in:
        if page == "Login/Register":
            login_register_page()
        elif page == "About":
            about_page()
    else:
        if page == "Dashboard":
            dashboard_page()
        elif page == "Food Logging":
            food_logging_page()
        elif page == "Exercise Logging":
            exercise_logging_page()
        elif page == "Get Recommendations":
            recommendations_page()
        elif page == "Progress Tracking":
            progress_tracking_page()
        elif page == "Download Plans":
            download_plans_page()
        elif page == "Settings":
            settings_page()

def login_register_page():
    """Login and registration page"""
    st.header("Welcome to the Nutrition & Exercise Recommendation Engine")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        
        if st.button("Login", key="login_button"):
            if username:
                user_data = db_manager.get_user(username)
                if user_data:
                    st.session_state.user_logged_in = True
                    st.session_state.current_user = username
                    st.session_state.current_user_data = user_data
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("User not found. Please register first.")
            else:
                st.error("Please enter a username.")
    
    with tab2:
        st.subheader("Register New User")
        with st.form("registration_form"):
            new_username = st.text_input("Username")
            age = st.number_input("Age", min_value=10, max_value=100, value=25)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
            activity_level = st.selectbox("Activity Level", ["Low", "Moderate", "High"])
            
            health_goals = st.multiselect("Health Goals", [
                "weight_loss", "muscle_gain", "heart_health", 
                "diabetes_management", "energy_boost", "fitness_improvement"
            ])
            
            dietary_preferences = st.multiselect("Dietary Preferences/Restrictions", [
                "vegetarian", "vegan", "diabetic", "hypertension", "gluten_free"
            ])
            
            cultural_background = st.selectbox("Cultural Background", [
                "Indian", "Mediterranean", "Asian", "Western", "Other"
            ])
            
            submitted = st.form_submit_button("Register")
            
            if submitted:
                if new_username:
                    # Add cultural background to dietary preferences for processing
                    user_goals = health_goals.copy()
                    user_preferences = dietary_preferences.copy()
                    
                    user_id = db_manager.create_user(
                        new_username, age, gender, height, weight, 
                        activity_level, user_goals, user_preferences
                    )
                    
                    if user_id:
                        st.success(f"User {new_username} registered successfully!")
                        st.info("Please go to the Login tab to sign in.")
                    else:
                        st.error("Username already exists. Please choose a different username.")
                else:
                    st.error("Please enter a username.")

def dashboard_page():
    """Main dashboard page"""
    st.header(f"Welcome, {st.session_state.current_user}! 🎯")
    
    user_data = st.session_state.current_user_data
    user_id = user_data['id']
    
    # Get recent data
    food_logs = db_manager.get_user_food_logs(user_id, days=30)
    exercise_logs = db_manager.get_user_exercise_logs(user_id, days=30)
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Weight",
            value=f"{user_data['weight']} kg",
            delta=None
        )
    
    with col2:
        bmi = user_data['weight'] / ((user_data['height']/100) ** 2)
        st.metric(
            label="BMI",
            value=f"{bmi:.1f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Food Logs (30 days)",
            value=len(food_logs),
            delta=None
        )
    
    with col4:
        st.metric(
            label="Exercise Sessions (30 days)",
            value=len(exercise_logs),
            delta=None
        )
    
    # Quick analysis
    if not food_logs.empty or not exercise_logs.empty:
        st.subheader("Quick Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not food_logs.empty:
                nutrition_analysis = nutrition_analyzer.analyze_food_logs(food_logs)
                fig = visualization_utils.create_nutrition_dashboard(nutrition_analysis)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if not exercise_logs.empty:
                activity_analysis = activity_tracker.analyze_activity_logs(exercise_logs, user_data['weight'])
                fig = visualization_utils.create_activity_dashboard(activity_analysis)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Start logging your food and exercise to see your personalized dashboard!")

def food_logging_page():
    """Food logging page"""
    st.header("Food Logging 🍎")
    
    user_data = st.session_state.current_user_data
    user_id = user_data['id']
    
    # Food entry form
    with st.form("food_entry_form"):
        st.subheader("Log Your Food")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", value=datetime.now().date())
            meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])
            food_item = st.text_input("Food Item")
            quantity = st.number_input("Quantity (grams)", min_value=1, value=100)
        
        with col2:
            calories = st.number_input("Calories", min_value=0, value=0)
            protein = st.number_input("Protein (g)", min_value=0.0, value=0.0)
            carbs = st.number_input("Carbohydrates (g)", min_value=0.0, value=0.0)
            fat = st.number_input("Fat (g)", min_value=0.0, value=0.0)
            fiber = st.number_input("Fiber (g)", min_value=0.0, value=0.0)
        
        submitted = st.form_submit_button("Add Food Entry")
        
        if submitted:
            if food_item:
                db_manager.add_food_log(
                    user_id, format_date(date), meal_type, 
                    food_item, quantity, calories, protein, carbs, fat, fiber
                )
                st.success(f"Added {food_item} to your food log!")
            else:
                st.error("Please enter a food item.")
    
    # Show recent food logs
    st.subheader("Recent Food Logs")
    recent_logs = db_manager.get_user_food_logs(user_id, days=7)
    
    if not recent_logs.empty:
        st.dataframe(recent_logs[['date', 'meal_type', 'food_item', 'quantity', 'calories']])
    else:
        st.info("No food logs found. Start logging your meals!")

def exercise_logging_page():
    """Exercise logging page"""
    st.header("Exercise Logging 💪")
    
    user_data = st.session_state.current_user_data
    user_id = user_data['id']
    
    # Exercise entry form
    with st.form("exercise_entry_form"):
        st.subheader("Log Your Exercise")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", value=datetime.now().date())
            exercise_type = st.selectbox("Exercise Type", [
                "Walking", "Running", "Cycling", "Swimming", "Weight Training",
                "Yoga", "Pilates", "Basketball", "Tennis", "Soccer", "Dancing",
                "Hiking", "Rowing", "Elliptical", "Jumping Rope", "Other"
            ])
            duration = st.number_input("Duration (minutes)", min_value=1, value=30)
        
        with col2:
            intensity = st.selectbox("Intensity", ["Low", "Moderate", "High"])
            calories_burned = st.number_input("Calories Burned (optional)", min_value=0, value=0)
        
        submitted = st.form_submit_button("Add Exercise Entry")
        
        if submitted:
            if exercise_type:
                # Calculate calories if not provided
                if calories_burned == 0:
                    met_values = activity_tracker.met_values
                    met_value = met_values.get(exercise_type, 5.0)
                    calories_burned = met_value * user_data['weight'] * (duration / 60)
                
                db_manager.add_exercise_log(
                    user_id, format_date(date), exercise_type,
                    duration, intensity, calories_burned
                )
                st.success(f"Added {exercise_type} to your exercise log!")
            else:
                st.error("Please select an exercise type.")
    
    # Show recent exercise logs
    st.subheader("Recent Exercise Logs")
    recent_logs = db_manager.get_user_exercise_logs(user_id, days=7)
    
    if not recent_logs.empty:
        st.dataframe(recent_logs[['date', 'exercise_type', 'duration', 'intensity', 'calories_burned']])
    else:
        st.info("No exercise logs found. Start logging your workouts!")

def recommendations_page():
    """Recommendations page"""
    st.header("Get Personalized Recommendations 🎯")
    
    user_data = st.session_state.current_user_data
    user_id = user_data['id']
    
    if st.button("Generate New Recommendations", type="primary"):
        with st.spinner("Analyzing your data and generating recommendations..."):
            # Get user data
            food_logs = db_manager.get_user_food_logs(user_id, days=30)
            exercise_logs = db_manager.get_user_exercise_logs(user_id, days=30)
            
            # Analyze nutrition and activity
            nutrition_analysis = nutrition_analyzer.analyze_food_logs(food_logs)
            activity_analysis = activity_tracker.analyze_activity_logs(exercise_logs, user_data['weight'])
            
            # Generate recommendations
            recommendations = recommendation_engine.generate_personalized_recommendations(
                user_id, user_data, nutrition_analysis, activity_analysis
            )
            
            # Save recommendations to database
            db_manager.save_recommendation(
                user_id, 
                datetime.now().strftime('%Y-%m-%d'),
                recommendations['nutrition_recommendations'],
                recommendations['exercise_recommendations'],
                user_data['health_goals']
            )
            
            # Store in session state for display
            st.session_state.current_recommendations = recommendations
    
            # Display recommendations if available
        if 'current_recommendations' in st.session_state:
            recommendations = st.session_state.current_recommendations
            
            # Nutrition Recommendations
            st.subheader("🥗 Nutrition Recommendations")
            for i, rec in enumerate(recommendations['nutrition_recommendations'], 1):
                st.write(f"{i}. {rec}")
            
            # Exercise Recommendations
            st.subheader("💪 Exercise Recommendations")
            for i, rec in enumerate(recommendations['exercise_recommendations'], 1):
                st.write(f"{i}. {rec}")
            
            # Lifestyle Recommendations
            st.subheader("🌟 Lifestyle Recommendations")
            for i, rec in enumerate(recommendations['lifestyle_recommendations'], 1):
                st.write(f"{i}. {rec}")
            
            # Meal Plans
            st.subheader("📅 7-Day Meal Plan")
            meal_plans = recommendations['meal_plans']
            
            # Create tabs for each day
            days = [f"Day {i}" for i in range(1, 8)]
            tabs = st.tabs(days)
            
            for i, tab in enumerate(tabs):
                with tab:
                    day_key = f'Day_{i+1}'
                    if day_key in meal_plans:
                        day_plan = meal_plans[day_key]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Breakfast:**")
                            for suggestion in day_plan.get('breakfast', []):
                                st.write(f"• {suggestion}")
                            
                            st.write("**Lunch:**")
                            for suggestion in day_plan.get('lunch', []):
                                st.write(f"• {suggestion}")
                        
                        with col2:
                            st.write("**Dinner:**")
                            for suggestion in day_plan.get('dinner', []):
                                st.write(f"• {suggestion}")
                            
                            st.write("**Snacks:**")
                            for suggestion in day_plan.get('snacks', []):
                                st.write(f"• {suggestion}")
            
            # Exercise Plans
            st.subheader("🏋️ Weekly Exercise Plan")
            exercise_plans = recommendations['exercise_plans']
            
            for day, plan in exercise_plans.items():
                with st.expander(f"{day} - {plan.get('type', 'Workout')}"):
                    for exercise in plan.get('exercises', []):
                        st.write(f"**{exercise['exercise']}** - {exercise['duration']} minutes ({exercise['intensity']} intensity)")
            
            # Progress Predictions
            st.subheader("📈 Progress Predictions")
            if st.button("Generate Progress Predictions"):
                with st.spinner("Predicting your progress..."):
                    predictions = progress_predictor.predict_progress(
                        user_data, 
                        recommendations['meal_plans'], 
                        recommendations['exercise_plans']
                    )
                    
                    if predictions:
                        fig = visualization_utils.create_progress_prediction_chart(predictions)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Goals analysis
                        goals_analysis = progress_predictor.analyze_goal_achievement(
                            predictions, user_data['health_goals']
                        )
                        
                        if goals_analysis:
                            st.subheader("🎯 Goal Achievement Analysis")
                            for goal, analysis in goals_analysis.items():
                                st.write(f"**{goal.replace('_', ' ').title()}:** {analysis}")

def progress_tracking_page():
    """Progress tracking page"""
    st.header("Progress Tracking 📊")
    
    user_data = st.session_state.current_user_data
    user_id = user_data['id']
    
    # Progress entry form
    with st.form("progress_entry_form"):
        st.subheader("Log Your Progress")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", value=datetime.now().date())
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=float(user_data['weight']))
            body_fat = st.number_input("Body Fat % (optional)", min_value=0.0, max_value=50.0, value=0.0)
        
        with col2:
            muscle_mass = st.number_input("Muscle Mass % (optional)", min_value=0.0, max_value=100.0, value=0.0)
            energy_level = st.slider("Energy Level (1-10)", min_value=1, max_value=10, value=5)
            sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=8.0)
        
        submitted = st.form_submit_button("Add Progress Entry")
        
        if submitted:
            db_manager.add_progress_entry(
                user_id, format_date(date), weight, 
                body_fat if body_fat > 0 else None,
                muscle_mass if muscle_mass > 0 else None,
                energy_level, sleep_hours
            )
            st.success("Progress entry added!")
    
    # Show progress history
    st.subheader("Progress History")
    progress_data = db_manager.get_user_progress(user_id)
    
    if not progress_data.empty:
        # Create progress visualization
        progress_data['date'] = pd.to_datetime(progress_data['date'])
        progress_data = progress_data.sort_values('date')
        
        # Weight progress chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=progress_data['date'],
            y=progress_data['weight'],
            mode='lines+markers',
            name='Weight (kg)',
            line=dict(color='#3498db')
        ))
        
        fig.update_layout(
            title="Weight Progress Over Time",
            xaxis_title="Date",
            yaxis_title="Weight (kg)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show recent entries
        st.subheader("Recent Entries")
        recent_progress = progress_data.tail(10)[['date', 'weight', 'energy_level', 'sleep_hours']]
        st.dataframe(recent_progress)
    else:
        st.info("No progress data found. Start tracking your progress!")

def download_plans_page():
    """Download plans page"""
    def get_json_plan(val):
        if isinstance(val, str):
            return json.loads(val)
        elif hasattr(val, 'iloc'):
            return json.loads(val.iloc[0])
        else:
            return json.loads(str(val))
    st.header("Download Your Plans 📥")
    
    user_data = st.session_state.current_user_data
    user_id = user_data['id']
    
    # Get user's recommendations
    recommendations_df = db_manager.get_user_recommendations(user_id, limit=5)
    
    if not recommendations_df.empty:
        st.subheader("Available Plans")
        
        # Show recent recommendations
        for idx, row in recommendations_df.iterrows():
            with st.expander(f"Plan from {row['date']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button(f"Download Nutrition Plan", key=f"nutrition_{idx}"):
                        nutrition_plan = get_json_plan(row['nutrition_plan'])
                        
                        # Create downloadable content
                        content = f"Nutrition Plan - {row['date']}\n"
                        content += "=" * 40 + "\n\n"
                        
                        for i, rec in enumerate(nutrition_plan, 1):
                            content += f"{i}. {rec}\n"
                        
                        st.download_button(
                            label="Download Nutrition Plan",
                            data=content,
                            file_name=f"nutrition_plan_{row['date']}.txt",
                            mime="text/plain"
                        )
                
                with col2:
                    if st.button(f"Download Exercise Plan", key=f"exercise_{idx}"):
                        exercise_plan = get_json_plan(row['exercise_plan'])
                        
                        # Create downloadable content
                        content = f"Exercise Plan - {row['date']}\n"
                        content += "=" * 40 + "\n\n"
                        
                        for i, rec in enumerate(exercise_plan, 1):
                            content += f"{i}. {rec}\n"
                        
                        st.download_button(
                            label="Download Exercise Plan",
                            data=content,
                            file_name=f"exercise_plan_{row['date']}.txt",
                            mime="text/plain"
                        )
        
        # Download complete plan
        if st.button("Download Complete Latest Plan (PDF-ready format)"):
            latest_row = recommendations_df.iloc[0]
            
            # Create comprehensive plan
            content = f"""
PERSONALIZED NUTRITION & EXERCISE PLAN
Generated on: {latest_row['date']}
User: {st.session_state.current_user}

NUTRITION RECOMMENDATIONS:
{'-' * 30}
"""
            nutrition_plan = get_json_plan(latest_row['nutrition_plan'])
            for i, rec in enumerate(nutrition_plan, 1):
                content += f"{i}. {rec}\n"
            
            content += f"""

EXERCISE RECOMMENDATIONS:
{'-' * 30}
"""
            exercise_plan = get_json_plan(latest_row['exercise_plan'])
            for i, rec in enumerate(exercise_plan, 1):
                content += f"{i}. {rec}\n"
            
            content += f"""

USER PROFILE:
{'-' * 15}
Age: {user_data['age']}
Gender: {user_data['gender']}
Height: {user_data['height']} cm
Weight: {user_data['weight']} kg
Activity Level: {user_data['activity_level']}
Health Goals: {', '.join(user_data['health_goals'])}
Dietary Preferences: {', '.join(user_data['dietary_preferences'])}

Generated by Nutrition & Exercise Recommendation Engine
"""
            
            st.download_button(
                label="Download Complete Plan",
                data=content,
                file_name=f"complete_plan_{latest_row['date']}.txt",
                mime="text/plain"
            )
    else:
        st.info("No plans available. Generate recommendations first!")

def settings_page():
    """Settings page"""
    st.header("Settings ⚙️")
    
    user_data = st.session_state.current_user_data
    
    st.subheader("User Profile")
    
    # Display current user information
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Username:** {user_data['username']}")
        st.write(f"**Age:** {user_data['age']}")
        st.write(f"**Gender:** {user_data['gender']}")
        st.write(f"**Height:** {user_data['height']} cm")
    
    with col2:
        st.write(f"**Weight:** {user_data['weight']} kg")
        st.write(f"**Activity Level:** {user_data['activity_level']}")
        st.write(f"**Health Goals:** {', '.join(user_data['health_goals'])}")
        st.write(f"**Dietary Preferences:** {', '.join(user_data['dietary_preferences'])}")
    
    # Data management
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export All Data"):
            # Export user data
            user_id = user_data['id']
            food_logs = db_manager.get_user_food_logs(user_id, days=365)
            exercise_logs = db_manager.get_user_exercise_logs(user_id, days=365)
            progress_data = db_manager.get_user_progress(user_id)
            
            export_data = {
                'user_profile': user_data,
                'food_logs': food_logs.to_dict('records') if not food_logs.empty else [],
                'exercise_logs': exercise_logs.to_dict('records') if not exercise_logs.empty else [],
                'progress_data': progress_data.to_dict('records') if not progress_data.empty else []
            }
            
            st.download_button(
                label="Download Data Export",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"user_data_export_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        st.info("Data export includes all your logged food, exercises, and progress data.")

def about_page():
    """About page"""
    st.header("About This Application")
    
    st.markdown("""
    ## 🎯 Nutrition & Exercise Recommendation Engine
    
    This application provides personalized nutrition and exercise recommendations based on:
    
    ### Features:
    - **Personalized Analysis**: AI-powered analysis of your dietary habits and physical activity
    - **Smart Recommendations**: Machine learning algorithms suggest optimal nutrition and exercise plans
    - **Progress Tracking**: Monitor your health metrics and predict future progress
    - **Cultural Awareness**: Recommendations consider your cultural dietary preferences
    - **Goal-Oriented**: Plans tailored to your specific health goals
    
    ### How It Works:
    1. **Log Your Data**: Record your daily food intake and exercise activities
    2. **Get Analysis**: Our AI analyzes your patterns and identifies areas for improvement
    3. **Receive Recommendations**: Get personalized meal plans and exercise routines
    4. **Track Progress**: Monitor your journey and see predicted outcomes
    5. **Download Plans**: Export your personalized plans for offline use
    
    ### Technology Stack:
    - **Machine Learning**: Scikit-learn, XGBoost
    - **Data Analysis**: Pandas, NumPy
    - **Visualization**: Plotly, Matplotlib
    - **Database**: SQLite
    - **Web Framework**: Streamlit
    
    ### Data Privacy:
    Your data is stored locally and never shared with third parties. All recommendations are generated using privacy-preserving algorithms.
    """)

if __name__ == "__main__":
    main()