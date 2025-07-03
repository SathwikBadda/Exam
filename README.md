# Nutrition & Exercise Recommendation Engine

## Overview
An AI-powered system that provides personalized nutrition and exercise recommendations based on user data, dietary habits, and physical activity patterns.

## Features
- Personalized nutrition and exercise recommendations
- Food and exercise logging
- Progress tracking and prediction
- Downloadable plans
- Cultural and dietary preference support

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
├── setup.py                    # Setup and initialization script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Data files
│   ├── food_nutrition.csv      # Nutrition database
│   └── exercise_data.csv       # Exercise database
├── models/                     # Trained ML models (auto-generated)
├── exports/                    # User data exports
└── logs/                       # Application logs
```

## Setup Instructions

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd nutrition-exercise-engine
```

### 2. Create and Activate a Python Environment (Recommended)
```bash
conda create -n myenv python=3.9 -y
conda activate myenv
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Setup Script (to generate sample data)
```bash
python setup.py
```

### 5. Start the Streamlit App
```bash
streamlit run main_app.py
```

Or, from the parent directory:
```bash
streamlit run nutrition-exercise-engine/main_app.py
```

Then open your browser to [http://localhost:8501](http://localhost:8501)

## Usage
- Register or log in as a user
- Log your food and exercise
- Get personalized recommendations
- Track your progress
- Download your plans

## Sample Users
- **demo_user** (Male, 28, vegetarian)
- **test_user** (Female, 32, gluten-free)

## Technology Stack
- Python, Streamlit, Pandas, NumPy, Scikit-learn, XGBoost, SQLite, Plotly, Matplotlib, Seaborn

## Troubleshooting
- If you see errors about missing modules, re-run `pip install -r requirements.txt`
- If you see database errors, delete `nutrition_exercise.db` and re-run `python setup.py`
- For port conflicts, run Streamlit on a different port: `streamlit run main_app.py --server.port 8502`

## License
This project is for educational and personal use. 
