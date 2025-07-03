import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class VisualizationUtils:
    def __init__(self):
        # Set style for matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color schemes
        self.colors = {
            'primary': '#3498db',
            'secondary': '#e74c3c',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'info': '#9b59b6',
            'dark': '#2c3e50'
        }
    
    def create_nutrition_dashboard(self, nutrition_analysis):
        """Create comprehensive nutrition dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Nutritional Intake', 'Nutritional Gaps', 
                          'Meal Distribution', 'Food Categories'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "pie"}]]
        )
        
        # Daily nutritional intake
        daily_intake = nutrition_analysis.get('daily_intake', {})
        nutrients = list(daily_intake.keys())
        values = list(daily_intake.values())
        
        fig.add_trace(
            go.Bar(x=nutrients, y=values, name="Current Intake", 
                   marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # Nutritional gaps
        gaps = nutrition_analysis.get('nutritional_gaps', {})
        if gaps:
            gap_nutrients = list(gaps.keys())
            gap_values = [gap['deficit'] for gap in gaps.values()]
            
            fig.add_trace(
                go.Bar(x=gap_nutrients, y=gap_values, name="Deficits", 
                       marker_color=self.colors['secondary']),
                row=1, col=2
            )
        
        # Meal distribution
        eating_patterns = nutrition_analysis.get('eating_patterns', {})
        meal_distribution = eating_patterns.get('meal_distribution', {})
        
        if meal_distribution:
            fig.add_trace(
                go.Pie(labels=list(meal_distribution.keys()), 
                       values=list(meal_distribution.values()),
                       name="Meal Distribution"),
                row=2, col=1
            )
        
        # Food categories (if available)
        category_distribution = eating_patterns.get('category_distribution', {})
        if category_distribution:
            fig.add_trace(
                go.Pie(labels=list(category_distribution.keys()), 
                       values=list(category_distribution.values()),
                       name="Food Categories"),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=True, 
                         title_text="Nutrition Analysis Dashboard")
        
        return fig
    
    def create_activity_dashboard(self, activity_analysis):
        """Create comprehensive activity dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Exercise Frequency', 'Activity Duration', 
                          'Intensity Distribution', 'Exercise Categories'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "pie"}]]
        )
        
        activity_patterns = activity_analysis.get('activity_patterns', {})
        
        # Exercise frequency (weekly)
        exercises_per_week = activity_patterns.get('exercises_per_week', 0)
        fig.add_trace(
            go.Bar(x=['Current', 'Recommended'], y=[exercises_per_week, 5], 
                   name="Exercise Frequency", 
                   marker_color=[self.colors['primary'], self.colors['success']]),
            row=1, col=1
        )
        
        # Activity duration
        avg_duration = activity_patterns.get('avg_duration_minutes', 0)
        total_duration = activity_patterns.get('total_weekly_duration', 0)
        
        fig.add_trace(
            go.Scatter(x=['Average Session', 'Weekly Total'], 
                      y=[avg_duration, total_duration],
                      mode='markers+lines', name="Duration (minutes)",
                      marker_color=self.colors['info']),
            row=1, col=2
        )
        
        # Intensity distribution
        intensity_distribution = activity_patterns.get('intensity_distribution', {})
        if intensity_distribution:
            fig.add_trace(
                go.Pie(labels=list(intensity_distribution.keys()), 
                       values=list(intensity_distribution.values()),
                       name="Intensity Distribution"),
                row=2, col=1
            )
        
        # Exercise categories
        category_distribution = activity_patterns.get('category_distribution', {})
        if category_distribution:
            fig.add_trace(
                go.Pie(labels=list(category_distribution.keys()), 
                       values=list(category_distribution.values()),
                       name="Exercise Categories"),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=True, 
                         title_text="Activity Analysis Dashboard")
        
        return fig
    
    def create_progress_prediction_chart(self, predictions):
        """Create progress prediction visualization"""
        if not predictions:
            return None
        
        weeks = list(predictions.keys())
        weights = [predictions[week]['weight'] for week in weeks]
        bmis = [predictions[week]['bmi'] for week in weeks]
        energy_levels = [predictions[week]['energy_level'] for week in weeks]
        dates = [predictions[week]['date'] for week in weeks]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Weight Prediction', 'BMI Prediction', 
                          'Energy Level Prediction', 'Weekly Weight Change'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )
        
        # Weight prediction
        fig.add_trace(
            go.Scatter(x=weeks, y=weights, mode='lines+markers',
                      name='Predicted Weight', line_color=self.colors['primary']),
            row=1, col=1
        )
        
        # BMI prediction
        fig.add_trace(
            go.Scatter(x=weeks, y=bmis, mode='lines+markers',
                      name='Predicted BMI', line_color=self.colors['secondary']),
            row=1, col=2
        )
        
        # Energy level prediction
        fig.add_trace(
            go.Scatter(x=weeks, y=energy_levels, mode='lines+markers',
                      name='Energy Level', line_color=self.colors['success']),
            row=2, col=1
        )
        
        # Weekly weight change
        weight_changes = [predictions[week]['weight_change'] for week in weeks]
        colors = [self.colors['success'] if change < 0 else self.colors['warning'] 
                 for change in weight_changes]
        
        fig.add_trace(
            go.Bar(x=weeks, y=weight_changes, name='Weight Change',
                   marker_color=colors),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, 
                         title_text="Progress Predictions (Next 4 Weeks)")
        
        return fig
    
    def create_cluster_visualization(self, cluster_data, cluster_labels):
        """Create cluster visualization"""
        if cluster_data.empty:
            return None
        
        # Create 2D visualization using first two principal components or features
        fig = px.scatter(
            x=cluster_data.iloc[:, 0], 
            y=cluster_data.iloc[:, 1],
            color=cluster_labels,
            title="User Clusters",
            labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Cluster'},
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_goals_progress_chart(self, goals_analysis):
        """Create goals progress visualization"""
        if not goals_analysis:
            return None
        
        goals = list(goals_analysis.keys())
        # Create a simple progress indicator (this could be enhanced with actual progress data)
        progress_scores = []
        colors = []
        
        for goal, status in goals_analysis.items():
            if "on track" in status.lower():
                progress_scores.append(80)
                colors.append(self.colors['success'])
            elif "moderate" in status.lower() or "slow" in status.lower():
                progress_scores.append(60)
                colors.append(self.colors['warning'])
            else:
                progress_scores.append(30)
                colors.append(self.colors['secondary'])
        
        fig = go.Figure(data=[
            go.Bar(x=goals, y=progress_scores, marker_color=colors,
                   text=[f"{score}%" for score in progress_scores],
                   textposition='auto')
        ])
        
        fig.update_layout(
            title="Goal Achievement Progress",
            xaxis_title="Goals",
            yaxis_title="Progress (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )
        
        return fig
    
    def create_meal_plan_calendar(self, meal_plans):
        """Create a visual meal plan calendar"""
        if not meal_plans:
            return None
        
        # Create a simple text-based calendar visualization
        calendar_data = []
        
        for day, meals in meal_plans.items():
            if day.startswith('Day_'):
                day_num = day.split('_')[1]
                for meal_type, suggestions in meals.items():
                    if suggestions:
                        calendar_data.append({
                            'Day': f'Day {day_num}',
                            'Meal': meal_type.capitalize(),
                            'Suggestion': suggestions[0] if suggestions else 'No suggestion'
                        })
        
        if not calendar_data:
            return None
        
        df = pd.DataFrame(calendar_data)
        
        # Create a heatmap-style visualization
        pivot_df = df.pivot(index='Meal', columns='Day', values='Suggestion')
        
        fig = px.imshow(
            [[1 if pd.notna(cell) else 0 for cell in row] for row in pivot_df.values],
            x=pivot_df.columns,
            y=pivot_df.index,
            title="Meal Plan Overview (Green = Planned)",
            color_continuous_scale='Greens'
        )
        
        fig.update_layout(height=300)
        return fig
    
    def create_nutrition_comparison_chart(self, current_intake, recommended_intake):
        """Create nutrition comparison chart"""
        nutrients = list(current_intake.keys())
        current_values = list(current_intake.values())
        recommended_values = [recommended_intake.get(nutrient, 0) for nutrient in nutrients]
        
        fig = go.Figure(data=[
            go.Bar(name='Current Intake', x=nutrients, y=current_values,
                   marker_color=self.colors['primary']),
            go.Bar(name='Recommended', x=nutrients, y=recommended_values,
                   marker_color=self.colors['success'])
        ])
        
        fig.update_layout(
            title="Current vs Recommended Nutrition Intake",
            xaxis_title="Nutrients",
            yaxis_title="Amount",
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_exercise_intensity_pie(self, intensity_distribution):
        """Create exercise intensity distribution pie chart"""
        if not intensity_distribution:
            return None
        
        fig = go.Figure(data=[go.Pie(
            labels=list(intensity_distribution.keys()),
            values=list(intensity_distribution.values()),
            hole=.3,
            marker_colors=[self.colors['success'], self.colors['warning'], self.colors['secondary']]
        )])
        
        fig.update_layout(
            title="Exercise Intensity Distribution",
            height=300
        )
        
        return fig
    
    def save_dashboard_as_html(self, figures, filename="dashboard.html"):
        """Save multiple figures as an HTML dashboard"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Nutrition & Exercise Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .dashboard-section { margin-bottom: 40px; }
                .dashboard-title { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            </style>
        </head>
        <body>
            <h1 class="dashboard-title">Nutrition & Exercise Recommendation Dashboard</h1>
        """
        
        for i, (title, fig) in enumerate(figures.items()):
            if fig is not None:
                html_content += f"""
                <div class="dashboard-section">
                    <h2>{title}</h2>
                    <div id="chart_{i}"></div>
                    <script>
                        Plotly.newPlot('chart_{i}', {fig.to_json()});
                    </script>
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename
    
    def create_summary_metrics_card(self, metrics):
        """Create summary metrics visualization"""
        if not metrics:
            return None
        
        fig = go.Figure()
        
        # Create a table-like visualization for key metrics
        fig.add_trace(go.Table(
            header=dict(values=['Metric', 'Current Value', 'Target/Recommended', 'Status'],
                       fill_color=self.colors['primary'],
                       font=dict(color='white', size=12),
                       align="left"),
            cells=dict(values=[
                list(metrics.keys()),
                [str(metrics[key].get('current', 'N/A')) for key in metrics.keys()],
                [str(metrics[key].get('target', 'N/A')) for key in metrics.keys()],
                [metrics[key].get('status', 'Unknown') for key in metrics.keys()]
            ],
            fill_color='lightgrey',
            align="left")
        ))
        
        fig.update_layout(title="Health Metrics Summary", height=300)
        return fig