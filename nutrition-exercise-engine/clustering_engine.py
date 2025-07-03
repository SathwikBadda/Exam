import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ClusteringEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca = PCA(n_components=2)
        self.user_clusters = {}
        self.cluster_profiles = {}
    
    def prepare_user_features(self, users_data, food_logs_data, exercise_logs_data):
        """Prepare feature matrix for clustering"""
        features_list = []
        user_ids = []
        
        for user_id, user_info in users_data.items():
            # User demographic features
            age = user_info.get('age', 30)
            gender = 1 if user_info.get('gender') == 'Male' else 0
            height = user_info.get('height', 170)
            weight = user_info.get('weight', 70)
            bmi = weight / ((height/100) ** 2)
            
            # Activity level encoding
            activity_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
            activity_level = activity_mapping.get(user_info.get('activity_level', 'Moderate'), 2)
            
            # Nutrition features from food logs
            user_food_logs = food_logs_data[food_logs_data['user_id'] == user_id]
            if not user_food_logs.empty:
                avg_calories = user_food_logs['calories'].mean()
                avg_protein = user_food_logs['protein'].mean()
                avg_carbs = user_food_logs['carbs'].mean()
                avg_fat = user_food_logs['fat'].mean()
                avg_fiber = user_food_logs['fiber'].mean()
                meals_per_day = len(user_food_logs) / user_food_logs['date'].nunique() if user_food_logs['date'].nunique() > 0 else 0
            else:
                avg_calories = avg_protein = avg_carbs = avg_fat = avg_fiber = meals_per_day = 0
            
            # Exercise features from exercise logs
            user_exercise_logs = exercise_logs_data[exercise_logs_data['user_id'] == user_id]
            if not user_exercise_logs.empty:
                avg_exercise_duration = user_exercise_logs['duration'].mean()
                exercises_per_week = len(user_exercise_logs) / 4  # Assuming 4 weeks of data
                total_calories_burned = user_exercise_logs['calories_burned'].mean()
            else:
                avg_exercise_duration = exercises_per_week = total_calories_burned = 0
            
            # Combine all features
            user_features = [
                age, gender, height, weight, bmi, activity_level,
                avg_calories, avg_protein, avg_carbs, avg_fat, avg_fiber, meals_per_day,
                avg_exercise_duration, exercises_per_week, total_calories_burned
            ]
            
            features_list.append(user_features)
            user_ids.append(user_id)
        
        # Create feature dataframe
        feature_names = [
            'age', 'gender', 'height', 'weight', 'bmi', 'activity_level',
            'avg_calories', 'avg_protein', 'avg_carbs', 'avg_fat', 'avg_fiber', 'meals_per_day',
            'avg_exercise_duration', 'exercises_per_week', 'total_calories_burned'
        ]
        
        features_df = pd.DataFrame(features_list, columns=feature_names, index=user_ids)
        
        # Handle missing values
        features_df = features_df.fillna(features_df.mean())
        
        return features_df
    
    def perform_kmeans_clustering(self, features_df, n_clusters=None):
        """Perform K-means clustering on user features"""
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters(features_df)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features_df)
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(scaled_features)
        
        # Store cluster assignments
        for user_id, cluster in zip(features_df.index, cluster_labels):
            self.user_clusters[user_id] = cluster
        
        # Generate cluster profiles
        self.generate_cluster_profiles(features_df, cluster_labels)
        
        return cluster_labels
    
    def perform_dbscan_clustering(self, features_df, eps=0.5, min_samples=2):
        """Perform DBSCAN clustering for density-based grouping"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features_df)
        
        # Perform DBSCAN clustering
        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = self.dbscan_model.fit_predict(scaled_features)
        
        return cluster_labels
    
    def determine_optimal_clusters(self, features_df, max_clusters=8):
        """Determine optimal number of clusters using elbow method and silhouette score"""
        scaled_features = self.scaler.fit_transform(features_df)
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, min(max_clusters + 1, len(features_df)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            inertias.append(kmeans.inertia_)
            
            if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                silhouette_scores.append(silhouette_score(scaled_features, cluster_labels))
            else:
                silhouette_scores.append(0)
        
        # Find elbow point (simplified)
        if len(silhouette_scores) > 0:
            optimal_k = K_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 3  # Default fallback
        
        return optimal_k
    
    def generate_cluster_profiles(self, features_df, cluster_labels):
        """Generate profiles for each cluster"""
        self.cluster_profiles = {}
        
        for cluster_id in set(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = features_df[cluster_mask]
            
            profile = {
                'size': len(cluster_data),
                'avg_features': cluster_data.mean().to_dict(),
                'characteristics': self.interpret_cluster_characteristics(cluster_data.mean())
            }
            
            self.cluster_profiles[cluster_id] = profile
    
    def interpret_cluster_characteristics(self, avg_features):
        """Interpret cluster characteristics based on average features"""
        characteristics = []
        
        # BMI interpretation
        bmi = avg_features.get('bmi', 25)
        if bmi < 18.5:
            characteristics.append("Underweight")
        elif bmi < 25:
            characteristics.append("Normal weight")
        elif bmi < 30:
            characteristics.append("Overweight")
        else:
            characteristics.append("Obese")
        
        # Activity level interpretation
        activity_level = avg_features.get('activity_level', 2)
        if activity_level < 1.5:
            characteristics.append("Low activity")
        elif activity_level < 2.5:
            characteristics.append("Moderate activity")
        else:
            characteristics.append("High activity")
        
        # Caloric intake interpretation
        calories = avg_features.get('avg_calories', 2000)
        if calories < 1500:
            characteristics.append("Low caloric intake")
        elif calories < 2500:
            characteristics.append("Moderate caloric intake")
        else:
            characteristics.append("High caloric intake")
        
        # Exercise frequency interpretation
        exercises_per_week = avg_features.get('exercises_per_week', 0)
        if exercises_per_week < 2:
            characteristics.append("Infrequent exerciser")
        elif exercises_per_week < 4:
            characteristics.append("Regular exerciser")
        else:
            characteristics.append("Frequent exerciser")
        
        # Age group interpretation
        age = avg_features.get('age', 30)
        if age < 25:
            characteristics.append("Young adult")
        elif age < 40:
            characteristics.append("Adult")
        elif age < 60:
            characteristics.append("Middle-aged")
        else:
            characteristics.append("Senior")
        
        return characteristics
    
    def get_user_cluster(self, user_id):
        """Get cluster assignment for a specific user"""
        return self.user_clusters.get(user_id, -1)
    
    def get_similar_users(self, user_id, top_n=5):
        """Get similar users based on cluster membership"""
        user_cluster = self.get_user_cluster(user_id)
        if user_cluster == -1:
            return []
        
        similar_users = [uid for uid, cluster in self.user_clusters.items() 
                        if cluster == user_cluster and uid != user_id]
        
        return similar_users[:top_n]
    
    def get_cluster_recommendations(self, cluster_id):
        """Get recommendations based on cluster profile"""
        if cluster_id not in self.cluster_profiles:
            return []
        
        profile = self.cluster_profiles[cluster_id]
        characteristics = profile['characteristics']
        recommendations = []
        
        # Generate recommendations based on characteristics
        if "Low activity" in characteristics:
            recommendations.append("Start with light exercises like walking or swimming")
            recommendations.append("Gradually increase activity level over time")
        
        if "High caloric intake" in characteristics:
            recommendations.append("Focus on portion control and nutrient-dense foods")
            recommendations.append("Consider consulting a nutritionist for meal planning")
        
        if "Low caloric intake" in characteristics:
            recommendations.append("Ensure adequate nutrition with balanced meals")
            recommendations.append("Consider healthy weight gain strategies if needed")
        
        if "Infrequent exerciser" in characteristics:
            recommendations.append("Set realistic fitness goals and track progress")
            recommendations.append("Find enjoyable activities to build consistency")
        
        if "Overweight" in characteristics or "Obese" in characteristics:
            recommendations.append("Focus on sustainable weight loss through diet and exercise")
            recommendations.append("Consider high-intensity interval training (HIIT)")
        
        if "Underweight" in characteristics:
            recommendations.append("Focus on muscle-building exercises and protein intake")
            recommendations.append("Ensure adequate caloric intake for healthy weight gain")
        
        return recommendations
    
    def visualize_clusters(self, features_df, cluster_labels, save_path=None):
        """Visualize clusters using PCA"""
        # Apply PCA for 2D visualization
        scaled_features = self.scaler.fit_transform(features_df)
        pca_features = self.pca.fit_transform(scaled_features)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                            c=cluster_labels, cmap='viridis', alpha=0.7)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('User Clusters Visualization')
        plt.colorbar(scatter)
        
        if save_path:
            plt.savefig(save_path)
        
        return plt
    
    def analyze_cluster_trends(self):
        """Analyze trends across different clusters"""
        trends = {}
        
        for cluster_id, profile in self.cluster_profiles.items():
            avg_features = profile['avg_features']
            
            trends[cluster_id] = {
                'dominant_characteristics': profile['characteristics'][:3],  # Top 3 characteristics
                'avg_bmi': round(avg_features.get('bmi', 0), 2),
                'avg_calories': round(avg_features.get('avg_calories', 0), 2),
                'avg_exercise_frequency': round(avg_features.get('exercises_per_week', 0), 2),
                'cluster_size': profile['size']
            }
        
        return trends