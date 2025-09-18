import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class F1LapTimePredictor:
    """AI model to predict lap times based on track, weather, and car characteristics"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.track_encoder = LabelEncoder()
        self.weather_encoder = LabelEncoder()
        self.car_encoder = LabelEncoder()
        self.is_trained = False
        
    def generate_training_data(self, n_samples=5000):
        """Generate synthetic F1 training data"""
        np.random.seed(42)
        
        # Track data (simplified)
        tracks = ['Monaco', 'Silverstone', 'Monza', 'Spa', 'Suzuka', 'COTA', 'Interlagos', 'Abu Dhabi']
        track_base_times = [78.0, 88.0, 81.0, 106.0, 91.0, 96.0, 71.0, 97.0]  # Base lap times in seconds
        track_lengths = [3.337, 5.891, 5.793, 7.004, 5.807, 5.513, 4.309, 5.554]  # km
        
        # Weather conditions
        weather_conditions = ['Dry', 'Light Rain', 'Heavy Rain', 'Wet']
        weather_impact = [0, 8, 15, 12]  # seconds added to lap time
        
        # Car types (teams)
        car_types = ['Red Bull', 'Mercedes', 'Ferrari', 'McLaren', 'Alpine', 'Aston Martin', 'Williams', 'AlphaTauri', 'Alfa Romeo', 'Haas']
        car_performance = [0, 0.3, 0.2, 0.8, 1.2, 0.9, 2.1, 1.5, 1.8, 2.3]  # seconds slower than fastest
        
        data = []
        for _ in range(n_samples):
            track_idx = np.random.randint(0, len(tracks))
            weather_idx = np.random.randint(0, len(weather_conditions))
            car_idx = np.random.randint(0, len(car_types))
            
            # Base lap time calculation
            base_time = track_base_times[track_idx]
            weather_penalty = weather_impact[weather_idx]
            car_penalty = car_performance[car_idx]
            
            # Add some randomness for driver skill, tire wear, fuel load
            random_factor = np.random.normal(0, 1.5)
            
            lap_time = base_time + weather_penalty + car_penalty + random_factor
            
            data.append({
                'track': tracks[track_idx],
                'track_length': track_lengths[track_idx],
                'weather': weather_conditions[weather_idx],
                'car_type': car_types[car_idx],
                'temperature': np.random.uniform(15, 35),
                'humidity': np.random.uniform(30, 90),
                'lap_time': lap_time
            })
            
        return pd.DataFrame(data)
    
    def train(self, data=None):
        """Train the lap time prediction model"""
        if data is None:
            data = self.generate_training_data()
            
        # Encode categorical variables
        data['track_encoded'] = self.track_encoder.fit_transform(data['track'])
        data['weather_encoded'] = self.weather_encoder.fit_transform(data['weather'])
        data['car_encoded'] = self.car_encoder.fit_transform(data['car_type'])
        
        # Prepare features
        features = ['track_encoded', 'track_length', 'weather_encoded', 'car_encoded', 'temperature', 'humidity']
        X = data[features]
        y = data['lap_time']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate accuracy
        y_pred = self.model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        print(f"Lap Time Predictor trained successfully! MAE: {mae:.2f} seconds")
        
    def predict_lap_time(self, track, weather, car_type, temperature=25, humidity=50):
        """Predict lap time for given conditions"""
        if not self.is_trained:
            print("Model not trained yet. Training now...")
            self.train()
            
        # Get track length (simplified mapping)
        track_lengths = {
            'Monaco': 3.337, 'Silverstone': 5.891, 'Monza': 5.793, 'Spa': 7.004,
            'Suzuka': 5.807, 'COTA': 5.513, 'Interlagos': 4.309, 'Abu Dhabi': 5.554
        }
        track_length = track_lengths.get(track, 5.0)
        
        try:
            # Encode inputs
            track_encoded = self.track_encoder.transform([track])[0]
            weather_encoded = self.weather_encoder.transform([weather])[0]
            car_encoded = self.car_encoder.transform([car_type])[0]
            
            # Prepare features
            features = np.array([[track_encoded, track_length, weather_encoded, car_encoded, temperature, humidity]])
            features_scaled = self.scaler.transform(features)
            
            # Predict
            predicted_time = self.model.predict(features_scaled)[0]
            return predicted_time
            
        except ValueError as e:
            return f"Error: Unknown track, weather condition, or car type. {e}"


class DriverPerformanceAnalyzer:
    """AI model to analyze and compare driver performance"""
    
    def __init__(self):
        self.consistency_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.crash_model = GradientBoostingClassifier(random_state=42)
        self.overtaking_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_driver_data(self, n_samples=2000):
        """Generate synthetic driver performance data"""
        np.random.seed(42)
        
        drivers = ['Verstappen', 'Hamilton', 'Russell', 'Leclerc', 'Sainz', 'Norris', 'Piastri', 'Alonso', 'Stroll', 'Ocon']
        
        # Driver skill levels (affects all metrics)
        driver_skills = [0.95, 0.92, 0.88, 0.90, 0.87, 0.85, 0.83, 0.89, 0.82, 0.84]
        
        data = []
        for _ in range(n_samples):
            driver_idx = np.random.randint(0, len(drivers))
            skill = driver_skills[driver_idx]
            
            # Race conditions
            laps_completed = np.random.randint(50, 70)
            weather_severity = np.random.uniform(0, 1)  # 0 = dry, 1 = very wet
            car_competitiveness = np.random.uniform(0.7, 1.0)
            
            # Performance metrics
            consistency = skill * (1 - weather_severity * 0.3) + np.random.normal(0, 0.05)
            crash_probability = (1 - skill) * weather_severity * 0.8 + np.random.uniform(0, 0.1)
            overtaking_efficiency = skill * car_competitiveness + np.random.normal(0, 0.1)
            
            data.append({
                'driver': drivers[driver_idx],
                'laps_completed': laps_completed,
                'weather_severity': weather_severity,
                'car_competitiveness': car_competitiveness,
                'track_difficulty': np.random.uniform(0, 1),
                'consistency_score': max(0, min(1, consistency)),
                'crash_risk': max(0, min(1, crash_probability)),
                'overtaking_efficiency': max(0, min(1, overtaking_efficiency))
            })
            
        return pd.DataFrame(data)
    
    def train(self, data=None):
        """Train driver performance models"""
        if data is None:
            data = self.generate_driver_data()
            
        features = ['laps_completed', 'weather_severity', 'car_competitiveness', 'track_difficulty']
        X = data[features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Train consistency model
        self.consistency_model.fit(X_scaled, data['consistency_score'])
        
        # Train crash risk model (classification)
        crash_binary = (data['crash_risk'] > 0.3).astype(int)  # High risk threshold
        self.crash_model.fit(X_scaled, crash_binary)
        
        # Train overtaking model
        self.overtaking_model.fit(X_scaled, data['overtaking_efficiency'])
        
        self.is_trained = True
        print("Driver Performance Analyzer trained successfully!")
        
    def analyze_driver(self, laps_completed, weather_severity, car_competitiveness, track_difficulty):
        """Analyze driver performance for given conditions"""
        if not self.is_trained:
            print("Model not trained yet. Training now...")
            self.train()
            
        features = np.array([[laps_completed, weather_severity, car_competitiveness, track_difficulty]])
        features_scaled = self.scaler.transform(features)
        
        consistency = self.consistency_model.predict(features_scaled)[0]
        crash_risk_prob = self.crash_model.predict_proba(features_scaled)[0][1]  # Probability of high risk
        overtaking_eff = self.overtaking_model.predict(features_scaled)[0]
        
        return {
            'consistency_score': consistency,
            'crash_risk_probability': crash_risk_prob,
            'overtaking_efficiency': overtaking_eff
        }


class VirtualRaceEngineer:
    """AI Race Engineer Chatbot with strategic recommendations"""
    
    def __init__(self):
        self.lap_predictor = F1LapTimePredictor()
        self.driver_analyzer = DriverPerformanceAnalyzer()
        self.race_state = {
            'current_lap': 1,
            'total_laps': 50,
            'tire_age': 0,
            'tire_compound': 'Medium',
            'fuel_load': 100,
            'position': 5,
            'weather': 'Dry',
            'track': 'Silverstone',
            'car_type': 'McLaren',
            'temperature': 25
        }
        
    def initialize_models(self):
        """Initialize and train all AI models"""
        print("Initializing Virtual Race Engineer...")
        print("Training Lap Time Predictor...")
        self.lap_predictor.train()
        print("Training Driver Performance Analyzer...")
        self.driver_analyzer.train()
        print("Virtual Race Engineer ready! 🏁")
        
    def update_race_state(self, **kwargs):
        """Update current race conditions"""
        self.race_state.update(kwargs)
        
    def get_pit_strategy(self):
        """Analyze current conditions and suggest pit strategy"""
        tire_age = self.race_state['tire_age']
        tire_compound = self.race_state['tire_compound']
        current_lap = self.race_state['current_lap']
        total_laps = self.race_state['total_laps']
        
        # Tire degradation model (simplified)
        if tire_compound == 'Soft':
            optimal_stint = 15
            critical_age = 20
        elif tire_compound == 'Medium':
            optimal_stint = 25
            critical_age = 30
        else:  # Hard
            optimal_stint = 35
            critical_age = 40
            
        laps_remaining = total_laps - current_lap
        
        recommendations = []
        
        if tire_age > critical_age:
            recommendations.append("⚠️ CRITICAL: Tires are heavily degraded! Box this lap!")
            recommendations.append("🔧 Pit window: IMMEDIATE")
        elif tire_age > optimal_stint:
            recommendations.append("📊 Tires past optimal window")
            recommendations.append(f"🔧 Consider pitting in next {3} laps")
        elif laps_remaining < 15 and tire_age > optimal_stint * 0.7:
            recommendations.append("🏁 Final stint strategy: These tires should last to the end")
        else:
            recommendations.append("✅ Tires in good condition, continue current stint")
            
        return recommendations
    
    def get_weather_strategy(self):
        """Analyze weather and provide strategic advice"""
        weather = self.race_state['weather']
        
        if weather == 'Light Rain':
            return [
                "🌧️ Light rain detected",
                "💡 Consider intermediate tires if rain intensifies",
                "⚠️ Reduce pace by 2-3 seconds for safety"
            ]
        elif weather == 'Heavy Rain':
            return [
                "🌧️ Heavy rain conditions",
                "🔧 Full wet tires recommended",
                "⚠️ Extreme caution required, reduce pace significantly"
            ]
        elif weather == 'Wet':
            return [
                "💧 Wet track conditions",
                "🔧 Intermediate tires optimal",
                "⚠️ Watch for dry line development"
            ]
        else:
            return ["☀️ Dry conditions, optimal for performance"]
    
    def predict_lap_time(self):
        """Get lap time prediction for current conditions"""
        predicted_time = self.lap_predictor.predict_lap_time(
            track=self.race_state['track'],
            weather=self.race_state['weather'],
            car_type=self.race_state['car_type'],
            temperature=self.race_state['temperature']
        )
        
        if isinstance(predicted_time, (int, float)):
            return f"🏎️ Predicted lap time: {predicted_time:.2f} seconds"
        else:
            return f"⚠️ {predicted_time}"
    
    def analyze_performance(self):
        """Analyze current driver performance"""
        analysis = self.driver_analyzer.analyze_driver(
            laps_completed=self.race_state['current_lap'],
            weather_severity=0.0 if self.race_state['weather'] == 'Dry' else 0.7,
            car_competitiveness=0.85,  # McLaren competitiveness
            track_difficulty=0.6
        )
        
        return [
            f"📊 Driver Consistency: {analysis['consistency_score']:.2%}",
            f"⚠️ Crash Risk: {analysis['crash_risk_probability']:.2%}",
            f"🏃 Overtaking Efficiency: {analysis['overtaking_efficiency']:.2%}"
        ]
    
    def chat_response(self, user_input):
        """Main chatbot interface"""
        user_input = user_input.lower()
        
        if 'pit' in user_input or 'box' in user_input:
            return self.get_pit_strategy()
        elif 'weather' in user_input or 'rain' in user_input:
            return self.get_weather_strategy()
        elif 'lap time' in user_input or 'predict' in user_input:
            return [self.predict_lap_time()]
        elif 'performance' in user_input or 'analyze' in user_input:
            return self.analyze_performance()
        elif 'status' in user_input or 'update' in user_input:
            return [
                f"🏁 Current Status:",
                f"Lap: {self.race_state['current_lap']}/{self.race_state['total_laps']}",
                f"Position: P{self.race_state['position']}",
                f"Tires: {self.race_state['tire_compound']} ({self.race_state['tire_age']} laps)",
                f"Weather: {self.race_state['weather']}",
                f"Track: {self.race_state['track']}"
            ]
        else:
            return [
                "🤖 Virtual Race Engineer Commands:",
                "'pit' or 'box' - Get pit strategy advice",
                "'weather' - Get weather-related strategy",
                "'lap time' or 'predict' - Get lap time prediction",
                "'performance' - Analyze driver performance",
                "'status' - Get current race status",
                "Ask me anything about race strategy! 🏁"
            ]


# Demo and Testing Functions
def run_demo():
    """Run a comprehensive demo of all features"""
    print("="*60)
    print("🏁 F1 AI RACE ENGINEER SYSTEM DEMO")
    print("="*60)
    
    # Initialize the Virtual Race Engineer
    engineer = VirtualRaceEngineer()
    engineer.initialize_models()
    
    print("\n" + "="*50)
    print("🚗 LAP TIME PREDICTION DEMO")
    print("="*50)
    
    # Test lap time predictions
    test_conditions = [
        ('Monaco', 'Dry', 'Red Bull', 22, 45),
        ('Silverstone', 'Light Rain', 'Mercedes', 18, 70),
        ('Monza', 'Dry', 'Ferrari', 28, 40),
        ('Spa', 'Heavy Rain', 'McLaren', 15, 85)
    ]
    
    for track, weather, car, temp, humidity in test_conditions:
        lap_time = engineer.lap_predictor.predict_lap_time(track, weather, car, temp, humidity)
        print(f"{track} | {weather} | {car} | {temp}°C | {humidity}% → {lap_time:.2f}s")
    
    print("\n" + "="*50)
    print("📊 DRIVER PERFORMANCE ANALYSIS DEMO")
    print("="*50)
    
    # Test driver performance analysis
    test_scenarios = [
        (30, 0.0, 0.95, 0.3),  # Dry, good car, easy track
        (45, 0.8, 0.75, 0.8),  # Wet, average car, difficult track
        (60, 0.2, 0.85, 0.5)   # Light wet, good car, medium track
    ]
    
    scenario_names = ["Optimal Conditions", "Challenging Conditions", "Mixed Conditions"]
    
    for i, (laps, weather, car_comp, track_diff) in enumerate(test_scenarios):
        analysis = engineer.driver_analyzer.analyze_driver(laps, weather, car_comp, track_diff)
        print(f"\n{scenario_names[i]}:")
        print(f"  Consistency: {analysis['consistency_score']:.2%}")
        print(f"  Crash Risk: {analysis['crash_risk_probability']:.2%}")
        print(f"  Overtaking Efficiency: {analysis['overtaking_efficiency']:.2%}")
    
    print("\n" + "="*50)
    print("🤖 VIRTUAL RACE ENGINEER DEMO")
    print("="*50)
    
    # Simulate a race scenario
    engineer.update_race_state(
        current_lap=25,
        tire_age=18,
        tire_compound='Medium',
        weather='Light Rain',
        track='Silverstone',
        position=3
    )
    
    # Demo different queries
    queries = [
        "What's our current status?",
        "Should we pit?",
        "What about the weather?",
        "Predict our lap time",
        "How's our performance?"
    ]
    
    for query in queries:
        print(f"\n👤 Engineer: {query}")
        responses = engineer.chat_response(query)
        for response in responses:
            print(f"🤖 {response}")
    
    print("\n" + "="*50)
    print("🏁 INTERACTIVE MODE")
    print("="*50)
    print("Ask the Virtual Race Engineer anything!")
    print("Type 'quit' to exit")
    
    while True:
        user_query = input("\n👤 You: ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("🏁 Thanks for using the F1 AI Race Engineer! See you at the next race!")
            break
        
        responses = engineer.chat_response(user_query)
        for response in responses:
            print(f"🤖 {response}")


# Feature showcase functions
def showcase_lap_time_predictor():
    """Detailed showcase of lap time prediction capabilities"""
    predictor = F1LapTimePredictor()
    predictor.train()
    
    print("🏎️ LAP TIME PREDICTOR SHOWCASE")
    print("-" * 40)
    
    # Create comparison table
    tracks = ['Monaco', 'Silverstone', 'Monza', 'Spa']
    weathers = ['Dry', 'Light Rain', 'Heavy Rain']
    cars = ['Red Bull', 'Mercedes', 'Ferrari', 'McLaren']
    
    results = []
    for track in tracks:
        for weather in weathers:
            for car in cars:
                lap_time = predictor.predict_lap_time(track, weather, car)
                if isinstance(lap_time, (int, float)):
                    results.append({'Track': track, 'Weather': weather, 'Car': car, 'Lap Time': lap_time})
    
    df = pd.DataFrame(results)
    print("\nSample Predictions:")
    print(df.head(10).to_string(index=False))
    
    # Weather impact analysis
    print("\n🌧️ Weather Impact Analysis:")
    weather_impact = df.groupby('Weather')['Lap Time'].mean().sort_values()
    for weather, avg_time in weather_impact.items():
        print(f"{weather}: {avg_time:.2f}s average")


if __name__ == "__main__":
    # Run the complete demo
    run_demo()
    
    print("\n" + "="*60)
    print("🏁 ADDITIONAL FEATURES AVAILABLE:")
    print("="*60)
    print("• showcase_lap_time_predictor() - Detailed lap time analysis")
    print("• Individual model testing and customization")
    print("• Race strategy optimization")
    print("• Historical performance tracking")
    print("• Weather impact analysis")
    print("• Multi-driver comparison tools")
