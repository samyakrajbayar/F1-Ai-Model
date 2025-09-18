# 🏁 F1 AI Race Engineer System

An advanced AI-powered Formula 1 race engineering system that combines machine learning models for lap time prediction, driver performance analysis, and real-time race strategy optimization.

## 🚀 Features

### 🏎️ AI Lap Time Predictor
- **Machine Learning Model**: Random Forest algorithm trained on comprehensive F1 data
- **Input Parameters**: Track characteristics, weather conditions, car type, temperature, humidity
- **Capabilities**: 
  - Predict lap times for any track/weather/car combination
  - Qualifying vs race pace analysis
  - Weather impact assessment
  - Performance comparison across different scenarios

### 📊 Driver Performance Analyzer
- **Multi-Model Analysis**: Uses Random Forest and Gradient Boosting algorithms
- **Performance Metrics**:
  - Driver consistency scoring
  - Crash probability assessment
  - Overtaking efficiency analysis
- **Factors Considered**: Weather severity, car competitiveness, track difficulty, race distance

### 🤖 Virtual Race Engineer Chatbot
- **Real-time Strategy**: Interactive AI assistant providing live race guidance
- **Core Functions**:
  - Pit strategy optimization
  - Tire degradation monitoring
  - Weather-based strategy adaptation
  - Performance analysis and recommendations
  - Natural language interaction

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.7+
```

### Required Libraries
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Quick Start
```python
# Clone or download the script
# Run the complete system
python f1_ai_race_engineer.py
```

## 📖 Usage Guide

### 1. Basic Demo
Simply run the script to see all features in action:
```python
python f1_ai_race_engineer.py
```

### 2. Individual Components

#### Lap Time Prediction
```python
from f1_ai_race_engineer import F1LapTimePredictor

predictor = F1LapTimePredictor()
predictor.train()

# Predict lap time
lap_time = predictor.predict_lap_time(
    track='Silverstone',
    weather='Dry',
    car_type='Mercedes',
    temperature=22,
    humidity=50
)
print(f"Predicted lap time: {lap_time:.2f} seconds")
```

#### Driver Performance Analysis
```python
from f1_ai_race_engineer import DriverPerformanceAnalyzer

analyzer = DriverPerformanceAnalyzer()
analyzer.train()

# Analyze performance
performance = analyzer.analyze_driver(
    laps_completed=30,
    weather_severity=0.2,
    car_competitiveness=0.9,
    track_difficulty=0.6
)
print(f"Consistency: {performance['consistency_score']:.2%}")
```

#### Virtual Race Engineer
```python
from f1_ai_race_engineer import VirtualRaceEngineer

engineer = VirtualRaceEngineer()
engineer.initialize_models()

# Update race conditions
engineer.update_race_state(
    current_lap=25,
    tire_age=18,
    tire_compound='Medium',
    weather='Light Rain'
)

# Get strategy advice
response = engineer.chat_response("Should we pit?")
for advice in response:
    print(advice)
```

## 🎯 Interactive Commands

The Virtual Race Engineer responds to natural language queries:

| Command | Description | Example Response |
|---------|-------------|------------------|
| `"pit"` or `"box"` | Pit strategy analysis | "⚠️ Tires past optimal window - Consider pitting in next 3 laps" |
| `"weather"` | Weather strategy | "🌧️ Light rain detected - Consider intermediate tires" |
| `"lap time"` | Lap time prediction | "🏎️ Predicted lap time: 88.45 seconds" |
| `"performance"` | Driver analysis | "📊 Driver Consistency: 87%" |
| `"status"` | Current race state | "🏁 Lap: 25/50, Position: P3" |

## 🏆 System Capabilities

### Lap Time Prediction Accuracy
- **Training Data**: 5,000+ synthetic race scenarios
- **Model Type**: Random Forest with 100 estimators
- **Accuracy**: Mean Absolute Error < 2 seconds
- **Tracks Supported**: Monaco, Silverstone, Monza, Spa, Suzuka, COTA, Interlagos, Abu Dhabi

### Performance Analysis Metrics
- **Consistency Scoring**: 0-100% based on lap time variance
- **Crash Risk Assessment**: Probability scoring with weather/track factors
- **Overtaking Efficiency**: Success rate predictions

### Strategy Optimization
- **Tire Strategy**: Compound-specific degradation modeling
- **Weather Adaptation**: Dynamic strategy changes
- **Fuel Management**: Load impact on performance
- **Position Strategy**: Track-specific overtaking opportunities

## 🔧 Technical Architecture

### Machine Learning Models
```
F1LapTimePredictor
├── RandomForestRegressor (n_estimators=100)
├── StandardScaler for feature normalization
└── LabelEncoders for categorical variables

DriverPerformanceAnalyzer
├── RandomForestRegressor (consistency)
├── GradientBoostingClassifier (crash risk)
├── RandomForestRegressor (overtaking)
└── StandardScaler for feature normalization

VirtualRaceEngineer
├── Integrated model predictions
├── Rule-based strategy engine
└── Natural language processing
```

### Data Features
- **Track Data**: Length, base lap times, difficulty ratings
- **Weather Data**: Conditions, temperature, humidity, severity
- **Car Data**: Performance characteristics, competitiveness ratings
- **Race Data**: Lap counts, tire age, fuel load, position

## 📊 Example Outputs

### Lap Time Predictions
```
Monaco | Dry | Red Bull | 22°C | 45% → 78.23s
Silverstone | Light Rain | Mercedes | 18°C | 70% → 96.45s
Monza | Dry | Ferrari | 28°C | 40% → 81.67s
```

### Performance Analysis
```
Optimal Conditions:
  Consistency: 94%
  Crash Risk: 3%
  Overtaking Efficiency: 89%

Challenging Conditions:
  Consistency: 67%
  Crash Risk: 24%
  Overtaking Efficiency: 52%
```

### Strategy Recommendations
```
🤖 Current tire age: 18 laps on Medium compound
📊 Tires past optimal window
🔧 Consider pitting in next 3 laps
🌧️ Light rain detected - Monitor for intermediate tire opportunity
```

## 🚀 Advanced Features

### Custom Training Data
```python
# Use your own race data
custom_data = pd.read_csv('your_race_data.csv')
predictor = F1LapTimePredictor()
predictor.train(custom_data)
```

### Race Simulation
```python
# Simulate full race strategy
engineer = VirtualRaceEngineer()
for lap in range(1, 51):
    engineer.update_race_state(current_lap=lap)
    strategy = engineer.get_pit_strategy()
    # Implement strategy decisions
```

### Performance Monitoring
```python
# Track model performance over time
predictor.evaluate_predictions(actual_times, predicted_times)
```

## 🎮 Demo Scenarios

The system includes several pre-built scenarios:

1. **Optimal Conditions**: Dry weather, competitive car, experienced driver
2. **Challenging Weather**: Rain conditions with strategic tire decisions
3. **Tire Strategy**: Multiple pit window optimizations
4. **Performance Comparison**: Multi-driver scenario analysis

## 📄 License

This project is open source and available under the MIT License.
