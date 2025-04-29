# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
data = pd.read_csv('data/user_interaction_data.csv')

# Features and target
X = data[['avg_reading_time', 'avg_video_time', 'quiz_accuracy', 'clicks']]
y = data['preferred_style']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'models/learning_style_model.pkl')

print('✅ Model trained and saved to models/learning_style_model.pkl')
