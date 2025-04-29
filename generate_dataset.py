# generate_dataset.py

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Define the number of users
num_users = 1000

# Generate features
avg_reading_time = np.random.normal(loc=300, scale=50, size=num_users)  # seconds
avg_video_time = np.random.normal(loc=400, scale=60, size=num_users)    # seconds
quiz_accuracy = np.random.uniform(low=0.5, high=1.0, size=num_users)     # 0.5 to 1.0
clicks = np.random.poisson(lam=20, size=num_users)                      # clicks per session

# Generate preferred learning styles based on behavior
preferred_styles = []
for r_time, v_time, q_acc, clk in zip(avg_reading_time, avg_video_time, quiz_accuracy, clicks):
    if v_time > r_time and q_acc > 0.7:
        preferred_styles.append('Visual')
    elif r_time > v_time and clk > 15:
        preferred_styles.append('Auditory')
    else:
        preferred_styles.append('Kinesthetic')

# Create DataFrame
df = pd.DataFrame({
    'avg_reading_time': avg_reading_time,
    'avg_video_time': avg_video_time,
    'quiz_accuracy': quiz_accuracy,
    'clicks': clicks,
    'preferred_style': preferred_styles
})

# Save dataset
df.to_csv('data/user_interaction_data.csv', index=False)

print('✅ Dataset generated and saved to data/user_interaction_data.csv')