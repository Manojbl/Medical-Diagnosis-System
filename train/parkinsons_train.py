import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("data/parkinsons.csv")  # path to your CSV

# Drop non-numeric columns
if 'name' in df.columns:
    df.drop('name', axis=1, inplace=True)

# Features and Target
X = df.drop('status', axis=1)
y = df['status']  # 0=Healthy, 1=Parkinson's

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save trained model
pickle.dump(model, open("models/parkinsons_model.pkl", "wb"))

# Print accuracy
accuracy = model.score(X_test, y_test)
print("Parkinson's Disease Model Trained Successfully!")
print(f"Accuracy on test set: {accuracy*100:.2f}%")
