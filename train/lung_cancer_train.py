import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Suppress FutureWarning for replace downcasting
pd.set_option('future.no_silent_downcasting', True)

# Load dataset
df = pd.read_csv("data/lung_cancer.csv")

# Strip spaces from column names for safety
df.columns = [c.strip().replace(" ", "_").upper() for c in df.columns]

# List of categorical columns to encode
categorical_cols = [
    'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
    'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING',
    'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
]

# Replace Yes/No and Male/Female with 1/0
for col in categorical_cols:
    df[col] = df[col].replace({'Yes': 1, 'No': 0, 'MALE': 1, 'FEMALE': 0})

# Drop missing values if any
df.dropna(inplace=True)

# Features and Target
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save trained model
pickle.dump(model, open("models/lung_cancer_model.pkl", "wb"))

# Print accuracy
accuracy = model.score(X_test, y_test)
print("Lung Cancer Model Trained Successfully!")
print(f"Accuracy on test set: {accuracy*100:.2f}%")
