import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("data/thyroid.csv")  # path to your CSV

# Features & Target
X = df.drop("binaryClass", axis=1)
y = df["binaryClass"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save trained model
pickle.dump(model, open("models/thyroid_model.pkl", "wb"))

# Print accuracy
accuracy = model.score(X_test, y_test)
print("Thyroid Model Trained Successfully!")
print(f"Accuracy on test set: {accuracy*100:.2f}%")
