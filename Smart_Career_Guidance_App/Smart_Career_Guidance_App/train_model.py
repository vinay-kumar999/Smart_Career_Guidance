import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("dataset/career_data.csv")


# Split into X and y
X = data.drop("Career", axis=1)
y = data["Career"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")

# Save model
with open("model/career_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")

