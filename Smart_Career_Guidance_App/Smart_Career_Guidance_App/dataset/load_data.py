import pandas as pd
data = pd.read_csv("dataset/career_data.csv")
print(data.head())

print("\nColumn names:")
print(data.columns)

print("\nDataset info:")
print(data.info())

#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#separating input(x) and output(y)
X = data.drop("Career", axis=1)
y = data["Career"]

print("X shape:", X.shape)
print("y shape:", y.shape)

#split the data train and test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

#tain the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Model trained successfully!")

#checking accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#test the data
#new_student = [[8.5, 4, 4, 3, 1, 1, 0]]
new_student=[[9.5,2,1,3,0,0,0]]
prediction = model.predict(new_student)
print("Predicted Career:", prediction[0])
