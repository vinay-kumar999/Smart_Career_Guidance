import pickle
model.fit(X_train, y_train)
with open("career_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
