import pickle
import pandas as pd

with open("model/career_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

new_student = pd.DataFrame([{
    "CGPA": 8.5,
    "Coding_Skill": 4,
    "Math_Skill": 4,
    "Communication": 3,
    "Interest_AI": 1,
    "Interest_Web": 0,
    "Interest_Cyber": 0
}])

prediction = loaded_model.predict(new_student)
print("Predicted Career:", prediction[0])
