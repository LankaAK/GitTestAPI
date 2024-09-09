import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. Initialize FastAPI app
app = FastAPI()

# 2. Load the trained model
model_path = os.getenv('MODEL_PATH', 'titanic_model.pkl')  # Read model path from environment variable
model = joblib.load(model_path)

# 3. Define the input data model
class PassengerInput(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_Q: int
    Embarked_S: int
    FamilySize: int

# 4. Define the prediction endpoint
@app.post("/predict/")
def predict_survival(input_data: PassengerInput):
    # Convert input data to DataFrame for model
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Return prediction result
    return {"Survived": int(prediction[0])}

# 5. Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
