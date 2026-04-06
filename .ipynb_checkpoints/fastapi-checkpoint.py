from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

classifier = joblib.load('Models/placement_classifier.pkl')
regressor = joblib.load('Models/salary_regressor.pkl')
placement_encoder = joblib.load('Models/placement_encoder.pkl')
salary_encoder = joblib.load('Models/salary_encoder.pkl')

nominal_cols = [
    'college_tier', 'country', 'university_ranking_band',
    'specialization', 'industry'
]


class StudentInput(BaseModel):
    cgpa: float
    backlogs: int
    college_tier: str
    country: str
    university_ranking_band: str
    internship_count: int
    aptitude_score: float
    communication_score: float
    specialization: str
    industry: str
    internship_quality_score: float

def preprocess(data: StudentInput, encoder):
    df = pd.DataFrame([data.dict()])
    
    # Encode nominal columns
    encoded = encoder.transform(df[nominal_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(nominal_cols)
    )
    
    # Drop original nominal cols and concat encoded
    df = df.drop(columns=nominal_cols)
    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
    return df

@app.post('/predict-placement')
def predict_placement(data: StudentInput):
    df = preprocess(data, placement_encoder)
    proba = classifier.predict_proba(df)[:, 1][0]
    prediction = int(proba >= 0.4)
    return {
        'placement_probability': round(proba, 3),
        'placement_status': 'Placed' if prediction == 1 else 'Not Placed'
    }


@app.post('/predict-salary')
def predict_salary(data: StudentInput):
    # Only run if placed
    placement_df = preprocess(data, placement_encoder)
    proba = classifier.predict_proba(placement_df)[:, 1][0]
    
    if proba < 0.4:
        return {'message': 'Student is predicted as Not Placed — no salary estimate'}
    
    salary_df = preprocess(data, salary_encoder)
    salary = regressor.predict(salary_df)[0]
    return {'predicted_salary': round(salary, 2)}



