
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("model.pkl")


class LoanApplicant(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@app.get("/")
def home():
    return {"message": "Loan Approval Prediction API"}


@app.post("/predict")
def predict_loan(applicant: LoanApplicant):
    data = pd.DataFrame([applicant.model_dump()])
    prediction = model.predict(data)[0]

    status = "Approved" if prediction == 1 else "Rejected"

    return {"loan_status": status}
