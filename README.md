
<img width="1240" height="867" alt="image" src="https://github.com/user-attachments/assets/6a6528e7-b471-4ff2-9107-9feee5d19d41" />







# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved or rejected.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- FastAPI

## Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest

## Features
- Data preprocessing
- Model comparison and hyperparameter tuning
- Model evaluation
- Loan prediction using FastAPI

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Start the API:

```bash
python -m uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs to test the API.

## Project Structure

```text
loan-approval-prediction/
├── data/
├── app.py
├── train.py
├── model.pkl
├── notebook.ipynb
└── requirements.txt
```

## Evaluation

The model achieved approximately 86% accuracy on the test dataset.

## Author

Rahul R
