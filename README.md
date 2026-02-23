# Student Performance Analytics

This project is about analyzing student scores (math, reading, writing) and finding patterns that put students "at risk." We used a Logistic Regression model to help classify students and give them study tips.

## Files
- `data/`: Student performance data.
- `notebooks/`: EDA and early testing.
- `src/`: Core logic for cleaning and models.
- `models/`: Saved pkl files (model and scaler).
- `app.py`: The dashboard for the project.
- `main.tex`: The research report.

## How to setup
1. Install stuff: `pip install -r requirements.txt`
2. Run app: `streamlit run app.py`
3. If you want to retrain: `python3 src/models.py`

## Main Features
- Cleans and encodes demographic data automatically.
- Predicts risk levels using a trained ML model.
- Gives personalized study advice based on specific scores.
