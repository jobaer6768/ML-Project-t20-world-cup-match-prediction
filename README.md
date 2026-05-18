# T20 World Cup Forecast

This repository contains an ML notebook for T20 World Cup forecasting and a simple Streamlit app for exploring the results.

## Run locally

1. Create and activate your Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
4. Open the local URL shown in the terminal.

## App pages

- `Overview`: dataset summary and forecast preview
- `Historical Data`: browse match records and charts
- `Forecast`: tournament probability forecast
- `Match Predictor`: enter two teams, toss winner, and toss decision to predict the likely winner

## Data requirements

- `world_cup_last_30_years.csv` should be available in the project root.
- `data_processed/tournament_forecast_2026.csv` should exist after running `train.ipynb`.

If files are missing, run the notebook first to generate them.

## Deployment options

- Streamlit Cloud
- Render
- Railway
- Azure App Service
- Google Cloud Run

## Notes

The current app reads saved CSV outputs from the notebook. For a more robust deployment, the notebook logic should be refactored into Python modules and the model should be saved to disk for inference.
