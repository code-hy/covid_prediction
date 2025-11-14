import pandas as pd

# Example: Create lag features (last 7 days) for each country
def create_features(df, lag=7):
    df = df.set_index('date')
    for i in range(1, lag+1):
        df[f'lag_{i}'] = df['new_cases'].shift(i)
    return df.dropna().reset_index()

def main():
    # Load dataset used for training
    df = pd.read_csv("cleaned_covid_data.csv")
    df['date'] = pd.to_datetime(df['date'])

    # Group by country and apply
    models = {}
    from xgboost import XGBRegressor

    for country in df['location'].unique():
        country_df = df[df['location'] == country].copy()
        if len(country_df) < 30:
            continue  # skip countries with little data

        feat_df = create_features(country_df)
        X = feat_df[[f'lag_{i}' for i in range(1,8)]]
        y = feat_df['new_cases']

        model = XGBRegressor(n_estimators=100)
        model.fit(X, y)
        models[country] = model

    # Save model (e.g., using joblib)
    import joblib
    joblib.dump(models, 'covid_forecast_models.pkl')


if __name__ == "__main__":
    main()