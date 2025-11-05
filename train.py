import sys
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def load_and_preprocess(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date')
    df = df[['Date','Open','High','Low','Close','Volume']].dropna()
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    df['MA_5'] = df['Close'].rolling(window=5).mean().fillna(method='bfill')
    df['MA_10'] = df['Close'].rolling(window=10).mean().fillna(method='bfill')
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    return df


if __name__ == '__main__':
    csv_in = sys.argv[1] if len(sys.argv) > 1 else 'Tesla.csv'
    out_model = sys.argv[2] if len(sys.argv) > 2 else 'model.joblib'
    df = load_and_preprocess(csv_in)
    features = ['Open','High','Low','Close','Volume','Daily_Return','MA_5','MA_10']
    X = df[features].values
    y = df['Target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    joblib.dump({'model': model, 'features': features}, out_model)
    metrics = {'rmse': rmse, 'r2': r2, 'n_train': len(X_train), 'n_test': len(X_test)}
    with open('metrics.json','w') as f:
        json.dump(metrics, f)
    print('RMSE:', rmse)
    print('R2:', r2)
    print('Saved model to', out_model)
    print('Saved metrics to metrics.json')
