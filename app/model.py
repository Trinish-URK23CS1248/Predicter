from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from .utils import load_and_preprocess, find_dataset_path


MODEL_PATH = Path(__file__).resolve().parent.parent / 'model.joblib'


def train(csv_path=None):
    csv_path = csv_path or find_dataset_path()
    df = load_and_preprocess(csv_path)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'MA_5', 'MA_10']
    X = df[features].values
    y = df['Target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    joblib.dump({'model': model, 'features': features}, MODEL_PATH)
    return {'rmse': rmse, 'r2': r2, 'model_path': str(MODEL_PATH)}


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_next_from_last():
    # Train if model missing
    if not MODEL_PATH.exists():
        train()
    data = load_model()
    model = data['model']
    features = data['features']
    # load latest row
    df = load_and_preprocess()
    last = df.iloc[-1]
    x = [float(last[f]) for f in features]
    pred = model.predict([x])[0]
    return {
        'date': str(last['Date'].date()),
        'current_close': float(last['Close']),
        'predicted_next_close': float(pred),
        'predicted_return': float((pred - last['Close']) / last['Close'])
    }
