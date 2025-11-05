from pathlib import Path
import pandas as pd


def find_dataset_path(filename="Tesla.csv"):
    # Walk up from this file to find the CSV in repository root
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        candidate = parent / filename
        if candidate.exists():
            return str(candidate)
    # fallback to filename
    return filename


def load_and_preprocess(path=None):
    path = path or find_dataset_path()
    df = pd.read_csv(path, parse_dates=["Date"]) if Path(path).exists() else pd.DataFrame()
    if df.empty:
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = df.sort_values("Date").reset_index(drop=True)
    # keep essential columns
    cols = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    df = df[cols].dropna()
    # features
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    df['MA_5'] = df['Close'].rolling(window=5).mean().fillna(method='bfill')
    df['MA_10'] = df['Close'].rolling(window=10).mean().fillna(method='bfill')
    # target = next day close
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df
