from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import threading
from .model import train, predict_next_from_last, load_model

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Tesla Buy-Signal Demo")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.on_event("startup")
def on_startup():
    # Train model in background if not present
    def _train_if_missing():
        if not (BASE_DIR / 'model.joblib').exists():
            train()
    t = threading.Thread(target=_train_if_missing, daemon=True)
    t.start()


@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})


@app.get('/api/signal')
async def api_signal():
    model = load_model()
    if model is None:
        # train synchronously as fallback
        train()
    info = predict_next_from_last()
    # simple rule: buy if predicted return > 0.007 (0.7%) OR predicted_next_close > MA_5
    signal = 'HOLD'
    reason = ''
    if info['predicted_return'] > 0.007:
        signal = 'BUY'
        reason = 'predicted return > 0.7%'
    return {**info, 'signal': signal, 'reason': reason}
