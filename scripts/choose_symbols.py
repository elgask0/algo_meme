import os
import requests
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# Carga variables de entorno desde .env
load_dotenv(override=True)
API_KEY = os.getenv('COINAPI_KEY')

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_FILE = os.path.join(BASE_DIR, 'trading_data.db')
TICKERS_FILE = os.path.join(BASE_DIR, 'tickers.txt')


def fetch_candidates(generic: str) -> list:
    """
    Obtiene de CoinAPI todos los symbol_id para un asset genérico.
    """
    url = "https://rest.coinapi.io/v1/symbols"
    headers = {'Accept': 'application/json', 'X-CoinAPI-Key': API_KEY}
    params = {'filter_asset_id': generic.upper()}
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        print(f"[ERROR] CoinAPI {resp.status_code}: {resp.text}")
        return []
    return resp.json()

def parse_start(item: dict) -> datetime:
    ds = item.get('data_trade_start') or item.get('data_start')
    if not ds:
        return datetime.min
    if 'T' not in ds:
        ds += 'T00:00:00'
    ds = ds.rstrip('Z')
    try:
        return datetime.fromisoformat(ds)
    except ValueError:
        return datetime.min

def choose_mapping():
    """Interactivo: reasigna o confirma mapeos genérico → symbol_id.

    Nota: La tabla 'symbol_info' debe existir previamente en la base de datos.
    """
    with open(TICKERS_FILE) as f:
        tickers = [line.strip() for line in f if line.strip()]

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    for ticker in tickers:

        # Obtener candidatos y ordenarlos para SPOT y PERPETUAL
        type_map = {
            'SPOT': 'SPOT',
            'PERP': 'PERPETUAL'
        }
        for stype_label, stype_api in type_map.items():
            cur.execute(
                "SELECT 1 FROM symbol_info WHERE ticker = ? AND symbol_type = ?",
                (ticker, stype_api)
            )
            if cur.fetchone():
                print(f"'{ticker}' tipo {stype_label} ya mapeado, se omite.")
                continue

            candidates = fetch_candidates(ticker)
            candidates = [c for c in candidates if c.get('symbol_type') == stype_api]
            if not candidates:
                print(f"No hay candidatos para '{ticker}' tipo {stype_label}, se salta.")
                continue
            candidates.sort(key=parse_start)

            print(f"\nGenérico '{ticker}' tipo {stype_label} → {len(candidates)} candidatos:")
            for i, item in enumerate(candidates):
                start = item.get('data_trade_start') or item.get('data_start') or 'N/A'
                print(f" [{i}] {item['symbol_id']} ({item['exchange_id']}, {item['symbol_type']}), start: {start}")

            sel = input(f"Selecciona índice para {stype_label} de '{ticker}' (enter para saltar): ")
            if not sel.strip():
                print(f"'- Saltado '{ticker}' tipo {stype_label}.")
                continue
            try:
                idx = int(sel)
                chosen = candidates[idx]['symbol_id']
            except (ValueError, IndexError):
                print("Selección inválida, saltando.")
                continue

            cur.execute(
                "INSERT OR REPLACE INTO symbol_info (symbol_id, ticker, symbol_type) VALUES (?, ?, ?)",
                (chosen, ticker, stype_api)
            )
            conn.commit()
            print(f"→ '{ticker}' tipo {stype_label} → '{chosen}'")

    conn.close()
    print("Mapeo completado.")
    
if __name__ == '__main__':
    choose_mapping()