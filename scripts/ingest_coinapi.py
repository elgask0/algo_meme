import os
import time
import pandas as pd
import requests
import sqlite3
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

from concurrent.futures import ThreadPoolExecutor

# --- Utilidades específicas para símbolos MEXC --------------------------------
MEXC_PERP_PREFIX = "MEXCFTS_PERP_"

def mexc_api_symbol(symbol_id: str) -> str:
    """
    Convierte un `symbol_id` almacenado (ej. 'MEXCFTS_PERP_SPX_USDT')
    en el nombre de instrumento que la API pública de MEXC espera
    (ej. 'SPX_USDT'). Si el símbolo no lleva el prefijo, se devuelve tal cual.
    """
    return symbol_id[len(MEXC_PERP_PREFIX):] if symbol_id.startswith(MEXC_PERP_PREFIX) else symbol_id

def fetch_mexc_funding_rate_history(symbol: str, page_size: int = 1000) -> list:
    """
    Descarga todo el histórico de funding rates de MEXC para un símbolo dado.
    """
    url = "https://contract.mexc.com/api/v1/contract/funding_rate/history"
    all_records = []
    page = 1
    while True:
        params = {"symbol": symbol, "page_num": page, "page_size": page_size}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("success", False):
            raise RuntimeError(f"Error en API MEXC: {payload}")
        data = payload["data"]
        all_records.extend(data["resultList"])
        if page >= data["totalPage"]:
            break
        page += 1
    return all_records

def ingest_mexc_funding(symbol: str, conn, cur):
    """
    Inserta el histórico de funding rates de MEXC en la tabla mexc_funding_rate_history.
    """
    api_sym = mexc_api_symbol(symbol)
    # Obtener el último timestamp almacenado para este símbolo
    cur.execute(
        "SELECT MAX(ts) FROM mexc_funding_rate_history WHERE symbol = ?",
        (symbol,)
    )
    last_row = cur.fetchone()
    if last_row and last_row[0]:
        last_dt = pd.to_datetime(last_row[0])
        last_ms = int(last_dt.timestamp() * 1000)
    else:
        last_ms = None

    all_records = fetch_mexc_funding_rate_history(api_sym)
    # Filtrar solo los funding rates posteriores al último registrado
    records = [
        rec for rec in all_records
        if last_ms is None or rec.get('settleTime', 0) > last_ms
    ]
    rows = []
    for rec in records:
        ts_iso = pd.to_datetime(rec['settleTime'], unit='ms').isoformat()
        rows.append((symbol, ts_iso, rec['fundingRate'], rec['collectCycle']))
    sql = """
        INSERT OR REPLACE INTO mexc_funding_rate_history
        (symbol, ts, funding_rate, collect_cycle)
        VALUES (?, ?, ?, ?);
    """
    if rows:
        cur.executemany(sql, rows)
        conn.commit()
    print(f"[MEXC] Insertados {len(rows)} registros de funding para {symbol}")

# Carga variables de entorno (.env)
load_dotenv()
API_KEY  = os.getenv('COINAPI_KEY')
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_FILE  = os.path.join(BASE_DIR, 'trading_data.db')

# Initialize a persistent HTTP session to reuse connections and set API key header
session = requests.Session()
session.headers.update({"X-CoinAPI-Key": API_KEY})

def safe_get(url, params=None, retries=2, delay=0.5):
    """
    Llama a session.get con reintentos exponenciales ante fallos transitorios.
    """
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise

def fetch_symbol_info(symbol: str) -> dict:
    """
    Obtiene metadatos de un símbolo de CoinAPI.
    """
    url    = "https://rest.coinapi.io/v1/symbols"
    params = {"filter_symbol_id": symbol}
    resp   = safe_get(url, params=params)
    data   = resp.json()
    return data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})

def ingest_symbol_info(symbol: str, conn, cur):
    """
    Actualiza metadatos de un símbolo en la tabla symbol_info de SQLite.
    """
    info = fetch_symbol_info(symbol)
    if not info or 'symbol_id' not in info:
        print(f"[CoinAPI] No se encontraron metadatos para {symbol}.")
        return

    cur.execute(
        """
        UPDATE symbol_info
        SET exchange_id    = ?,
            symbol_type    = ?,
            asset_id_base  = ?,
            asset_id_quote = ?,
            data_start     = ?,
            data_end       = ?
        WHERE symbol_id = ?;
        """,
        (
            info.get('exchange_id'),
            info.get('symbol_type'),
            info.get('asset_id_base'),
            info.get('asset_id_quote'),
            info.get('data_start'),
            info.get('data_end'),
            symbol
        )
    )
    conn.commit()
    print(f"[CoinAPI] Metadata actualizada para {symbol}")

def fetch_orderbook_5min(symbol: str, date: str) -> pd.DataFrame:
    """
    Descarga snapshots L2 y remuestrea a 5-min para un símbolo y fecha dados,
    manteniendo hasta 3 niveles de profundidad.
    """
    url    = f"https://rest.coinapi.io/v1/orderbooks/{symbol}/history"
    params = {"date": date, "limit_levels": 3}
    resp   = safe_get(url, params=params)
    snaps  = resp.json()

    records = []
    for snap in snaps:
        ts  = pd.to_datetime(snap["time_exchange"])
        rec = {"ts": ts}
        bids = snap.get("bids", [])[:3]
        asks = snap.get("asks", [])[:3]
        for i in range(3):
            bid = bids[i] if i < len(bids) else {"price": None, "size": None}
            ask = asks[i] if i < len(asks) else {"price": None, "size": None}
            rec[f"bid{i+1}_px"] = bid["price"]
            rec[f"bid{i+1}_sz"] = bid["size"]
            rec[f"ask{i+1}_px"] = ask["price"]
            rec[f"ask{i+1}_sz"] = ask["size"]
        records.append(rec)

    df = (
        pd.DataFrame(records)
          .set_index("ts")
          .sort_index()
          .resample("5min")
          .first()
          .dropna(how="all")
    )
    return df

def ingest_orderbook(symbol: str, date: str, conn, cur):
    """
    Inserta datos de orderbook 5-min (hasta 3 niveles) en la tabla coinapi_orderbook.
    """
    df = fetch_orderbook_5min(symbol, date)

    placeholders = ", ".join(["?"] * 15)
    sql = f"""
        INSERT OR REPLACE INTO coinapi_orderbook
        (symbol_id, ts, date,
         bid1_px, bid1_sz, bid2_px, bid2_sz, bid3_px, bid3_sz,
         ask1_px, ask1_sz, ask2_px, ask2_sz, ask3_px, ask3_sz)
        VALUES ({placeholders});
    """

    rows = []
    for ts, row in df.iterrows():
        values = []
        for attr in [
            'bid1_px', 'bid1_sz', 'bid2_px', 'bid2_sz',
            'bid3_px', 'bid3_sz', 'ask1_px', 'ask1_sz',
            'ask2_px', 'ask2_sz', 'ask3_px', 'ask3_sz'
        ]:
            val = getattr(row, attr)
            if pd.isna(val):
                values.append(None)
            else:
                values.append(float(val))
        rows.append((symbol, ts.isoformat(), date, *values))

    if rows:
        cur.executemany(sql, rows)
    # Commit siempre para asegurar persistencia incluso sin datos
    conn.commit()
    print(f"[Orderbook] Prepared {len(rows)} rows for {symbol} on {date}")

def process_symbol(symbol_row):
    symbol, start_str, end_str = symbol_row
    # Abrir conexión SQLite propia para este hilo
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur  = conn.cursor()
    # PRAGMAs para rendimiento
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA cache_size=10000;")

    # Última fecha procesada
    cur.execute("SELECT last_date FROM ingestion_progress WHERE symbol_id = ?", (symbol,))
    row = cur.fetchone()
    if row and row[0]:
        last_date = datetime.fromisoformat(row[0]).date()
    else:
        last_date = datetime.fromisoformat(start_str).date() - timedelta(days=1)

    # Extiende el rango hasta ayer como mínimo para evitar bloqueos si data_end está atrasado
    stored_end   = datetime.fromisoformat(end_str).date()
    today_minus1 = datetime.now(timezone.utc).date() - timedelta(days=1)
    end_date     = max(stored_end, today_minus1)

    # Fechas pendientes
    pending_dates = []
    curr = last_date + timedelta(days=1)
    while curr <= end_date:
        cur.execute(
            "SELECT 1 FROM coinapi_orderbook WHERE symbol_id = ? AND date = ?",
            (symbol, curr.isoformat())
        )
        if not cur.fetchone():
            pending_dates.append(curr.isoformat())
        curr += timedelta(days=1)

    # Procesar en batch con captura de errores
    for date_str in pending_dates:
        try:
            ingest_orderbook(symbol, date_str, conn, cur)
            # Actualizar progreso siempre, incluso si no se insertaron filas
            cur.execute(
                "INSERT OR REPLACE INTO ingestion_progress(symbol_id, last_date) VALUES (?, ?)",
                (symbol, date_str)
            )
            # Commit tras cada fecha (datos + progreso)
            conn.commit()
        except Exception as e:
            # Log sin abortar el hilo
            print(f"[Orderbook] Error en {symbol} {date_str}: {e}")
            continue

    # conn.commit()  # Removed commit at the end of process_symbol
    conn.close()
    print(f"[Orderbook] Completed {len(pending_dates)} days for {symbol}")

def main():
    parser = argparse.ArgumentParser(description="Ingestión de datos CoinAPI")
    parser.add_argument('--symbol-info', action='store_true', help='Actualizar metadata de símbolos')
    parser.add_argument('--orderbook',   action='store_true', help='Ingestar orderbook histórico por día')
    parser.add_argument('--funding', action='store_true',
                        help='Ingestar histórico de funding rate de MEXC')
    parser.add_argument('--symbol', '-s', type=str, help='Ingestar orderbook para un símbolo específico; por defecto ingesta todos.')
    args = parser.parse_args()

    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()

    if args.symbol_info:
        cur.execute("SELECT symbol_id FROM symbol_info;")
        symbols = [r[0] for r in cur.fetchall()]
        for sym in symbols:
            try:
                ingest_symbol_info(sym, conn, cur)
            except Exception as e:
                print(f"[CoinAPI] Error metadata {sym}: {e}")

    if args.orderbook:
        if args.symbol:
            cur.execute(
                "SELECT symbol_id, data_start, data_end FROM symbol_info WHERE symbol_id = ?;",
                (args.symbol,)
            )
            symbol_rows = cur.fetchall()
            if not symbol_rows:
                print(f"[Error] Símbolo {args.symbol} no encontrado en symbol_info.")
                return
        else:
            cur.execute("SELECT symbol_id, data_start, data_end FROM symbol_info;")
            symbol_rows = cur.fetchall()
        conn.close()  # Cada hilo abre su propia conexión
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(process_symbol, symbol_rows)
        return

    if args.funding:
        if args.symbol:
            symbols = [args.symbol]  # Solo el símbolo indicado por CLI
        else:
            # Todos los contratos perpetuos registrados en symbol_info
            cur.execute(
                "SELECT symbol_id FROM symbol_info WHERE symbol_id LIKE 'MEXCFTS_PERP_%';"
            )
            symbols = [r[0] for r in cur.fetchall()]

        for sym in symbols:
            try:
                ingest_mexc_funding(sym, conn, cur)
            except Exception as e:
                print(f"[MEXC] Error en funding para {sym}: {e}")
                continue
        conn.close()
        return

    conn.close()

if __name__ == '__main__':
    main()