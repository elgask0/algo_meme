def process_ohlcv_for_date_task(symbol, date_str, period_id):
    """
    Wrapper que abre su propia conexión a SQLite y llama a ingest_ohlcv_for_date.
    """
    logs = []
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur  = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    # Evitar llamada a API si ya existen datos para este día (comparando fecha ISO)
    cur.execute(
        "SELECT 1 FROM coinapi_ohlcv WHERE symbol = ? AND substr(time_period_start, 1, 10) = ?;",
        (symbol, date_str)
    )
    if cur.fetchone():
        logs.append(f"[OHLCV] Ya existen datos para {symbol} en {date_str}, omitiendo API.")
        conn.close()
        return logs
    try:
        logs = ingest_ohlcv_for_date(symbol, date_str, conn, cur, period_id)
    except Exception as e:
        logs = [f"[OHLCV] Error en {symbol} para {date_str}: {e}"]
    conn.close()
    return logs

# --- Wrappers para symbol_info y funding ---
def process_symbol_info_task(symbol):
    """
    Wrapper que abre conexión propia y llama a ingest_symbol_info.
    """
    logs = []
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
    try:
        logs = ingest_symbol_info(symbol, conn, cur)
    except Exception as e:
        logs = [f"[CoinAPI] Error metadata {symbol}: {e}"]
    conn.close()
    return logs

def process_funding_task(symbol):
    """
    Wrapper que abre conexión propia y llama a ingest_mexc_funding.
    """
    logs = []
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    try:
        logs = ingest_mexc_funding(symbol, conn, cur)
    except Exception as e:
        logs = [f"[MEXC] Error en funding para {symbol}: {e}"]
    conn.close()
    return logs
import os
import time
import pandas as pd
import requests
from requests.exceptions import HTTPError
import sqlite3
import argparse
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

from concurrent.futures import ThreadPoolExecutor

# --- Helper para ejecución paralela y logs ordenados ---
def run_parallel(func, args_list, max_workers=4):
    """
    Ejecuta func(args) en paralelo, recopila listas de logs y las imprime en orden.
    """
    print(f"[{func.__name__}] Lanzando {len(args_list)} tareas en paralelo (max_workers={max_workers})")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for logs in executor.map(lambda arg: func(*arg), args_list):
            results.append(logs)
    # Imprimir los bloques de logs en orden
    for logs in results:
        for line in logs:
            print(line)

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
    logs = []
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

    # Evitar llamada a MEXC API si no hay datos nuevos
    if last_ms is not None:
        last_dt = pd.to_datetime(last_row[0])
        last_date = last_dt.date()
        if last_date >= datetime.now(timezone.utc).date() - timedelta(days=1):
            logs.append(
                f"[MEXC] Funding ya actualizado para {symbol} hasta {last_date.isoformat()}, omitiendo API."
            )
            return logs

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
    logs.append(f"[MEXC] Insertados {len(rows)} registros de funding para {symbol}")
    return logs

# Carga variables de entorno (.env)
load_dotenv()
API_KEY  = os.getenv('COINAPI_KEY')
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_FILE  = os.path.join(BASE_DIR, 'trading_data.db')

session = requests.Session()
session.headers.update({"X-CoinAPI-Key": API_KEY})

# Delay between CoinAPI requests to respect rate limits
RATE_LIMIT_DELAY = 0.5  # seconds

def safe_get(url, params=None, retries=3, delay=0.5):
    """
    Llama a session.get con reintentos exponenciales y maneja código 429 (Too Many Requests).
    """
    for attempt in range(retries):
        # Espacio fijo entre peticiones para no superar el rate limit
        time.sleep(RATE_LIMIT_DELAY)
        try:
            resp = session.get(url, params=params)
            resp.raise_for_status()
            return resp
        except HTTPError as e:
            status = e.response.status_code if e.response else None
            if status == 429:
                retry_after = e.response.headers.get("Retry-After")
                # Si viene cabecera Retry-After la usamos; si no, usamos tiempo exponencial
                wait = int(retry_after) if retry_after and retry_after.isdigit() else delay
                print(f"[CoinAPI] Rate limit hit (429). Retrying after {wait} seconds.")
                time.sleep(wait)
                delay *= 2
                continue
            # Para otros HTTP errors, relanzar
            raise
        except Exception as e:
            if attempt < retries - 1:
                print(f"[CoinAPI] Transient error ({e}). Retrying in {delay} seconds.")
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
    return data[0] if isinstance(data, list) and data else {}


def ingest_symbol_info(symbol: str, conn, cur):
    """
    Actualiza metadatos de un símbolo en la tabla symbol_info de SQLite.
    """
    logs = []
    # Evitar llamada a CoinAPI si ya hay metadata
    cur.execute(
        "SELECT data_end FROM symbol_info WHERE symbol_id = ?",
        (symbol,)
    )
    meta = cur.fetchone()
    if meta and meta[0]:
        logs.append(f"[CoinAPI] Metadata ya actualizada para {symbol} (data_end={meta[0]}), omitiendo API.")
        return logs

    info = fetch_symbol_info(symbol)
    if not info or 'symbol_id' not in info:
        logs.append(f"[CoinAPI] No se encontraron metadatos para {symbol}.")
        return logs

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
    logs.append(f"[CoinAPI] Metadata actualizada para {symbol}")
    return logs


def fetch_ohlcv_5min(symbol: str, time_start: str, time_end: str, period_id: str = "5MIN") -> pd.DataFrame:
    """
    Descarga OHLCV para un símbolo entre dos fechas (ISO strings) con period_id variable.
    """
    url = (
        f"https://rest.coinapi.io/v1/ohlcv/{symbol}/history"
        f"?period_id={period_id}"
        f"&time_start={time_start}&time_end={time_end}"
        f"&limit=1000&include_empty_items=true"
    )
    resp = safe_get(url)
    data = resp.json()
    df = pd.DataFrame(data)
    # Convertir timestamps a datetime
    df["time_period_start"] = pd.to_datetime(df["time_period_start"])
    df["time_period_end"]   = pd.to_datetime(df["time_period_end"])
    # Asegurar columnas time_open y time_close (algunos días pueden no incluirlas)
    if "time_open" in df.columns:
        df["time_open"] = pd.to_datetime(df["time_open"])
    else:
        df["time_open"] = df["time_period_start"]
    if "time_close" in df.columns:
        df["time_close"] = pd.to_datetime(df["time_close"])
    else:
        df["time_close"] = df["time_period_end"]
    # Índice temporal
    return df.set_index("time_period_start").sort_index()


def ingest_ohlcv(symbol: str, time_start: str, time_end: str, conn, cur, period_id: str = "5MIN"):
    """
    Inserta datos OHLCV en la tabla coinapi_ohlcv usando el period_id especificado.
    Retorna el número de filas insertadas.
    """
    # Extraer la fecha del periodo para logging
    date = time_start.split("T")[0] if "T" in time_start else time_start
    df = fetch_ohlcv_5min(symbol, time_start, time_end, period_id)
    # Filtrar periodos sin operaciones (trades_count == 0)
    total_bars = len(df)
    df = df[df['trades_count'] > 0]
    skipped = total_bars - len(df)
    # (No imprimir aquí)
    rows = []
    for ts, row in df.iterrows():
        rows.append(
            (
                symbol,
                ts.isoformat(),
                row["time_period_end"].isoformat(),
                row["time_open"].isoformat(),
                row["time_close"].isoformat(),
                float(row["price_open"]),
                float(row["price_high"]),
                float(row["price_low"]),
                float(row["price_close"]),
                float(row["volume_traded"]),
                int(row["trades_count"])
            )
        )
    sql = """
        INSERT OR REPLACE INTO coinapi_ohlcv
        (symbol, time_period_start, time_period_end,
         time_open, time_close,
         price_open, price_high, price_low, price_close,
         volume_traded, trades_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    cur.executemany(sql, rows)
    conn.commit()
    return len(rows)

def ingest_ohlcv_for_date(symbol: str, date_str: str, conn, cur, period_id: str = "5MIN"):
    """
    Ingesta OHLCV para un único día (00:00 UTC a 24:00 UTC).
    Devuelve lista de logs.
    """
    logs = []
    start = f"{date_str}T00:00:00"
    end_dt = datetime.fromisoformat(date_str) + timedelta(days=1)
    end = end_dt.isoformat()
    # Llamar a ingest_ohlcv y obtener cantidad de filas
    rows_inserted = ingest_ohlcv(symbol, start, end, conn, cur, period_id)
    # Si no hay datos, insertar un marcador para evitar reintentos futuros
    if rows_inserted == 0:
        cur.execute(
            """
            INSERT OR REPLACE INTO coinapi_ohlcv
            (symbol, time_period_start, time_period_end,
             time_open, time_close,
             price_open, price_high, price_low, price_close,
             volume_traded, trades_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                start,
                end,
                start,
                end,
                0, 0, 0, 0, 0, 0
            )
        )
        conn.commit()
        logs.append(f"[OHLCV] Sin datos para {symbol} en {date_str}, insertado marcador.")
    else:
        logs.append(f"[OHLCV] Insertados {rows_inserted} registros de OHLCV ({period_id}) para {symbol} en {date_str}")
    return logs


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
    Devuelve lista de logs.
    """
    logs = []
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
    else:
        # Insertar marcador para días sin snapshots y evitar reintentos
        placeholder = (
            symbol,
            f"{date}T00:00:00",
            date,
            # 12 campos de None para bid/ask px y size
            None, None, None, None, None, None, None, None, None, None, None, None
        )
        cur.execute(sql, placeholder)
    conn.commit()

    if rows:
        logs.append(f"[Orderbook] Prepared {len(rows)} rows for {symbol} on {date}")
    else:
        logs.append(f"[Orderbook] Sin snapshots para {symbol} en {date}, insertado marcador.")
    return logs


# --- Wrapper para procesamiento de orderbook por fecha ---
def process_orderbook_for_date_task(symbol, date_str):
    """
    Wrapper que abre su propia conexión y procesa orderbook para una fecha concreta.
    """
    logs = []
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur  = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    # Omite si ya existe marcador o datos para este día
    cur.execute(
        "SELECT 1 FROM coinapi_orderbook WHERE symbol_id = ? AND date = ?;",
        (symbol, date_str)
    )
    if cur.fetchone():
        logs.append(f"[Orderbook] Ya existen datos para {symbol} en {date_str}, omitiendo API.")
        conn.close()
        return logs
    try:
        logs = ingest_orderbook(symbol, date_str, conn, cur)
    except Exception as e:
        logs = [f"[Orderbook] Error en {symbol} para {date_str}: {e}"]
    conn.close()
    return logs




def main():
    parser = argparse.ArgumentParser(description="Ingestión de datos CoinAPI")
    parser.add_argument('--symbol-info', action='store_true', help='Actualizar metadata de símbolos')
    parser.add_argument('--orderbook',   action='store_true', help='Ingestar orderbook histórico por día')
    parser.add_argument('--funding', action='store_true', help='Ingestar histórico de funding rate de MEXC')
    parser.add_argument('--symbol', '-s', nargs='+', type=str, help='Ingestar datos solo para los símbolos especificados; por defecto ingesta todos.')
    parser.add_argument('--ohlcv', action='store_true', help='Ingestar OHLCV histórico por rango de fechas')
    parser.add_argument('--period-id', '-p', type=str, default='5MIN', help='Period ID para OHLCV, p.ej. 5MIN, 1HRS, etc.')
    args = parser.parse_args()

    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()

    if args.symbol_info:
        # Obtener lista de símbolos
        if args.symbol:
            symbols = args.symbol
        else:
            cur.execute("SELECT symbol_id FROM symbol_info;")
            symbols = [r[0] for r in cur.fetchall()]
        conn.close()
        # Ejecutar en paralelo con wrappers
        tasks = [(sym,) for sym in symbols]
        run_parallel(process_symbol_info_task, tasks)
        return

    if args.orderbook:
        # Obtener símbolos y fechas pendientes
        if args.symbol:
            symbols = args.symbol
        else:
            cur.execute("SELECT symbol_id, data_start FROM symbol_info;")
            symbols = cur.fetchall()  # [(symbol, data_start), ...]

        tasks = []
        today = datetime.now(timezone.utc).date()
        for symbol, start_str in symbols:
            start_date = datetime.fromisoformat(start_str).date()
            curr = start_date
            while curr <= today:
                date_iso = curr.isoformat()
                cur.execute(
                    "SELECT 1 FROM coinapi_orderbook WHERE symbol_id = ? AND date = ?;",
                    (symbol, date_iso)
                )
                if not cur.fetchone():
                    tasks.append((symbol, date_iso))
                curr += timedelta(days=1)

        # Si no hay tareas pendientes, omitir
        if not tasks:
            print("[Orderbook] No hay fechas pendientes para ningún símbolo, omitiendo descarga.")
            conn.close()
            return

        conn.close()
        # Ejecutar en paralelo tareas por fecha
        run_parallel(process_orderbook_for_date_task, tasks, max_workers=8)
        return

    if args.ohlcv:
        # Lista de símbolos a procesar
        if args.symbol:
            symbols = args.symbol
        else:
            cur.execute("SELECT symbol_id FROM symbol_info;")
            symbols = [r[0] for r in cur.fetchall()]

        # Detectar tareas pendientes: (símbolo, fecha, period_id)
        tasks = []
        for sym in symbols:
            cur.execute(
                "SELECT data_start, data_end FROM symbol_info WHERE symbol_id = ?;",
                (sym,)
            )
            row = cur.fetchone()
            if not row or not row[0]:
                # No metadata
                tasks.append((sym, None, args.period_id))
                continue
            start_date = datetime.fromisoformat(row[0]).date()
            end_date   = datetime.now(timezone.utc).date()
            curr = start_date
            while curr <= end_date:
                cur.execute(
                    "SELECT 1 FROM coinapi_ohlcv WHERE symbol = ? AND date(time_period_start) = ?;",
                    (sym, curr.isoformat())
                )
                if not cur.fetchone():
                    tasks.append((sym, curr.isoformat(), args.period_id))
                curr += timedelta(days=1)

        # Si no hay tareas pendientes, salir sin llamar a la API
        if not tasks:
            print("[OHLCV] No hay fechas pendientes para ningún símbolo, omitiendo descarga.")
            conn.close()
            return

        conn.close()
        # Ejecutar en paralelo con conexiones propias
        run_parallel(process_ohlcv_for_date_task, tasks)
        return

    if args.funding:
        # Obtener lista de símbolos perp
        if args.symbol:
            symbols = args.symbol
        else:
            cur.execute(
                "SELECT symbol_id FROM symbol_info WHERE symbol_id LIKE 'MEXCFTS_PERP_%';"
            )
            symbols = [r[0] for r in cur.fetchall()]
        conn.close()
        # Ejecutar en paralelo con wrappers
        tasks = [(sym,) for sym in symbols]
        run_parallel(process_funding_task, tasks)
        return

    conn.close()

if __name__ == '__main__':
    main()