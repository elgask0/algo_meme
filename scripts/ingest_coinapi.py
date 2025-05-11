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
    """
    # Extraer la fecha del periodo para logging
    date = time_start.split("T")[0] if "T" in time_start else time_start
    df = fetch_ohlcv_5min(symbol, time_start, time_end, period_id)
    # Filtrar periodos sin operaciones (trades_count == 0)
    total_bars = len(df)
    df = df[df['trades_count'] > 0]
    skipped = total_bars - len(df)
    if skipped > 0:
        print(f"[OHLCV] Saltados {skipped} periodos vacíos para {symbol} en {date}")
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
    print(f"[OHLCV] Insertados {len(rows)} registros de OHLCV ({period_id}) para {symbol} en {date}")

def ingest_ohlcv_for_date(symbol: str, date_str: str, conn, cur, period_id: str = "5MIN"):
    """
    Ingesta OHLCV para un único día (00:00 UTC a 24:00 UTC).
    """
    start = f"{date_str}T00:00:00"
    end_dt = datetime.fromisoformat(date_str) + timedelta(days=1)
    end = end_dt.isoformat()
    ingest_ohlcv(symbol, start, end, conn, cur, period_id)


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
    conn.commit()
    print(f"[Orderbook] Prepared {len(rows)} rows for {symbol} on {date}")


def process_symbol(symbol_row):
    symbol, start_str, end_str = symbol_row
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur  = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA cache_size=10000;")

    cur.execute("SELECT last_date FROM ingestion_progress WHERE symbol_id = ?", (symbol,))
    row = cur.fetchone()
    if row and row[0]:
        last_date = datetime.fromisoformat(row[0]).date()
    else:
        last_date = datetime.fromisoformat(start_str).date() - timedelta(days=1)

    stored_end   = datetime.fromisoformat(end_str).date()
    today_minus1 = datetime.now(timezone.utc).date() - timedelta(days=1)
    end_date     = max(stored_end, today_minus1)

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

    for date_str in pending_dates:
        try:
            ingest_orderbook(symbol, date_str, conn, cur)
            cur.execute(
                "INSERT OR REPLACE INTO ingestion_progress(symbol_id, last_date) VALUES (?, ?)",
                (symbol, date_str)
            )
            conn.commit()
        except Exception as e:
            print(f"[Orderbook] Error en {symbol} {date_str}: {e}")
            continue

    conn.close()
    print(f"[Orderbook] Completed {len(pending_dates)} days for {symbol}")


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
        # Ingestar metadata solo para símbolos especificados o todos si no se indica
        if args.symbol:
            symbols = args.symbol
        else:
            cur.execute("SELECT symbol_id FROM symbol_info;")
            symbols = [r[0] for r in cur.fetchall()]
        for sym in symbols:
            try:
                ingest_symbol_info(sym, conn, cur)
            except Exception as e:
                print(f"[CoinAPI] Error metadata {sym}: {e}")

    if args.orderbook:
        if args.symbol:
            # Construir lista de rows para cada símbolo dado
            symbol_rows = []
            for sym in args.symbol:
                cur.execute(
                    "SELECT symbol_id, data_start, data_end FROM symbol_info WHERE symbol_id = ?;",
                    (sym,)
                )
                row = cur.fetchone()
                if not row:
                    print(f"[Error] Símbolo {sym} no encontrado en symbol_info.")
                else:
                    symbol_rows.append(row)
            if not symbol_rows:
                return
        else:
            cur.execute("SELECT symbol_id, data_start, data_end FROM symbol_info;")
            symbol_rows = cur.fetchall()
        conn.close()
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(process_symbol, symbol_rows)
        return

    if args.ohlcv:
        # Lista de símbolos a procesar
        if args.symbol:
            symbols = args.symbol
        else:
            cur.execute("SELECT symbol_id FROM symbol_info;")
            symbols = [r[0] for r in cur.fetchall()]

        for sym in symbols:
            # Obtener rango disponible desde symbol_info
            cur.execute(
                "SELECT data_start, data_end FROM symbol_info WHERE symbol_id = ?;",
                (sym,)
            )
            row = cur.fetchone()
            if not row or not row[0]:
                print(f"[OHLCV] No hay metadata de fechas para {sym}. Ejecute --symbol-info primero.")
                continue
            # Rango: desde data_start hasta hoy (fecha de ejecución)
            data_start, _ = row
            start_date = datetime.fromisoformat(data_start).date()
            # Utilizar la fecha de hoy UTC como fecha final
            end_date = datetime.now(timezone.utc).date()

            # Detectar fechas faltantes
            pending_dates = []
            curr = start_date
            while curr <= end_date:
                cur.execute(
                    "SELECT 1 FROM coinapi_ohlcv WHERE symbol = ? AND date(time_period_start) = ?;",
                    (sym, curr.isoformat())
                )
                if not cur.fetchone():
                    pending_dates.append(curr.isoformat())
                curr += timedelta(days=1)

            # Ingestar solo días faltantes
            for date_str in pending_dates:
                try:
                    ingest_ohlcv_for_date(sym, date_str, conn, cur, args.period_id)
                except Exception as e:
                    print(f"[OHLCV] Error en {sym} para {date_str}: {e}")
                    continue

        conn.close()
        return

    if args.funding:
        # Ingestar funding rates para símbolos especificados o todos los perp si no se indica
        if args.symbol:
            symbols = args.symbol
        else:
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