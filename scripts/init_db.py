import sqlite3
import os

def main():
    # Borra la base de datos si existe (opcional, para recrear desde cero)
    # if os.path.exists('trading_data.db'):
    #     os.remove('trading_data.db')

    conn = sqlite3.connect('trading_data.db')
    cur = conn.cursor()

    # 1) Tabla de información de símbolos (ticker y metadatos)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS symbol_info (
      symbol_id       TEXT PRIMARY KEY,
      ticker          TEXT,
      exchange_id     TEXT,
      symbol_type     TEXT,
      asset_id_base   TEXT,
      asset_id_quote  TEXT,
      data_start      TEXT,
      data_end        TEXT
    );
    """)

    # 2) Tabla de datos OHLCV (5-min)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS coinapi_ohlcv (
      symbol             TEXT,
      time_period_start  TEXT,
      time_period_end    TEXT,
      time_open          TEXT,
      time_close         TEXT,
      price_open         REAL,
      price_high         REAL,
      price_low          REAL,
      price_close        REAL,
      volume_traded      REAL,
      trades_count       INTEGER,
      PRIMARY KEY(symbol, time_period_start)
    );
    """)

    # Eliminar tabla coinapi_orderbook antigua
    cur.execute("DROP TABLE IF EXISTS coinapi_orderbook;")

    # 3) Tabla de snapshots de orderbook (5 niveles, 5-min)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS coinapi_orderbook (
      symbol_id   TEXT,
      ts          TEXT,
      date        TEXT,
      bid1_px     REAL, bid1_sz   REAL,
      bid2_px     REAL, bid2_sz   REAL,
      bid3_px     REAL, bid3_sz   REAL,
      ask1_px     REAL, ask1_sz   REAL,
      ask2_px     REAL, ask2_sz   REAL,
      ask3_px     REAL, ask3_sz   REAL,
      PRIMARY KEY(symbol_id, ts)
    );
    """)

    # 4) Tabla para seguimiento de ingestión diaria de orderbook
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ingestion_progress (
      symbol_id TEXT PRIMARY KEY,
      last_date TEXT
    );
    """)

    # Índice para acelerar búsquedas por símbolo y fecha
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_date
      ON coinapi_orderbook(symbol_id, date);
    """)

    # 5) Tabla de funding rate histórico de MEXC
    cur.execute("""
    CREATE TABLE IF NOT EXISTS mexc_funding_rate_history (
      symbol         TEXT,
      ts             TEXT,
      funding_rate   REAL,
      collect_cycle  INTEGER,
      PRIMARY KEY(symbol, ts)
    );
    """)

    conn.commit()
    conn.close()
    print("Esquema inicial creado en trading_data.db con tablas symbol_info, coinapi_ohlcv y coinapi_orderbook.")

if __name__ == '__main__':
    main()
