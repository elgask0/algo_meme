import sqlite3
import pandas as pd
import pandera as pa
from pandera import DataFrameSchema, Column, Check
import numpy as np
from scipy.stats import median_abs_deviation
import os
import argparse
from dotenv import load_dotenv


load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DB_FILE = os.path.join(BASE_DIR, "trading_data.db")
DEFAULT_FREQ_CANDLE = "5min"
DEFAULT_FREQ_ORDERBOOK = "5s"
DEFAULT_MAX_GAP_CANDLES = 6

# Esquema de validación con Pandera
OHLCV_SCHEMA = DataFrameSchema(
    {
        "symbol": Column(str, nullable=False),
        "time_period_start": Column(pa.DateTime, nullable=False),
        "time_period_end": Column(pa.DateTime, nullable=False),
        "time_open": Column(pa.DateTime, nullable=True),
        "time_close": Column(pa.DateTime, nullable=True),
        "price_open": Column(float, Check.ge(0), nullable=False),
        "price_high": Column(float, Check.ge(0), nullable=False),
        "price_low": Column(float, Check.ge(0), nullable=False),
        "price_close": Column(float, Check.ge(0), nullable=False),
        "volume_traded": Column(float, Check.ge(0), nullable=False),
        "trades_count": Column(int, Check.ge(0), nullable=False),
    }
)

# Esquema de validación para snapshots de orderbook
ORDERBOOK_SCHEMA = DataFrameSchema(
    {
        "symbol_id": Column(str, nullable=False),
        "ts": Column(pa.DateTime, nullable=False),
        "date": Column(pa.DateTime, nullable=False),
        "bid1_px": Column(float, Check.ge(0), nullable=False),
        "bid1_sz": Column(float, Check.gt(0), nullable=False),
        "bid2_px": Column(float, Check.ge(0), nullable=False),
        "bid2_sz": Column(float, Check.gt(0), nullable=False),
        "bid3_px": Column(float, Check.ge(0), nullable=False),
        "bid3_sz": Column(float, Check.gt(0), nullable=False),
        "ask1_px": Column(float, Check.ge(0), nullable=False),
        "ask1_sz": Column(float, Check.gt(0), nullable=False),
        "ask2_px": Column(float, Check.ge(0), nullable=False),
        "ask2_sz": Column(float, Check.gt(0), nullable=False),
        "ask3_px": Column(float, Check.ge(0), nullable=False),
        "ask3_sz": Column(float, Check.gt(0), nullable=False),
    }
)

# Funciones del pipeline


def load_data(db_path: str, symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
    SELECT *
    FROM coinapi_ohlcv
    WHERE symbol = ?
    """
    df = pd.read_sql(
        query,
        conn,
        params=[symbol],
        parse_dates=["time_period_start", "time_period_end", "time_open", "time_close"],
    )
    conn.close()
    # Imputar timestamps faltantes
    df["time_open"] = df["time_open"].fillna(df["time_period_start"])
    df["time_close"] = df["time_close"].fillna(df["time_period_end"])
    # Eliminar zona horaria
    for col in ["time_period_start", "time_period_end", "time_open", "time_close"]:
        df[col] = df[col].dt.tz_localize(None)
    # Eliminar filas sin OHLCV completo
    before = len(df)
    df = df.dropna(subset=["price_open", "price_high", "price_low", "price_close"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[load_data] Eliminadas {dropped} filas sin datos OHLCV para {symbol}")
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Validación eager (excepción en la primera infracción)
    OHLCV_SCHEMA.validate(df, lazy=False)
    # Validación lazy (devuelve df limpio o lanza SchemaErrors)
    return OHLCV_SCHEMA.validate(df, lazy=True)


def reindex_time(df: pd.DataFrame, freq_candle: str) -> pd.DataFrame:
    idx = pd.date_range(
        start=df.time_period_start.min(),
        end=df.time_period_start.max(),
        freq=freq_candle,
    )
    out = df.set_index("time_period_start").reindex(idx)
    out = out.rename_axis("time_period_start").reset_index()
    out["symbol"] = df["symbol"].iloc[0]
    # Rellenar metadatos de tiempo y volumen/trades para las nuevas filas
    freq_delta = pd.Timedelta(freq_candle)
    out["time_period_end"] = out["time_period_start"] + freq_delta
    out["time_open"] = out["time_open"].fillna(out["time_period_start"])
    out["time_close"] = out["time_close"].fillna(out["time_period_end"])
    out["volume_traded"] = out["volume_traded"].fillna(0)
    out["trades_count"] = out["trades_count"].fillna(0)
    return out


def detect_structure(df: pd.DataFrame) -> int:
    conds = (
        (df.price_low <= df.price_open)
        & (df.price_open <= df.price_high)
        & (df.price_low <= df.price_close)
        & (df.price_close <= df.price_high)
        & (df.volume_traded >= 0)
        & (df.trades_count >= 0)
    )
    df["flag_bad_structure"] = ~conds
    return df["flag_bad_structure"].sum()


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MAD diario
    def flag_mad(sub):
        price_cols = ["price_open", "price_high", "price_low", "price_close"]
        outlier_mask = pd.Series(False, index=sub.index)
        for col in price_cols:
            series = sub[col].dropna()
            if series.empty:
                continue
            med = series.median()
            mad = median_abs_deviation(series, scale="normal") or 1e-9
            outlier_mask |= ~sub[col].between(med - 5 * mad, med + 5 * mad)
        sub["flag_outlier_mad"] = outlier_mask
        return sub

    df = df.groupby(df.time_period_start.dt.date, group_keys=False).apply(flag_mad)
    # Saltos bruscos
    df["delta_open"] = df.price_open.diff().abs()
    sigma = df.delta_open.rolling(12).std()
    sigma = sigma.replace(0, np.nan).ffill()
    df["flag_jump"] = df.delta_open > 10 * sigma
    return df


def impute_data(df: pd.DataFrame, max_gap_candles: int) -> pd.DataFrame:
    df = df.copy()
    # Convertir outliers a NaN y preparar relleno controlado
    mask_mad = df["flag_outlier_mad"]
    price_cols = ["price_open", "price_high", "price_low", "price_close"]
    df.loc[mask_mad, price_cols] = np.nan
    df.loc[mask_mad, ["volume_traded", "trades_count"]] = np.nan

    # Rellenado controlado para velas de 5 min
    max_gap = max_gap_candles  # número máximo de velas consecutivas para interpolar

    # Establecer índice temporal en time_period_start para interpolación basada en tiempo
    df = df.set_index("time_period_start")

    # Interpolación lineal para huecos cortos (≤ max_gap velas)
    df[price_cols] = df[price_cols].interpolate(
        method="time", limit=max_gap, limit_area="inside"
    )

    # Forward-fill para huecos más largos, limitado a 1 vela
    df[price_cols] = df[price_cols].ffill(limit=1)

    # Resetear índice al temporal original
    df = df.reset_index()

    # Rellenar volumen y conteo de trades con 0
    df["volume_traded"] = df["volume_traded"].fillna(0)
    df["trades_count"] = df["trades_count"].fillna(0)

    # ---------- FASE DE RELLENO FINAL ----------
    # Asegurar que no queden NaNs en los precios: forward‑fill y backward‑fill
    df[price_cols] = df[price_cols].ffill().bfill()

    # Reconstruir rangos coherentes en cualquier fila afectada
    mask_imputed = (
        df["flag_outlier_mad"] | df["price_high"].isna() | df["price_low"].isna()
    )
    df.loc[mask_imputed, "price_high"] = df.loc[mask_imputed, price_cols].max(axis=1)
    df.loc[mask_imputed, "price_low"] = df.loc[mask_imputed, price_cols].min(axis=1)


    return df


def reduce_columns(df: pd.DataFrame) -> pd.DataFrame:
    final_cols = [
        "time_period_start",
        "symbol",
        "time_period_end",
        "time_open",
        "time_close",
        "price_open",
        "price_high",
        "price_low",
        "price_close",
        "volume_traded",
        "trades_count",
        "flag_bad_structure",
        "flag_outlier_mad",
        "flag_jump",
    ]
    return df[final_cols]


def final_data_check(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    print(f"Filas: {rows}, Columnas: {cols}")
    print("Tipos de datos:")
    print(df.dtypes)
    dates = df.select_dtypes(include=["datetime64[ns]"])
    if not dates.empty:
        print("Rango fechas:")
        for c in dates:
            print(f" {c}: {df[c].min()} → {df[c].max()}")
    nums = df.select_dtypes(include=["number"])
    if not nums.empty:
        stats = nums.agg(["min", "max"]).T
        print("Min/Max numéricos:")
        print(stats)
    nulls = df.isna().mean().mul(100).round(2)
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        print("Valores nulos (%):")
        print(nulls)


# ----------------- ORDERBOOK PIPELINE ----------------- #
def load_orderbook(db_path: str, symbol: str) -> pd.DataFrame:
    """
    Carga el orderbook para un símbolo desde la tabla orderbook.
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT *
    FROM coinapi_orderbook
    WHERE symbol_id = ?
    """
    df = pd.read_sql(
        query,
        conn,
        params=[symbol],
    )
    conn.close()

    # Parseo de timestamps
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # Validación inicial con Pandera (eager)
    ORDERBOOK_SCHEMA.validate(df, lazy=False)

    # Orden y duplicados
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def detect_ob_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca filas con estructura inválida:
    - Jerarquía de precios bid y ask
    - Spread positivo
    - Tamaños positivos
    """
    cond = (
        (df["bid1_px"] > df["bid2_px"])
        & (df["bid2_px"] > df["bid3_px"])
        & (df["ask1_px"] < df["ask2_px"])
        & (df["ask2_px"] < df["ask3_px"])
        & (df["bid3_px"] < df["ask1_px"])
        & (df[[c for c in df.columns if c.endswith("_sz")]] > 0).all(axis=1)
    )
    df["flag_ob_bad_structure"] = ~cond
    return df


def detect_ob_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Señala snapshots con spreads o mid-price anómalos (MAD diario).
    """
    df = df.copy()
    df["mid_px"] = (df["bid1_px"] + df["ask1_px"]) / 2
    df["spread"] = (df["ask1_px"] - df["bid1_px"]) / df["mid_px"]

    def flag_day(sub):
        df = sub.copy() if False else sub  # keep original references
        series_spread = sub.spread.dropna()
        if not series_spread.empty:
            med_spread = series_spread.median()
            mad_spread = median_abs_deviation(series_spread, scale="normal")
        else:
            med_spread, mad_spread = 0, 1e-9
        series_mid = sub.mid_px.dropna()
        if not series_mid.empty:
            med_mid = series_mid.median()
            mad_mid = median_abs_deviation(series_mid, scale="normal")
        else:
            med_mid, mad_mid = 0, 1e-9
        sub["flag_spread_mad"] = ~sub.spread.between(med_spread - 5 * mad_spread, med_spread + 5 * mad_spread)
        sub["flag_mid_mad"] = ~sub.mid_px.between(med_mid - 5 * mad_mid, med_mid + 5 * mad_mid)
        return sub

    df = df.groupby(df.ts.dt.date, group_keys=False).apply(flag_day)
    return df


def reindex_ob(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Reindexa el orderbook a una rejilla regular (p.ej. 5 segundos),
    manteniendo la última snapshot conocida (forward fill).
    """
    idx = pd.date_range(start=df.ts.min(), end=df.ts.max(), freq=freq)
    out = df.set_index("ts").reindex(idx, method="ffill")
    out = out.rename_axis("ts").reset_index()
    return out


def aggregate_ob_to_candle(
    df_ob: pd.DataFrame, freq: str
) -> pd.DataFrame:
    """
    Convierte snapshots del orderbook en velas OHLC basadas en el mid‑price.
    """
    df = df_ob.copy()
    df["mid_px"] = (df["bid1_px"] + df["ask1_px"]) / 2
    ohlc = (
        df.set_index("ts")["mid_px"]
        .resample(freq)
        .ohlc()
        .reset_index()
        .rename(
            columns={
                "ts": "time_period_start",
                "open": "ob_open",
                "high": "ob_high",
                "low": "ob_low",
                "close": "ob_close",
            }
        )
    )
    return ohlc


def fill_ohlcv_from_ob(
    df_ohlcv: pd.DataFrame, df_ob_candle: pd.DataFrame
) -> pd.DataFrame:
    """
    Rellena valores NaN en OHLCV usando velas derivadas del orderbook.
    """
    df = df_ohlcv.merge(df_ob_candle, on="time_period_start", how="left")
    mapping = {
        "price_open": "ob_open",
        "price_high": "ob_high",
        "price_low": "ob_low",
        "price_close": "ob_close",
    }
    for target, source in mapping.items():
        df[target] = df[target].fillna(df[source])
    return df.drop(columns=list(mapping.values()))


def impute_ob(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina snapshots con estructura inválida o flags de outliers
    y asegura serie completa con forward-fill limitado.
    """
    price_cols = [c for c in df.columns if c.endswith("_px")]
    size_cols = [c for c in df.columns if c.endswith("_sz")]

    # Descartar bad structure y outliers fuertes
    bad_mask = df["flag_ob_bad_structure"] | df["flag_spread_mad"] | df["flag_mid_mad"]
    df.loc[bad_mask, price_cols + size_cols] = np.nan

    # Interpolación lineal para precios, ffill para tamaños
    df = df.set_index("ts")
    df[price_cols] = df[price_cols].interpolate(method="time", limit_area="inside")
    df[size_cols] = df[size_cols].ffill()

    # Relleno extremo
    df[price_cols] = df[price_cols].ffill().bfill()
    df[size_cols] = df[size_cols].fillna(0)

    df = df.reset_index()
    return df


# ------------------------------------------------------ #


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de limpieza de datos OHLCV y orderbook"
    )
    parser.add_argument(
        "--symbol",
        "-s",
        nargs="+",
        help="Símbolos a limpiar; por defecto limpia todos en la tabla coinapi_ohlcv",
    )
    parser.add_argument(
        "--db-file", "-d",
        default=DEFAULT_DB_FILE,
        help="Ruta a la base de datos SQLite (por defecto: trading_data.db en BASE_DIR)",
    )
    parser.add_argument(
        "--freq-candle",
        default=DEFAULT_FREQ_CANDLE,
        help="Resolución de velas OHLCV (por defecto: 5min)",
    )
    parser.add_argument(
        "--freq-orderbook",
        default=DEFAULT_FREQ_ORDERBOOK,
        help="Resolución objetivo para snapshots de orderbook (por defecto: 5S)",
    )
    parser.add_argument(
        "--max-gap-candles",
        type=int,
        default=DEFAULT_MAX_GAP_CANDLES,
        help="Máximo número de velas consecutivas a interpolar (por defecto: 2)",
    )
    parser.add_argument(
        "--skip-ohlcv",
        action="store_true",
        help="Omitir limpieza de tablas OHLCV",
    )
    parser.add_argument(
        "--skip-orderbook",
        action="store_true",
        help="Omitir limpieza de orderbook",
    )
    args = parser.parse_args()

    db_path = args.db_file
    freq_candle = args.freq_candle
    freq_orderbook = args.freq_orderbook
    max_gap_candles = args.max_gap_candles

    # Determinar símbolos a procesar
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if args.symbol:
        symbols = args.symbol
    else:
        cur.execute("SELECT DISTINCT symbol FROM coinapi_ohlcv;")
        symbols = [r[0] for r in cur.fetchall()]
    conn.close()
    total_symbols = len(symbols)
    inserted_ohlcv = 0
    skipped_ohlcv = 0

    for symbol in symbols:
        print(f"\n=== Procesando símbolo: {symbol} ===")
        print("  [OB] Iniciando limpieza de orderbook...")
        # ---------- Limpieza de ORDERBOOK para validación cruzada ----------
        if not args.skip_orderbook:
            df_ob = load_orderbook(db_path, symbol)
            df_ob = detect_ob_structure(df_ob)
            df_ob = detect_ob_outliers(df_ob)
            df_ob = reindex_ob(df_ob, freq=freq_orderbook)
            df_ob = impute_ob(df_ob)
            ob_data_check(df_ob)

            if not df_ob[["bid1_px", "ask1_px"]].isna().any().any():
                # Limpiar columnas temporales antes de persistir
                df_ob_persist = df_ob.drop(columns=["mid_px", "spread"], errors="ignore")
                conn_ob = sqlite3.connect(db_path)
                cur_ob = conn_ob.cursor()
                sql = (
                    "INSERT OR REPLACE INTO coinapi_orderbook_clean "
                    "(symbol_id, ts, date, bid1_px, bid1_sz, bid2_px, bid2_sz, "
                    "bid3_px, bid3_sz, ask1_px, ask1_sz, ask2_px, ask2_sz, "
                    "ask3_px, ask3_sz, flag_ob_bad_structure, flag_spread_mad, flag_mid_mad) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
                )
                batch_size = 200000
                batch = []
                count = 0
                for row in df_ob_persist.itertuples(index=False):
                    batch.append(
                        (
                            row.symbol_id,
                            row.ts.isoformat(),
                            row.date.isoformat(),
                            row.bid1_px,
                            row.bid1_sz,
                            row.bid2_px,
                            row.bid2_sz,
                            row.bid3_px,
                            row.bid3_sz,
                            row.ask1_px,
                            row.ask1_sz,
                            row.ask2_px,
                            row.ask2_sz,
                            row.ask3_px,
                            row.ask3_sz,
                            int(row.flag_ob_bad_structure),
                            int(row.flag_spread_mad),
                            int(row.flag_mid_mad),
                        )
                    )
                    if len(batch) >= batch_size:
                        cur_ob.executemany(sql, batch)
                        conn_ob.commit()
                        count += len(batch)
                        batch = []
                if batch:
                    cur_ob.executemany(sql, batch)
                    conn_ob.commit()
                    count += len(batch)
                    print(f"  [OB] coinapi_orderbook_clean inserted: {count} rows for {symbol}")
                conn_ob.close()
            else:
                print(f"[OB] Quedan NaNs en orderbook para {symbol}, se omite inserción.")

            df_ob_candle = aggregate_ob_to_candle(df_ob, freq=freq_candle)
            # --- Proxy de volumen y trades desde orderbook ---
            df_ob["bid_qty"] = df_ob[["bid1_sz", "bid2_sz", "bid3_sz"]].sum(axis=1)
            df_ob["ask_qty"] = df_ob[["ask1_sz", "ask2_sz", "ask3_sz"]].sum(axis=1)
            df_ob["delta_qty"] = (
                df_ob["bid_qty"].diff().abs() + df_ob["ask_qty"].diff().abs()
            )
            vol_proxy = (
                df_ob.set_index("ts")["delta_qty"]
                .resample(freq_candle)
                .sum()
                .reset_index()
                .rename(columns={"ts": "time_period_start", "delta_qty": "volume_proxy"})
            )
            trades_proxy = (
                df_ob.set_index("ts")["mid_px"]
                .resample(freq_candle)
                .count()
                .reset_index()
                .rename(columns={"ts": "time_period_start", "mid_px": "trades_proxy"})
            )
        else:
            df_ob_candle = None
            vol_proxy = None
            trades_proxy = None

        print("  [OHLCV] Iniciando limpieza de OHLCV...")
        if not args.skip_ohlcv:
            df = load_data(db_path, symbol)
            try:
                df = validate_schema(df)
            except pa.errors.SchemaErrors as err:
                print(f"ERROR de validación de esquema para {symbol}:")
                print(err.failure_cases)
                print("---- Saltando símbolo debido a errores de esquema ----")
                continue

            # Eliminar duplicados y ordenar por timestamp
            df = df.drop_duplicates(subset=["time_period_start"])
            df = df.sort_values("time_period_start").reset_index(drop=True)

            # Eliminar filas originales sin datos OHLCV antes de resample
            before_drop = len(df)
            df = df.dropna(subset=["price_open", "price_high", "price_low", "price_close"] )
            dropped_pre = before_drop - len(df)
            if dropped_pre > 0:
                print(f"  [OHLCV] Dropped {dropped_pre} rows without OHLCV before resample")
            # Rellenar rejilla de 5 minutos y contar gaps
            df = reindex_time(df, freq_candle)
            # Validación cruzada con el orderbook agregado
            if df_ob_candle is not None:
                df = fill_ohlcv_from_ob(df, df_ob_candle)
            # Estadísticas de missing tras OB fill
            missing_after_fill = df['price_open'].isna().sum()
            print(f"  [OHLCV] Missing after OB fill: {missing_after_fill} rows")
            # Rellenar volumen y trades desde proxy del orderbook
            if vol_proxy is not None:
                df = df.merge(vol_proxy, on="time_period_start", how="left")
                df["volume_traded"] = df["volume_traded"].fillna(df["volume_proxy"])
                df = df.drop(columns=["volume_proxy"])
            if trades_proxy is not None:
                df = df.merge(trades_proxy, on="time_period_start", how="left")
                df["trades_count"] = df["trades_count"].fillna(df["trades_proxy"])
                df = df.drop(columns=["trades_proxy"])
            gap_count = len(df) - df["price_open"].notna().sum()
            print(f"  [OHLCV] Resampled rows: {len(df)}, gaps filled: {gap_count}")

            # Primer chequeo de estructura (solo marca, NO filtra)
            struct_err_before = detect_structure(df)
            if struct_err_before > 0:
                print(f"  [OHLCV] Structure flags before imputation: {struct_err_before} rows")

            if df.empty:
                print(f"No quedan datos válidos para {symbol}, se omite inserción.")
                skipped_ohlcv += 1
                continue

            # Outliers e imputación
            df = detect_outliers(df)
            out_summary = {
                k: df[col].sum()
                for k, col in zip(
                    ["flag_outlier_mad", "flag_jump"], ["flag_outlier_mad", "flag_jump"]
                )
            }
            print(f"  [OHLCV] Outliers MAD: {out_summary['flag_outlier_mad']}, jumps: {out_summary['flag_jump']}")
            df = impute_data(df, max_gap_candles)
            # Re‑evaluar estructura tras la imputación; ahora sí eliminamos
            struct_err_after = detect_structure(df)
            if struct_err_after > 0:
                print(f"  [OHLCV] Structure flags after imputation (rows dropped): {struct_err_after}")
                df = df[~df.flag_bad_structure]
            print(f"  [OHLCV] Imputation done (max gap {max_gap_candles})")
            # Estadísticas tras imputación
            missing_after_imp = df['price_open'].isna().sum()
            if missing_after_imp > 0:
                df = df[df['price_open'].notna()]
                print(f"  [OHLCV] Dropped {missing_after_imp} rows after imputation")

            # Generar salida final
            df_final = reduce_columns(df)
            df_final = df_final.dropna()
            # final_data_check(df_final)

            if df_final.empty:
                print(f"No quedan datos válidos para {symbol}, se omite inserción OHLCV.")
                skipped_ohlcv += 1
                continue

            # Persistir por símbolo
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            rows = [
                (
                    row["symbol"],
                    row["time_period_start"].isoformat(),
                    row["time_period_end"].isoformat(),
                    row["time_open"].isoformat(),
                    row["time_close"].isoformat(),
                    float(row["price_open"]),
                    float(row["price_high"]),
                    float(row["price_low"]),
                    float(row["price_close"]),
                    float(row["volume_traded"]),
                    int(row["trades_count"]),
                    int(row["flag_bad_structure"]),
                    int(row["flag_outlier_mad"]),
                    int(row["flag_jump"]),
                )
                for _, row in df_final.iterrows()
            ]
            cur.executemany(
                "INSERT OR REPLACE INTO coinapi_ohlcv_clean "
                "(symbol, time_period_start, time_period_end, time_open, time_close, "
                "price_open, price_high, price_low, price_close, volume_traded, trades_count, "
                "flag_bad_structure, flag_outlier_mad, flag_jump) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                rows,
            )
            conn.commit()
            print(f"  [OHLCV] coinapi_ohlcv_clean inserted: {len(rows)} rows for {symbol}")
            inserted_ohlcv += 1
            conn.close()


    print(
        f"Resumen: símbolos procesados: {total_symbols}, "
        f"OHLCV insertados: {inserted_ohlcv}, omitidos: {skipped_ohlcv}"
    )


def ob_data_check(df: pd.DataFrame) -> None:
    rows, _ = df.shape
    print(f"[OB] Snapshots: {rows}")
    bad = df[["flag_ob_bad_structure", "flag_spread_mad", "flag_mid_mad"]].sum()
    print(
        f"[OB] Structure flags: {bad['flag_ob_bad_structure']}, "
        f"spread outliers: {bad['flag_spread_mad']}, "
        f"mid outliers: {bad['flag_mid_mad']}"
    )
    null_pct = df.isna().mean().mul(100).round(2)
    if null_pct.any():
        print("[OB] Nulls (%):")
        print(null_pct[null_pct > 0])


if __name__ == "__main__":
    main()
