# Guía de Umbrales Realistas para Validación de Pares en Criptomonedas

Este documento tiene como objetivo proporcionar una perspectiva sobre valores de umbral razonables y consideraciones específicas para el mercado de criptomonedas, que te ayudarán a refinar los criterios de selección de pares en tu script de validación. Es crucial entender que estos no son números mágicos, sino guías que deben ajustarse en función del *backtesting* y las características específicas de los activos que estés analizando (especialmente si son *memecoins*).

## Consideraciones Generales para Cripto

* **Alta Volatilidad:** Las criptomonedas son inherentemente más volátiles que los activos tradicionales. Esto puede afectar la estabilidad de las correlaciones y los parámetros de cointegración.
* **Mercados Jóvenes y Dinámicos:** Las relaciones entre activos pueden cambiar más rápidamente. Un par que hoy es bueno, mañana podría no serlo.
* **Eventos de Mercado:** Las noticias, *hypes* y caídas abruptas pueden distorsionar temporalmente las métricas.
* **Liquidez Variable:** Especialmente en *altcoins* o *memecoins*, la liquidez puede ser un factor limitante y afectar la calidad de los datos de precios.
* **Naturaleza de las Memecoins:** A menudo impulsadas por sentimiento y narrativas más que por fundamentales, sus relaciones pueden ser menos estables a largo plazo que pares de criptomonedas más establecidas.

## Discusión Detallada de Métricas y Umbrales

A continuación, se revisan las métricas de tu "Fase 1" y se proponen rangos o enfoques más adaptados a cripto:

### 1. Correlación ($\rho$) de Log-Precios

* **Métrica Original (`static_corr` > 0.80, `rolling_corr_value` > 0.80, `rolling_corr_pct_pass` >= 0.75)**
* **Propósito:** Asegurar que los precios de los activos se mueven generalmente en la misma dirección.
* **Consideraciones Cripto:**
    * Una correlación estática de >0.80 es un buen punto de partida, pero en cripto, incluso >0.70 o >0.75 podría ser aceptable si otras métricas (como la cointegración) son fuertes.
    * La `rolling_corr_value` (valor en cada ventana) podría ser más flexible, quizás >0.65 o >0.70, pero el `rolling_corr_pct_pass` (porcentaje de ventanas que cumplen) se vuelve más importante. Si la correlación es consistentemente decente, aunque no altísima, puede ser viable.
* **Sugerencias Flexibles:**
    * `static_corr`: **> 0.70 - 0.75** (Ideal: >0.80)
    * `rolling_corr_value`: **> 0.65 - 0.70** (Ideal: >0.75)
    * `rolling_corr_pct_pass`: **>= 0.60 - 0.70** (Ideal: >=0.75). Si es más bajo, se necesita cointegración muy robusta.
* **Razón:** Las criptos pueden tener períodos de desacoplamiento temporal debido a noticias específicas de un proyecto o flujos de capital especulativos. La consistencia (pct_pass) sobre un valor decente es clave.

### 2. Cointegración (Test de Engle-Granger)

* **Métrica Original (`static_coint_pvalue` < 0.05, `rolling_coint_pvalue` < 0.05, `rolling_coint_pct_pass` >= 0.80)**
* **Propósito:** Esencial. Indica que existe una relación de equilibrio a largo plazo entre los precios, y el spread (residuos de la regresión) es estacionario.
* **Consideraciones Cripto:**
    * El p-value < 0.05 es el estándar académico y es deseable mantenerlo. Sin embargo, en mercados más ruidosos, un p-value < 0.10 podría considerarse para exploración si va acompañado de un Half-Life razonable y un Hurst bajo.
    * El `rolling_coint_pct_pass` es crucial. Una relación que se rompe frecuentemente no es fiable. Quizás un 70-75% sea más realista que un 80% para algunos pares cripto, pero por debajo de eso, el riesgo aumenta considerablemente.
* **Sugerencias Flexibles:**
    * `static_coint_pvalue`: **< 0.05** (Exploratorio con cautela: < 0.10)
    * `rolling_coint_pvalue`: **< 0.05** (Exploratorio con cautela: < 0.10)
    * `rolling_coint_pct_pass`: **>= 0.70 - 0.75** (Ideal: >=0.80)
* **Razón:** La cointegración es la base del *pairs trading*. Ser demasiado laxo aquí es peligroso. La persistencia de la cointegración (pct_pass) es vital en mercados cambiantes.

### 3. Test de Raíz Unitaria sobre el Spread (ADF)

* **Métrica Original (`static_adf_pvalue` < 0.05)**
* **Propósito:** Confirmar la estacionariedad del spread, lo que implica que tiende a revertir a su media.
* **Consideraciones Cripto:** Similar a la cointegración. El p-value < 0.05 es el objetivo.
* **Sugerencias Flexibles:**
    * `static_adf_pvalue`: **< 0.05** (Exploratorio con cautela: < 0.10)
* **Razón:** Un spread no estacionario significa que las desviaciones de la media pueden ser permanentes, invalidando la estrategia.

### 4. Exponente de Hurst (H) del Spread

* **Métrica Original (0.20 <= H <= 0.40)**
* **Propósito:** Medir la "memoria" o tendencia de reversión a la media (H < 0.5).
* **Consideraciones Cripto:**
    * El rango 0.20-0.40 es bueno e indica una reversión a la media decente.
    * Valores muy bajos (ej. < 0.15-0.20) pueden indicar un spread demasiado ruidoso y difícil de operar, aunque revierta.
    * Valores cercanos a 0.5 (ej. 0.40-0.49) indican una reversión más débil.
* **Sugerencias Flexibles:**
    * `hurst_min`: **0.15 - 0.20**
    * `hurst_max`: **0.45 - 0.48** (Ideal: <0.40)
* **Razón:** Se busca un equilibrio. Demasiada reversión (H muy bajo) puede ser ruido, poca reversión (H cercano a 0.5) debilita la premisa.

### 5. Half-Life (HL) de Reversión del Spread

* **Métrica Original (5 <= HL <= 50 velas de 5 min)**
* **Propósito:** Estimar cuánto tarda el spread en volver a la mitad de su desviación.
* **Consideraciones Cripto:**
    * Para velas de 5 minutos:
        * HL < 5 velas (25 min): Puede ser demasiado rápido, indicando ruido o un spread muy volátil que es difícil de capturar con costes.
        * HL > 50 velas (4h 10min): Puede ser demasiado lento para una estrategia intradía o de corto plazo. El capital estaría inmovilizado mucho tiempo, y el riesgo de que la relación cambie antes de la reversión aumenta.
    * El rango óptimo depende de tu horizonte de trading y costes.
* **Sugerencias Flexibles:**
    * `half_life_min_candles`: **3 - 6** (Ideal: 5)
    * `half_life_max_candles`: **60 - 100** (Ideal: 50). Para *memecoins*, quizás preferir rangos más cortos (ej. máx 60-72 velas, que son 5-6 horas) debido a su naturaleza más efímera.
* **Razón:** Se busca una reversión que sea lo suficientemente rápida para ser operable dentro de un marco de tiempo razonable, pero no tan rápida que sea indistinguible del ruido del mercado y los costes de transacción.

### 6. Volatilidad del Spread ($\sigma_{spread}$)

* **Métrica Original (Volatilidad media > 0.5 * mediana($\sigma_{spread}$) de todos los pares candidatos)**
* **Propósito:** Evitar spreads casi planos que no ofrecen oportunidades de trading.
* **Consideraciones Cripto:** Este es un filtro relativo y sigue siendo útil. Asegura que el spread tiene "movimiento". No hay un valor absoluto universal.
* **Sugerencias Flexibles:** El enfoque relativo es bueno. Podrías considerar un umbral absoluto mínimo muy pequeño (ej. que la $\sigma_{spread}$ promedio sea al menos 0.05% o 0.1% del precio promedio de los activos) como un sanity check adicional si el filtro relativo no descarta suficientes pares.
* **Razón:** Un spread con volatilidad extremadamente baja, incluso si es estacionario, no generará señales de trading con suficiente amplitud para superar los costes.

### 7. Liquidez (Volumen y Profundidad)

* **Métrica Original (Volumen diario spot > 30,000 USDT por cada perpetuo, Profundidad libro >= 0.2 tamaño orden)**
* **Propósito:** Asegurar que se pueden ejecutar órdenes sin un *slippage* excesivo.
* **Consideraciones Cripto:**
    * **Volumen:** Para *memecoins*, 30k USDT diarios en el *spot subyacente* puede ser un umbral de entrada. Para *altcoins* más establecidas, se buscaría mucho más (ej. >200k-500k USDT). Para *majors* (BTC, ETH), millones. Es muy dependiente del activo.
    * **Profundidad:** El criterio de "0.2 del tamaño máximo de orden" es bueno. Si operas con nocionales pequeños (ej. 50-100 USDT por pata para un capital de 1000 USDT), la profundidad requerida será menor. Para *memecoins*, la profundidad en los primeros niveles puede ser escasa.
* **Sugerencias Flexibles:**
    * `spot_volume_min_avg_daily_usdt`:
        * Memecoins / Micro-caps: **15,000 - 50,000 USDT** (con mucha cautela si es bajo)
        * Altcoins Pequeñas/Medianas: **50,000 - 250,000 USDT**
        * Altcoins Grandes: **> 250,000 - 1,000,000 USDT**
    * `avg_depth_sum_sz_perp` (del script, que usa `mark_price_vwap`): En lugar de un umbral fijo, verifica que sea consistentemente positivo y no trivial. La comparación con el tamaño de orden es lo más relevante. Si tu orden típica es de 50 USDT, y el `avg_depth_sum_sz_perp` (que es suma de tamaños, no nocional) es, por ejemplo, 1000 unidades del activo, y el precio es 0.01 USDT/unidad, entonces hay 10 USDT en los 3 niveles. Esto es bajo. Necesitas convertir `depth_sum_sz` a nocional (multiplicar por precio promedio de esos niveles) para una mejor comparación. *El script actual solo comprueba que sea > 0, lo cual es demasiado básico.*
        * **Recomendación para profundidad:** Modificar el script para estimar el nocional en los 3 niveles y luego compararlo con el tamaño de tu orden. O, como proxy, asegurar que `avg_depth_sum_sz_perp * precio_medio_del_activo` sea significativamente mayor que tu tamaño de orden.
* **Razón:** La liquidez es crítica. Sin ella, los costes de *slippage* destruirán cualquier ventaja teórica. Para *memecoins*, este es a menudo el talón de Aquiles.

### 8. Distribución de Retornos del Spread

* **Métrica Original (Skewness: -1 a +1, Kurtosis < 5)**
* **Propósito:** Evaluar si la distribución del spread es problemática (colas muy pesadas, muy asimétrica).
* **Consideraciones Cripto:**
    * Los retornos en cripto raramente son normales. Es esperable mayor kurtosis (colas más gordas).
    * Una skewness muy pronunciada puede indicar un sesgo persistente que la estrategia podría no capturar bien.
* **Sugerencias Flexibles:**
    * `spread_return_skew_min`: **-1.5 a -2.0** (Ideal: > -1.0)
    * `spread_return_skew_max`: **+1.5 a +2.0** (Ideal: < +1.0)
    * `spread_return_kurtosis_max`: **7 - 10** (Ideal: < 5-6). Valores por encima de 10 indican colas extremadamente pesadas y riesgo de movimientos muy bruscos.
* **Razón:** Aunque se espera no-normalidad, valores extremos de skewness o kurtosis pueden indicar riesgos ocultos o que el modelo de Z-Score (que asume cierta simetría y normalidad en sus umbrales) no sea el más adecuado sin ajustes.

## Enfoque Flexible y Próximos Pasos

1.  **No seas demasiado rígido:** En lugar de un "pasa/no pasa" estricto para cada métrica, considera un sistema de puntuación o categorías (ej. "Óptimo", "Aceptable", "Precaución", "Rechazar"). Un par podría fallar ligeramente en una métrica pero ser excelente en otras.
2.  **Prioriza:** Cointegración y estacionariedad del spread (ADF) son las más críticas. Sin esto, no hay base para la estrategia.
3.  **El contexto importa:** ¿Son dos *memecoins* nuevas o dos *altcoins* con más historial? Ajusta tus expectativas.
4.  **Backtesting es el juez final:** Estos filtros son para encontrar candidatos prometedores. El rendimiento en un *backtest* robusto (con costes realistas) determinará la viabilidad final.
5.  **Itera:** Si ningún par pasa, relaja ligeramente los umbrales más restrictivos (pero no los fundamentales como cointegración) y vuelve a analizar. Observa qué métricas son las que más fallan tus pares.

Al ajustar el diccionario `THRESHOLDS` en tu script, puedes empezar con los valores "Ideales" o ligeramente más relajados de las sugerencias y ver qué pares emergen. Luego, puedes investigar individualmente los que están "cerca" de pasar.