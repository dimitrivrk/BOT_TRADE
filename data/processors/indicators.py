"""
Bibliothèque d'indicateurs techniques.
Implémentés en pandas/numpy pur (sans dépendance externe pandas-ta).
Inclut des indicateurs avancés custom.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
# =============================================================================
# TENDANCE
# =============================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def vwma(df: pd.DataFrame, period: int) -> pd.Series:
    """Volume Weighted Moving Average."""
    return (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()


def supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    SuperTrend indicator.
    Retourne : supertrend (valeur), direction (1=bull, -1=bear)
    """
    atr_val = atr(df, atr_period)
    hl2 = (df['high'] + df['low']) / 2

    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    supertrend_vals = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    supertrend_vals.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        prev_close = df['close'].iloc[i - 1]
        curr_close = df['close'].iloc[i]

        # Calcul des bandes finales
        if upper_band.iloc[i] < supertrend_vals.iloc[i - 1] or prev_close > supertrend_vals.iloc[i - 1]:
            final_upper = upper_band.iloc[i]
        else:
            final_upper = supertrend_vals.iloc[i - 1]

        if lower_band.iloc[i] > supertrend_vals.iloc[i - 1] or prev_close < supertrend_vals.iloc[i - 1]:
            final_lower = lower_band.iloc[i]
        else:
            final_lower = supertrend_vals.iloc[i - 1]

        if direction.iloc[i - 1] == -1 and curr_close > final_upper:
            direction.iloc[i] = 1
            supertrend_vals.iloc[i] = final_lower
        elif direction.iloc[i - 1] == 1 and curr_close < final_lower:
            direction.iloc[i] = -1
            supertrend_vals.iloc[i] = final_upper
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            supertrend_vals.iloc[i] = final_lower if direction.iloc[i] == 1 else final_upper

    return pd.DataFrame({
        'supertrend': supertrend_vals,
        'supertrend_dir': direction,
    }, index=df.index)


def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP intrajournalier (reset chaque jour)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()


# =============================================================================
# MOMENTUM
# =============================================================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI classique Wilder."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD : line, signal, histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': histogram,
    }, index=series.index)


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth_k: int = 3) -> pd.DataFrame:
    """Stochastique %K et %D."""
    lowest_low = df['low'].rolling(k).min()
    highest_high = df['high'].rolling(k).max()
    k_pct = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
    k_smooth = k_pct.rolling(smooth_k).mean()
    d_pct = k_smooth.rolling(d).mean()
    return pd.DataFrame({'stoch_k': k_smooth, 'stoch_d': d_pct}, index=df.index)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    mfr = pos_mf / (neg_mf + 1e-10)
    return 100 - (100 / (1 + mfr))


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R."""
    highest_high = df['high'].rolling(period).max()
    lowest_low = df['low'].rolling(period).min()
    return -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    mean_tp = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - mean_tp) / (0.015 * mean_dev + 1e-10)


# =============================================================================
# VOLATILITÉ
# =============================================================================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std: float = 2.0,
) -> pd.DataFrame:
    """Bandes de Bollinger + %B + Bandwidth."""
    mid = sma(series, period)
    std_val = series.rolling(period).std()
    upper = mid + std * std_val
    lower = mid - std * std_val
    pct_b = (series - lower) / (upper - lower + 1e-10)
    bandwidth = (upper - lower) / (mid + 1e-10)
    return pd.DataFrame({
        'bb_upper': upper,
        'bb_mid': mid,
        'bb_lower': lower,
        'bb_pct_b': pct_b,
        'bb_width': bandwidth,
    }, index=series.index)


def keltner_channels(
    df: pd.DataFrame,
    period: int = 20,
    atr_mult: float = 1.5,
) -> pd.DataFrame:
    """Keltner Channels."""
    mid = ema(df['close'], period)
    atr_val = atr(df, period)
    upper = mid + atr_mult * atr_val
    lower = mid - atr_mult * atr_val
    return pd.DataFrame({
        'kc_upper': upper,
        'kc_mid': mid,
        'kc_lower': lower,
    }, index=df.index)


def historical_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Volatilité historique annualisée (log-returns std)."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252 * 24)  # annualisé (bougies horaires)


# =============================================================================
# VOLUME
# =============================================================================

def obv(df: pd.DataFrame) -> pd.Series:
    """On Balance Volume."""
    direction = np.where(df['close'] > df['close'].shift(1), 1,
                         np.where(df['close'] < df['close'].shift(1), -1, 0))
    return pd.Series((direction * df['volume']).cumsum(), index=df.index)


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    mfv = clv * df['volume']
    return mfv.rolling(period).sum() / df['volume'].rolling(period).sum()


def volume_profile(df: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
    """Volume Profile : distribution du volume par niveau de prix."""
    price_range = np.linspace(df['low'].min(), df['high'].max(), bins + 1)
    profile = pd.DataFrame({'price': price_range[:-1], 'volume': 0.0})
    for _, row in df.iterrows():
        mask = (profile['price'] >= row['low']) & (profile['price'] <= row['high'])
        n_bins = mask.sum()
        if n_bins > 0:
            profile.loc[mask, 'volume'] += row['volume'] / n_bins
    return profile


# =============================================================================
# INDICATEURS AVANCÉS / CUSTOM
# =============================================================================

def squeeze_momentum(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
                     kc_period: int = 20, kc_mult: float = 1.5) -> pd.DataFrame:
    """
    Squeeze Momentum (LazyBear) :
    Détecte les phases de faible volatilité (squeeze) avant les explosions.
    """
    bb = bollinger_bands(df['close'], bb_period, bb_std)
    kc = keltner_channels(df, kc_period, kc_mult)

    # Squeeze = BB à l'intérieur des KC
    sq_on = (bb['bb_lower'] > kc['kc_lower']) & (bb['bb_upper'] < kc['kc_upper'])
    sq_off = ~sq_on

    # Momentum = régression linéaire sur delta de prix
    highest_high = df['high'].rolling(kc_period).max()
    lowest_low = df['low'].rolling(kc_period).min()
    mid = (highest_high + lowest_low) / 2
    delta = df['close'] - (mid + sma(df['close'], kc_period)) / 2
    momentum = delta.rolling(kc_period).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
    )

    return pd.DataFrame({
        'sq_momentum': momentum,
        'sq_on': sq_on.astype(int),
        'sq_off': sq_off.astype(int),
    }, index=df.index)


def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 20) -> pd.Series:
    """
    Exposant de Hurst (rolling) :
    H > 0.5 = tendance (mean-flight), H < 0.5 = mean-reversion, H ≈ 0.5 = random walk.
    Utile pour adapter la stratégie au régime de marché.
    """
    def _hurst(ts):
        lags = range(min_lag, max_lag)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        if len(tau) < 2 or np.any(np.array(tau) <= 0):
            return 0.5
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]

    return series.rolling(100).apply(_hurst, raw=True)


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX) + DI+/DI- en numpy/pandas pur."""
    high, low, close = df['high'], df['low'], df['close']
    # True Range
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    # Wilder smoothing
    def wilder_smooth(s, n):
        result = s.copy().astype(float)
        result.iloc[:n] = np.nan
        result.iloc[n] = s.iloc[1:n+1].sum()
        for i in range(n + 1, len(s)):
            result.iloc[i] = result.iloc[i - 1] - result.iloc[i - 1] / n + s.iloc[i]
        return result
    atr_w = wilder_smooth(tr, period)
    plus_dm_w = wilder_smooth(plus_dm, period)
    minus_dm_w = wilder_smooth(minus_dm, period)
    plus_di = 100 * plus_dm_w / (atr_w + 1e-10)
    minus_di = 100 * minus_dm_w / (atr_w + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx_vals = wilder_smooth(dx.fillna(0), period)
    return pd.DataFrame({'adx': adx_vals, 'dmp': plus_di, 'dmn': minus_di}, index=df.index)


def market_regime(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Détection du régime de marché basée sur ADX + Hurst.
    Returns : 'trending_bull' | 'trending_bear' | 'ranging' | 'volatile'
    """
    adx_df = adx(df, period)
    adx_vals = adx_df['adx']
    dmp = adx_df['dmp']
    dmn = adx_df['dmn']
    hv = historical_volatility(df['close'], period)
    hv_norm = (hv - hv.rolling(100).min()) / (hv.rolling(100).max() - hv.rolling(100).min() + 1e-10)

    regimes = []
    for i in range(len(df)):
        adx_v = adx_vals.iloc[i] if not pd.isna(adx_vals.iloc[i]) else 0
        hv_v = hv_norm.iloc[i] if not pd.isna(hv_norm.iloc[i]) else 0.5

        if hv_v > 0.8:
            regime = 'volatile'
        elif adx_v > 25:
            regime = 'trending_bull' if dmp.iloc[i] > dmn.iloc[i] else 'trending_bear'
        else:
            regime = 'ranging'
        regimes.append(regime)

    return pd.Series(regimes, index=df.index, name='market_regime')


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot Points classiques (support/résistance)."""
    pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    r1 = 2 * pivot - df['low'].shift(1)
    s1 = 2 * pivot - df['high'].shift(1)
    r2 = pivot + (df['high'].shift(1) - df['low'].shift(1))
    s2 = pivot - (df['high'].shift(1) - df['low'].shift(1))
    r3 = df['high'].shift(1) + 2 * (pivot - df['low'].shift(1))
    s3 = df['low'].shift(1) - 2 * (df['high'].shift(1) - pivot)
    return pd.DataFrame({'pivot': pivot, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2, 'r3': r3, 's3': s3}, index=df.index)
