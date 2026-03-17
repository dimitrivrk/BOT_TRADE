"""
Collecteur de donnees on-chain et sentiment pour crypto.

Features collectees :
  - Funding rate (Binance Futures)
  - Open Interest (Binance Futures)
  - Long/Short ratio (Binance Futures)
  - Fear & Greed Index (alternative.me API)
  - Liquidation volume estimee

Ces features sont des signaux tres puissants car elles mesurent le
positionnement et le sentiment du marche, pas juste le prix.

Reference: TFT multi-crypto (MDPI 2025) montre que l'ajout de on-chain
features ameliore significativement les predictions vs technical seul.
"""

import numpy as np
import pandas as pd
import requests
import time
from typing import Optional, Dict
from datetime import datetime, timezone

from utils.logger import setup_logger

logger = setup_logger("data.collectors.onchain")


class OnChainCollector:
    """
    Collecte les donnees on-chain et sentiment depuis les API publiques.
    Toutes les features sont normalisees pour etre utilisees directement.
    """

    def __init__(self, testnet: bool = True):
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
        # Fear & Greed est toujours en mainnet (API publique)
        self.fng_url = "https://api.alternative.me/fng/"
        self._cache = {}
        self._cache_ttl = 300  # 5 min cache

    def get_funding_rate(self, symbol: str = "BTCUSDT", limit: int = 100) -> pd.Series:
        """
        Recupere l'historique du funding rate.
        Funding rate > 0 = longs paient shorts (marche bullish, potentiel retournement)
        Funding rate < 0 = shorts paient longs (marche bearish, potentiel retournement)
        """
        try:
            url = f"{self.base_url}/fapi/v1/fundingRate"
            r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    df = pd.DataFrame(data)
                    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
                    df['fundingRate'] = df['fundingRate'].astype(float)
                    series = df.set_index('fundingTime')['fundingRate']
                    logger.debug(f"Funding rate: {len(series)} points, dernier={series.iloc[-1]:.6f}")
                    return series
        except Exception as e:
            logger.warning(f"Funding rate fetch failed: {e}")
        return pd.Series(dtype=float)

    def get_open_interest(self, symbol: str = "BTCUSDT", period: str = "1h", limit: int = 100) -> pd.DataFrame:
        """
        Recupere l'historique de l'open interest.
        OI en hausse + prix en hausse = tendance forte (argent frais entre)
        OI en hausse + prix en baisse = shorts accumulent
        OI en baisse + prix en hausse = short squeeze
        OI en baisse + prix en baisse = capitulation
        """
        try:
            url = f"{self.base_url}/futures/data/openInterestHist"
            r = requests.get(url, params={
                "symbol": symbol, "period": period, "limit": limit
            }, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
                    df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
                    df = df.set_index('timestamp')
                    logger.debug(f"Open Interest: {len(df)} points")
                    return df[['sumOpenInterest', 'sumOpenInterestValue']]
        except Exception as e:
            logger.warning(f"Open Interest fetch failed: {e}")
        return pd.DataFrame()

    def get_long_short_ratio(self, symbol: str = "BTCUSDT", period: str = "1h", limit: int = 100) -> pd.DataFrame:
        """
        Recupere le ratio long/short des top traders.
        Ratio > 1 = plus de longs que de shorts (foule bullish)
        Ratio >> 2 = extreme greed, potentiel retournement baissier
        Ratio < 1 = plus de shorts que de longs (foule bearish)
        """
        try:
            url = f"{self.base_url}/futures/data/topLongShortAccountRatio"
            r = requests.get(url, params={
                "symbol": symbol, "period": period, "limit": limit
            }, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['longShortRatio'] = df['longShortRatio'].astype(float)
                    df['longAccount'] = df['longAccount'].astype(float)
                    df['shortAccount'] = df['shortAccount'].astype(float)
                    df = df.set_index('timestamp')
                    logger.debug(f"Long/Short ratio: {len(df)} points")
                    return df
        except Exception as e:
            logger.warning(f"Long/Short ratio fetch failed: {e}")
        return pd.DataFrame()

    def get_fear_greed_index(self, limit: int = 30) -> pd.Series:
        """
        Recupere le Fear & Greed Index (0-100).
        0-25 = Extreme Fear (signal d'achat contrarian)
        25-50 = Fear
        50-75 = Greed
        75-100 = Extreme Greed (signal de vente contrarian)

        C'est un indicateur contrarian: acheter quand les gens ont peur,
        vendre quand les gens sont cupides.
        """
        cache_key = f"fng_{limit}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_data

        try:
            r = requests.get(self.fng_url, params={"limit": limit, "format": "json"}, timeout=10)
            if r.status_code == 200:
                data = r.json().get('data', [])
                if data:
                    records = []
                    for d in data:
                        ts = pd.Timestamp(int(d['timestamp']), unit='s')
                        records.append({'timestamp': ts, 'fng_value': int(d['value'])})
                    df = pd.DataFrame(records).set_index('timestamp').sort_index()
                    series = df['fng_value']
                    self._cache[cache_key] = (time.time(), series)
                    logger.debug(f"Fear & Greed: {len(series)} points, dernier={series.iloc[-1]}")
                    return series
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
        return pd.Series(dtype=float)

    def get_taker_buy_sell_volume(self, symbol: str = "BTCUSDT", period: str = "1h", limit: int = 100) -> pd.DataFrame:
        """
        Ratio volume taker buy / sell.
        > 1 = acheteurs agressifs (bullish)
        < 1 = vendeurs agressifs (bearish)
        """
        try:
            url = f"{self.base_url}/futures/data/takerlongshortRatio"
            r = requests.get(url, params={
                "symbol": symbol, "period": period, "limit": limit
            }, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['buySellRatio'] = df['buySellRatio'].astype(float)
                    df['buyVol'] = df['buyVol'].astype(float)
                    df['sellVol'] = df['sellVol'].astype(float)
                    df = df.set_index('timestamp')
                    return df
        except Exception as e:
            logger.warning(f"Taker buy/sell fetch failed: {e}")
        return pd.DataFrame()

    def compute_onchain_features(
        self,
        ohlcv_df: pd.DataFrame,
        symbol: str = "BTCUSDT",
    ) -> pd.DataFrame:
        """
        Calcule toutes les features on-chain et les aligne sur l'index OHLCV.

        Returns:
            DataFrame avec les colonnes:
            - funding_rate: funding rate brut
            - funding_rate_ma: moyenne mobile 8 periodes
            - funding_rate_extreme: 1 si extreme (>0.01%), -1 si extreme (<-0.01%), 0 sinon
            - oi_change_pct: variation % de l'open interest
            - oi_price_divergence: OI monte + prix baisse = bearish divergence
            - ls_ratio: long/short ratio normalise
            - ls_ratio_extreme: 1 si extreme bullish (>2), -1 si extreme bearish (<0.5)
            - fng_value: Fear & Greed index normalise [0, 1]
            - fng_extreme: 1 si greed extreme, -1 si fear extreme
            - taker_ratio: buy/sell taker ratio
        """
        features = pd.DataFrame(index=ohlcv_df.index)
        sym = symbol.replace('/', '')

        def _normalize_index(s: pd.Series) -> pd.Series:
            """Normalise l'index d'une Series en UTC pour le reindex."""
            idx = s.index
            if hasattr(idx, 'tz') and idx.tz is None:
                idx = idx.tz_localize('UTC')
            else:
                idx = idx.tz_convert('UTC')
            # Convertir en datetime64[ms, UTC] pour correspondre à l'index OHLCV
            s.index = idx.astype('datetime64[ms, UTC]')
            return s

        # --- Funding Rate ---
        try:
            fr = self.get_funding_rate(sym, limit=100)
            if not fr.empty:
                fr = _normalize_index(fr)
                fr_reindexed = fr.reindex(ohlcv_df.index, method='ffill')
                features['funding_rate'] = fr_reindexed
                features['funding_rate_ma'] = features['funding_rate'].rolling(8, min_periods=1).mean()
                features['funding_rate_extreme'] = 0.0
                features.loc[features['funding_rate'] > 0.0001, 'funding_rate_extreme'] = 1.0
                features.loc[features['funding_rate'] < -0.0001, 'funding_rate_extreme'] = -1.0
        except Exception as e:
            logger.debug(f"Funding rate features skipped: {e}")

        # --- Open Interest ---
        try:
            oi = self.get_open_interest(sym, period="1h", limit=100)
            if not oi.empty:
                oi_reindexed = _normalize_index(oi['sumOpenInterestValue']).reindex(ohlcv_df.index, method='ffill')
                features['oi_change_pct'] = oi_reindexed.pct_change(1)
                # Divergence OI vs prix: OI monte + prix baisse = bearish signal
                price_change = ohlcv_df['close'].pct_change(1)
                oi_change = features['oi_change_pct']
                features['oi_price_divergence'] = np.sign(oi_change) * (-np.sign(price_change))
        except Exception as e:
            logger.debug(f"Open Interest features skipped: {e}")

        # --- Long/Short Ratio ---
        try:
            ls = self.get_long_short_ratio(sym, period="1h", limit=100)
            if not ls.empty:
                ls_reindexed = _normalize_index(ls['longShortRatio']).reindex(ohlcv_df.index, method='ffill')
                features['ls_ratio'] = (ls_reindexed - 1.0)  # centre sur 0
                features['ls_ratio_extreme'] = 0.0
                features.loc[ls_reindexed > 2.0, 'ls_ratio_extreme'] = 1.0    # trop de longs
                features.loc[ls_reindexed < 0.5, 'ls_ratio_extreme'] = -1.0    # trop de shorts
        except Exception as e:
            logger.debug(f"Long/Short ratio features skipped: {e}")

        # --- Fear & Greed Index ---
        try:
            fng = self.get_fear_greed_index(limit=30)
            if not fng.empty:
                fng_reindexed = _normalize_index(fng).reindex(ohlcv_df.index, method='ffill')
                features['fng_value'] = fng_reindexed / 100.0  # normaliser [0, 1]
                features['fng_extreme'] = 0.0
                features.loc[fng_reindexed > 75, 'fng_extreme'] = 1.0     # extreme greed
                features.loc[fng_reindexed < 25, 'fng_extreme'] = -1.0    # extreme fear
        except Exception as e:
            logger.debug(f"Fear & Greed features skipped: {e}")

        # --- Taker Buy/Sell Ratio ---
        try:
            taker = self.get_taker_buy_sell_volume(sym, period="1h", limit=100)
            if not taker.empty:
                taker_reindexed = _normalize_index(taker['buySellRatio']).reindex(ohlcv_df.index, method='ffill')
                features['taker_ratio'] = taker_reindexed - 1.0  # centre sur 0
        except Exception as e:
            logger.debug(f"Taker ratio features skipped: {e}")

        # Fill NaN
        features = features.fillna(0.0)

        n_features = sum(1 for c in features.columns if features[c].abs().sum() > 0)
        logger.info(f"On-chain features: {n_features}/{len(features.columns)} actives sur {len(features)} lignes")

        return features
