"""
Feature Engineering pipeline.
Transforme les OHLCV bruts en features normalisées prêtes pour les modèles.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import yaml

from data.processors import indicators as ind
from utils.logger import setup_logger

logger = setup_logger("data.processors.features")


class FeatureEngineer:
    """
    Pipeline de feature engineering complet.
    Calcule ~80 features : indicateurs, patterns, features temporelles, interactions.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.feat_config = self.config["features"]
        self.lookbacks = self.feat_config.get("lookback_windows", [5, 10, 20, 50])
        self.norm_method = self.feat_config.get("normalization", "rolling_zscore")

    def compute_all(
        self,
        df: pd.DataFrame,
        higher_tf_df: Optional[pd.DataFrame] = None,
        include_regime: bool = True,
    ) -> pd.DataFrame:
        """
        Point d'entrée principal : calcule toutes les features.

        Args:
            df: OHLCV du timeframe principal
            higher_tf_df: OHLCV du higher timeframe (aligné sur l'index de df)
            include_regime: Ajouter la détection de régime de marché

        Returns:
            DataFrame de features normalisées
        """
        logger.debug(f"Feature engineering sur {len(df)} bougies...")
        features = pd.DataFrame(index=df.index)

        # --- Retours logarithmiques ---
        features = self._add_returns(df, features)

        # --- Indicateurs de tendance ---
        features = self._add_trend_indicators(df, features)

        # --- Indicateurs de momentum ---
        features = self._add_momentum_indicators(df, features)

        # --- Indicateurs de volatilité ---
        features = self._add_volatility_indicators(df, features)

        # --- Indicateurs de volume ---
        features = self._add_volume_indicators(df, features)

        # --- Features de structure de prix ---
        features = self._add_price_structure(df, features)

        # --- Features temporelles ---
        features = self._add_temporal_features(df, features)

        # --- Interactions entre features ---
        features = self._add_interactions(df, features)

        # --- Régime de marché ---
        if include_regime:
            try:
                regime = ind.market_regime(df)
                regime_dummies = pd.get_dummies(regime, prefix='regime')
                features = features.join(regime_dummies)
            except Exception as e:
                logger.warning(f"Régime de marché non calculé : {e}")

        # --- Higher timeframe features ---
        if higher_tf_df is not None:
            features = self._add_htf_features(higher_tf_df, features)

        # --- Normalisation ---
        features = self._normalize(features)

        # Supprimer les lignes avec trop de NaN (warmup)
        min_valid = 0.7  # 70% de features valides requises
        threshold = int(len(features.columns) * (1 - min_valid))
        features = features.dropna(thresh=len(features.columns) - threshold)

        logger.debug(f"Features calculées : {features.shape[1]} colonnes, {len(features)} lignes valides")
        return features

    def _add_returns(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Retours log à différentes périodes."""
        for period in [1, 2, 3, 5, 8, 13, 21]:
            features[f'ret_{period}'] = np.log(df['close'] / df['close'].shift(period))

        # Retours OHLC normalisés
        features['ret_open_close'] = (df['close'] - df['open']) / df['open']
        features['ret_high_low'] = (df['high'] - df['low']) / df['open']
        features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
        features['body_ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-10)
        return features

    def _add_trend_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        # EMAs normalisées par le prix
        for p in [9, 21, 50, 200]:
            ema_val = ind.ema(df['close'], p)
            features[f'ema_{p}_dist'] = (df['close'] - ema_val) / ema_val  # distance relative

        # Croisements EMA
        ema9 = ind.ema(df['close'], 9)
        ema21 = ind.ema(df['close'], 21)
        ema50 = ind.ema(df['close'], 50)
        features['ema_9_21_cross'] = (ema9 - ema21) / df['close']
        features['ema_21_50_cross'] = (ema21 - ema50) / df['close']

        # SuperTrend
        try:
            st = ind.supertrend(df)
            features['supertrend_dir'] = st['supertrend_dir']
            features['supertrend_dist'] = (df['close'] - st['supertrend']) / df['close']
        except Exception:
            pass

        # VWAP distance
        try:
            vwap_val = ind.vwap(df)
            features['vwap_dist'] = (df['close'] - vwap_val) / vwap_val
        except Exception:
            pass

        return features

    def _add_momentum_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        # RSI sur plusieurs périodes
        for p in [7, 14, 21]:
            features[f'rsi_{p}'] = ind.rsi(df['close'], p) / 100  # normaliser 0-1

        # MACD normalisé
        macd_df = ind.macd(df['close'])
        features['macd_diff'] = macd_df['macd_hist'] / df['close']
        features['macd_cross'] = np.sign(macd_df['macd_hist'])

        # Stochastique
        stoch = ind.stochastic(df)
        features['stoch_k'] = stoch['stoch_k'] / 100
        features['stoch_d'] = stoch['stoch_d'] / 100
        features['stoch_cross'] = (stoch['stoch_k'] - stoch['stoch_d']) / 100

        # MFI
        features['mfi'] = ind.mfi(df) / 100

        # Williams %R
        features['williams_r'] = (ind.williams_r(df) + 100) / 100  # 0 à 1

        # CCI normalisé
        cci_val = ind.cci(df)
        features['cci'] = cci_val / 200  # typiquement entre -200 et 200

        # Rate of Change
        for p in [5, 10, 20]:
            features[f'roc_{p}'] = df['close'].pct_change(p)

        return features

    def _add_volatility_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        # ATR normalisé
        atr_val = ind.atr(df)
        features['atr_pct'] = atr_val / df['close']

        # Bollinger Bands
        bb = ind.bollinger_bands(df['close'])
        features['bb_pct_b'] = bb['bb_pct_b']
        features['bb_width'] = bb['bb_width']

        # Keltner
        kc = ind.keltner_channels(df)
        features['kc_position'] = (df['close'] - kc['kc_lower']) / (kc['kc_upper'] - kc['kc_lower'] + 1e-10)

        # Volatilité historique multi-fenêtres
        for p in [5, 10, 20, 50]:
            features[f'hv_{p}'] = ind.historical_volatility(df['close'], p)

        # Squeeze Momentum
        try:
            sq = ind.squeeze_momentum(df)
            features['sq_momentum'] = sq['sq_momentum'] / df['close']
            features['sq_on'] = sq['sq_on']
        except Exception:
            pass

        return features

    def _add_volume_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        # Volume relatif
        for p in [10, 20, 50]:
            features[f'vol_ratio_{p}'] = df['volume'] / df['volume'].rolling(p).mean()

        # OBV trend
        obv_val = ind.obv(df)
        features['obv_trend'] = obv_val.pct_change(10)

        # CMF
        features['cmf'] = ind.cmf(df)

        # VWMA vs close
        vwma_val = ind.vwma(df, 20)
        features['vwma_dist'] = (df['close'] - vwma_val) / vwma_val

        return features

    def _add_price_structure(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Structure de prix : supports, résistances, levels."""
        # Distance aux plus hauts/bas récents
        for p in [10, 20, 50, 100]:
            rolling_high = df['high'].rolling(p).max()
            rolling_low = df['low'].rolling(p).min()
            features[f'dist_high_{p}'] = (rolling_high - df['close']) / df['close']
            features[f'dist_low_{p}'] = (df['close'] - rolling_low) / df['close']
            features[f'range_pct_{p}'] = (rolling_high - rolling_low) / rolling_low

        # Pivot points
        try:
            pp = ind.pivot_points(df)
            features['pivot_dist'] = (df['close'] - pp['pivot']) / df['close']
            features['r1_dist'] = (pp['r1'] - df['close']) / df['close']
            features['s1_dist'] = (df['close'] - pp['s1']) / df['close']
        except Exception:
            pass

        return features

    def _add_temporal_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Features temporelles cycliques (sin/cos encoding)."""
        idx = df.index
        if hasattr(idx, 'hour'):
            # Heure du jour (cycle 24h)
            features['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24)

            # Jour de la semaine (cycle 7j)
            features['dow_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7)
            features['dow_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7)

            # Semaine de l'année (saisonnalité crypto)
            features['week_sin'] = np.sin(2 * np.pi * idx.isocalendar().week.astype(int) / 52)
            features['week_cos'] = np.cos(2 * np.pi * idx.isocalendar().week.astype(int) / 52)

            # Mois (cycle annuel)
            features['month_sin'] = np.sin(2 * np.pi * idx.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * idx.month / 12)

        return features

    def _add_interactions(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Features d'interaction importantes pour les modèles."""
        # RSI × volatilité (oversold en forte volatilité = fort signal)
        if 'rsi_14' in features.columns and 'atr_pct' in features.columns:
            features['rsi_vol_interact'] = features['rsi_14'] * features['atr_pct']

        # Volume × momentum (confirmation du signal)
        if 'vol_ratio_20' in features.columns and 'ret_1' in features.columns:
            features['vol_momentum'] = features['vol_ratio_20'] * np.sign(features['ret_1'])

        # Trend alignment (EMA9 > EMA21 > EMA50)
        if all(f in features.columns for f in ['ema_9_21_cross', 'ema_21_50_cross']):
            features['trend_align'] = np.sign(features['ema_9_21_cross']) * np.sign(features['ema_21_50_cross'])

        return features

    def _add_htf_features(
        self,
        htf_df: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Ajoute les features du higher timeframe (alignées)."""
        htf_features = pd.DataFrame(index=htf_df.index)

        htf_features['htf_rsi'] = ind.rsi(htf_df['close']) / 100
        macd_htf = ind.macd(htf_df['close'])
        htf_features['htf_macd_hist'] = macd_htf['macd_hist'] / htf_df['close']
        htf_features['htf_ret_1'] = np.log(htf_df['close'] / htf_df['close'].shift(1))

        # Réindexer et forward-fill pour aligner sur le TF principal
        htf_reindexed = htf_features.reindex(features.index, method='ffill')
        htf_reindexed.columns = [f'htf_{c}' if not c.startswith('htf_') else c for c in htf_reindexed.columns]
        return features.join(htf_reindexed)

    def _normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalise les features selon la méthode configurée."""
        method = self.norm_method
        numeric_cols = features.select_dtypes(include=[np.number]).columns

        if method == "rolling_zscore":
            window = 200
            for col in numeric_cols:
                rolling_mean = features[col].rolling(window, min_periods=50).mean()
                rolling_std = features[col].rolling(window, min_periods=50).std()
                features[col] = (features[col] - rolling_mean) / (rolling_std + 1e-8)
                features[col] = features[col].clip(-5, 5)  # clipping anti-outliers

        elif method == "minmax":
            window = 200
            for col in numeric_cols:
                rolling_min = features[col].rolling(window, min_periods=50).min()
                rolling_max = features[col].rolling(window, min_periods=50).max()
                features[col] = (features[col] - rolling_min) / (rolling_max - rolling_min + 1e-8)

        elif method == "robust":
            window = 200
            for col in numeric_cols:
                rolling_median = features[col].rolling(window, min_periods=50).median()
                rolling_iqr = features[col].rolling(window, min_periods=50).quantile(0.75) - \
                              features[col].rolling(window, min_periods=50).quantile(0.25)
                features[col] = (features[col] - rolling_median) / (rolling_iqr + 1e-8)
                features[col] = features[col].clip(-5, 5)

        return features

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Retourne la liste des noms de features calculées."""
        sample = self.compute_all(df.tail(300))
        return list(sample.columns)
