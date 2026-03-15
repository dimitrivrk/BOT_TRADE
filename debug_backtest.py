"""
Script de diagnostic du pipeline de backtest.
Lance: python debug_backtest.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

print("=" * 60)
print("DIAGNOSTIC BACKTEST PIPELINE")
print("=" * 60)

# 1. Chargement des données
print("\n[1] Chargement des données...")
from data.collectors.historical import HistoricalDataCollector
collector = HistoricalDataCollector()
df = collector.load_data('BTC/USDT', '1h', start='2024-01-01', end='2024-12-31')
print(f"    OHLCV shape  : {df.shape}")
print(f"    NaN count    : {df.isna().sum().sum()}")
print(f"    Colonnes     : {list(df.columns)}")
print(f"    Index[0]     : {df.index[0]}")
print(f"    Index[-1]    : {df.index[-1]}")

# 2. Feature engineering
print("\n[2] Feature engineering...")
from data.processors.features import FeatureEngineer
fe = FeatureEngineer()
try:
    features = fe.compute_all(df)
    print(f"    Features shape : {features.shape}")
    print(f"    NaN total      : {features.isna().sum().sum()}")
    print(f"    Colonnes dispo : {list(features.columns[:10])}...")

    # Vérifier les colonnes clés
    key_cols = ['ema_9_21_cross', 'ema_21_50_cross', 'supertrend_dist',
                'rsi_14', 'macd_diff', 'cmf', 'vwap_dist']
    print("\n    Colonnes clés pour le signal :")
    for col in key_cols:
        if col in features.columns:
            s = features[col].dropna()
            print(f"      {col:25s} : mean={s.mean():.3f}  std={s.std():.3f}  "
                  f"min={s.min():.3f}  max={s.max():.3f}  NaN={features[col].isna().sum()}")
        else:
            print(f"      {col:25s} : ⚠ ABSENT")
except Exception as e:
    print(f"    ERREUR: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# 3. Génération des signaux
print("\n[3] Génération des signaux...")
score = pd.Series(0.0, index=features.index)
n = 0
for col, w in [('ema_9_21_cross', 0.25), ('ema_21_50_cross', 0.20),
               ('supertrend_dist', 0.20), ('vwap_dist', 0.10)]:
    if col in features.columns:
        score += np.sign(features[col].fillna(0)) * w
        n += w

if 'rsi_14' in features.columns:
    rsi_z = features['rsi_14'].fillna(0)
    score += np.where(rsi_z < -1.5, 0.3, np.where(rsi_z > 1.5, -0.3, np.sign(rsi_z) * 0.1))
    n += 0.3

if 'macd_diff' in features.columns:
    score += np.sign(features['macd_diff'].fillna(0)) * 0.15
    n += 0.15

if n > 0:
    score = score / n * 1.2
score = score.fillna(0).clip(-1, 1)

entry_long  = (score > 0.15)
entry_short = (score < -0.15)

print(f"    Score  : mean={score.mean():.4f}  std={score.std():.4f}  "
      f"min={score.min():.4f}  max={score.max():.4f}")
print(f"    entry_long  : {entry_long.sum()} signaux ({entry_long.mean():.1%})")
print(f"    entry_short : {entry_short.sum()} signaux ({entry_short.mean():.1%})")

if entry_long.sum() == 0:
    print("\n    ⚠ AUCUN SIGNAL LONG - détail du score:")
    print(f"    Valeurs uniques du score (top 20): {sorted(score.unique())[-20:]}")

# 4. Alignment
print("\n[4] Alignement features/df...")
common_index = features.index
df_aligned = df.loc[df.index.isin(common_index)].reindex(common_index).dropna()
print(f"    df avant alignement  : {len(df)} lignes")
print(f"    df après alignement  : {len(df_aligned)} lignes")
print(f"    Pertes               : {len(df) - len(df_aligned)} lignes")

# 5. Test vectorbt minimal
print("\n[5] Test vectorbt minimal (10 premiers signaux forcés)...")
import vectorbt as vbt
close = df_aligned['close'].iloc[:500]
# Forcer quelques entrées pour tester vectorbt
test_entries = pd.Series(False, index=close.index)
test_entries.iloc[10] = True
test_entries.iloc[100] = True
test_entries.iloc[200] = True

try:
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=test_entries,
        exits=~test_entries,
        size=1000.0,
        size_type="value",
        init_cash=10000.0,
        fees=0.001,
        freq="1h",
        accumulate=False,
    )
    n_trades = int(pf.trades.count())
    print(f"    Test vectorbt OK : {n_trades} trades générés")
    print(f"    Total return     : {pf.total_return():.2%}")
except Exception as e:
    print(f"    ERREUR vectorbt: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC TERMINÉ")
print("=" * 60)
