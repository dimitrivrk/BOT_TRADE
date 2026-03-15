"""
Temporal Fusion Transformer (TFT) pour la prédiction de prix multi-horizon.
Architecture basée sur : Lim et al. 2021 "Temporal Fusion Transformers for
Interpretable Multi-horizon Time Series Forecasting"

Implémentation via PyTorch Lightning + pytorch-forecasting.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import pickle

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from utils.logger import setup_logger

logger = setup_logger("models.tft")


class TFTPredictor:
    """
    Wrapper haut niveau autour du TFT de pytorch-forecasting.
    Gère l'entraînement, la prédiction et la sauvegarde.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg["models"]["tft"]
        self.hidden_size = int(self.cfg["hidden_size"])
        self.lstm_layers = int(self.cfg["lstm_layers"])
        self.attention_heads = int(self.cfg["num_attention_heads"])
        self.dropout = float(self.cfg["dropout"])
        self.horizon = int(self.cfg["horizon"])
        self.encoder_steps = int(self.cfg["encoder_steps"])
        self.batch_size = int(self.cfg["batch_size"])
        self.max_epochs = int(self.cfg["max_epochs"])
        self.lr = float(self.cfg["learning_rate"])
        self.checkpoint_dir = Path(self.cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[TemporalFusionTransformer] = None
        self.training_dataset: Optional[TimeSeriesDataSet] = None

    def prepare_dataset(
        self,
        features_df: pd.DataFrame,
        symbol: str,
        target_col: str = "ret_1",  # prédire le return à 1 pas
    ) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Prépare les TimeSeriesDataSet pour TFT.

        Args:
            features_df: DataFrame de features normalisées (index=timestamp)
            symbol: Symbole de la paire (ex: 'BTC/USDT')
            target_col: Colonne cible (return futur)

        Returns:
            (train_dataset, val_dataset)
        """
        df = features_df.copy().reset_index()
        df.columns = [str(c) for c in df.columns]
        df['symbol'] = symbol
        df['time_idx'] = np.arange(len(df))

        # Target : return futur (shift de -horizon)
        # On prédit la direction et magnitude
        df['target'] = df[target_col].shift(-1)  # next-step return
        df = df.dropna(subset=['target'])

        # Split train/val (80/20 chronologique)
        val_size = int(len(df) * 0.2)
        training_cutoff = df['time_idx'].max() - val_size

        # Identifier les features continues et catégorielles
        time_varying_known_reals = ['time_idx']
        time_varying_unknown_reals = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ['time_idx', 'target', 'timestamp']
        ]

        # ── Nettoyage : remplacer inf/-inf puis NaN par 0 ──────────────────────
        cols_to_clean = time_varying_unknown_reals + ['target']
        df[cols_to_clean] = df[cols_to_clean].replace([np.inf, -np.inf], np.nan)
        nan_before = df[cols_to_clean].isna().sum().sum()
        if nan_before > 0:
            logger.warning(f"TFT prepare_dataset : {nan_before} NaN/inf détectés, remplis à 0")
        df[cols_to_clean] = df[cols_to_clean].fillna(0)
        # ───────────────────────────────────────────────────────────────────────

        # Dataset d'entraînement
        training = TimeSeriesDataSet(
            df[df['time_idx'] <= training_cutoff],
            time_idx='time_idx',
            target='target',
            group_ids=['symbol'],
            min_encoder_length=self.encoder_steps // 2,
            max_encoder_length=self.encoder_steps,
            min_prediction_length=1,
            max_prediction_length=self.horizon,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=TorchNormalizer(method="robust"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        self.training_dataset = training

        # Dataset de validation (mêmes paramètres, predict=False pour avoir tous les samples)
        # On inclut encoder_steps rows de chevauchement pour que le premier sample soit complet
        validation = TimeSeriesDataSet.from_dataset(
            training,
            df[df['time_idx'] > training_cutoff - self.encoder_steps],
            predict=False,
            stop_randomization=True,
        )

        logger.info(
            f"Dataset TFT : train={len(training)} samples, val={len(validation)} samples, "
            f"features={len(time_varying_unknown_reals)}"
        )
        return training, validation

    def build_model(self, training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
        """Construit le modèle TFT."""
        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=self.lr,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_heads,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_size // 2,
            lstm_layers=self.lstm_layers,
            output_size=7,          # 7 quantiles : [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        logger.info(
            f"Modèle TFT : {sum(p.numel() for p in model.parameters()):,} paramètres"
        )
        return model

    def train(
        self,
        features_df: pd.DataFrame,
        symbol: str,
        gpu: bool = torch.cuda.is_available(),
    ) -> TemporalFusionTransformer:
        """Entraîne le TFT."""
        training, validation = self.prepare_dataset(features_df, symbol)

        # Sur CPU Windows, num_workers>0 cause des problèmes de spawn → 0
        n_workers = 0 if not torch.cuda.is_available() else 4
        train_loader = training.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=n_workers,
            persistent_workers=(n_workers > 0),
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=self.batch_size * 2, num_workers=n_workers,
            persistent_workers=(n_workers > 0),
        )

        model = self.build_model(training)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ModelCheckpoint(
                dirpath=str(self.checkpoint_dir),
                filename=f"tft_{symbol.replace('/', '_')}_{{epoch}}_{{val_loss:.4f}}",
                monitor="val_loss",
                save_top_k=3,
                mode="min",
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ]

        tb_logger = TensorBoardLogger(
            save_dir="logs/tensorboard",
            name=f"tft_{symbol.replace('/', '_')}",
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if gpu else "cpu",
            devices=1,
            gradient_clip_val=self.cfg.get("gradient_clip", 1.0),
            callbacks=callbacks,
            logger=tb_logger,
            log_every_n_steps=10,
            enable_progress_bar=True,
        )

        logger.info(f"Entraînement TFT pour {symbol}...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        self.model = model

        # Sauvegarder le dataset pour l'inférence
        with open(self.checkpoint_dir / f"training_dataset_{symbol.replace('/', '_')}.pkl", 'wb') as f:
            pickle.dump(training, f)

        logger.info(f"TFT entraîné ! Best val_loss={trainer.callback_metrics.get('val_loss', 'N/A'):.4f}")
        return model

    def predict(
        self,
        features_df: pd.DataFrame,
        symbol: str,
        return_quantiles: bool = True,
    ) -> Dict[str, float]:
        """
        Génère une prédiction de direction et confiance.

        Returns:
            dict avec 'direction' (-1/0/1), 'confidence' (0-1), 'quantiles'
        """
        if self.model is None:
            self.load(symbol)

        try:
            # Préparer les données pour l'inférence
            df = features_df.tail(self.encoder_steps + 10).copy().reset_index()
            df.columns = [str(c) for c in df.columns]
            df['symbol'] = symbol
            df['time_idx'] = np.arange(len(df))
            df['target'] = 0.0  # placeholder

            # Charger le dataset de training pour les paramètres
            dataset_path = self.checkpoint_dir / f"training_dataset_{symbol.replace('/', '_')}.pkl"
            with open(dataset_path, 'rb') as f:
                training = pickle.load(f)

            pred_dataset = TimeSeriesDataSet.from_dataset(
                training, df, predict=True, stop_randomization=True
            )
            pred_loader = pred_dataset.to_dataloader(
                train=False, batch_size=1, num_workers=0
            )

            with torch.no_grad():
                preds = self.model.predict(
                    pred_loader,
                    mode="quantiles",
                    return_index=True,
                )

            # Extraire la prédiction médiane (quantile 0.5) et l'intervalle
            median_pred = float(preds[0][0, -1, 3])  # horizon=1, quantile médian
            q10 = float(preds[0][0, -1, 1])
            q90 = float(preds[0][0, -1, 5])

            # Calculer la direction et la confiance
            direction = 1 if median_pred > 0 else (-1 if median_pred < 0 else 0)

            # Confiance : basée sur l'intervalle de confiance et la magnitude
            interval_width = abs(q90 - q10)
            magnitude = abs(median_pred)
            confidence = min(magnitude / (interval_width + 1e-6), 1.0)
            # Ajustement : si q10 et q90 du même côté → forte conviction
            if q10 > 0 and q90 > 0:
                confidence = min(confidence * 1.3, 1.0)
            elif q10 < 0 and q90 < 0:
                confidence = min(confidence * 1.3, 1.0)

            return {
                "direction": direction,
                "confidence": float(confidence),
                "predicted_return": float(median_pred),
                "q10": float(q10),
                "q90": float(q90),
                "model": "tft",
            }

        except Exception as e:
            logger.error(f"Erreur prédiction TFT : {e}")
            return {"direction": 0, "confidence": 0.0, "predicted_return": 0.0, "model": "tft"}

    def load(self, symbol: str, checkpoint_path: Optional[str] = None):
        """Charge un modèle depuis un checkpoint."""
        if checkpoint_path:
            path = checkpoint_path
        else:
            # Chercher le meilleur checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob(f"tft_{symbol.replace('/', '_')}*.ckpt"))
            if not checkpoints:
                raise FileNotFoundError(f"Aucun checkpoint TFT pour {symbol}")
            path = str(checkpoints[-1])

        self.model = TemporalFusionTransformer.load_from_checkpoint(path)
        self.model.eval()
        logger.info(f"TFT chargé depuis {path}")

    def get_attention_weights(self, features_df: pd.DataFrame, symbol: str) -> pd.Series:
        """
        Récupère les poids d'attention pour l'interprétabilité.
        Indique quelles features sont les plus importantes.
        """
        if self.model is None:
            self.load(symbol)
        # L'interprétabilité TFT est intégrée dans pytorch-forecasting
        # via model.interpret_output()
        interpretation = self.model.interpret_output(
            self.model.predict(
                features_df.tail(self.encoder_steps),
                mode="raw",
                return_index=True,
            ),
            reduction="sum",
        )
        return interpretation["encoder_variables"]
