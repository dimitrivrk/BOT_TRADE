"""
Module de sélection de features basé sur XGBoost pour le bot de trading crypto.

Ce module utilise XGBoost pour identifier les features les plus importantes
prédisant les rendements futurs, conformément aux recherches 2025 montrant
que la sélection de features XGBoost + ML surpassent les modèles autonomes.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import yaml

# Configuration du logging
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Sélecteur de features basé sur XGBoost pour prédiction des rendements crypto.

    Cet sélecteur entraîne un modèle XGBoost sur la prédiction des rendements futurs
    et utilise l'importance des features pour sélectionner les top-K features.
    Supporte plusieurs méthodes de sélection et met en cache les résultats.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise le sélecteur de features.

        Args:
            config_path: Chemin vers le fichier de configuration YAML
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.selected_features: Optional[List[str]] = None
        self.importance_df: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()
        self.model: Optional[xgb.XGBClassifier] = None

        logger.info("FeatureSelector initialisé avec config: %s", config_path)

    def _load_config(self) -> dict:
        """
        Charge la configuration depuis le fichier YAML.

        Returns:
            Configuration pour la sélection de features

        Raises:
            FileNotFoundError: Si le fichier config n'existe pas
            yaml.YAMLError: Si la configuration YAML est invalide
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('models', {}).get('feature_selection', {})
        except FileNotFoundError:
            logger.warning("Fichier config non trouvé: %s. Utilisation des defaults.",
                          self.config_path)
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error("Erreur lors du parsing YAML: %s", e)
            return self._get_default_config()

    @staticmethod
    def _get_default_config() -> dict:
        """
        Retourne la configuration par défaut.

        Returns:
            Configuration par défaut pour feature selection
        """
        return {
            'method': 'xgboost',
            'n_features': 30,
            'correlation_threshold': 0.95,
            'xgboost_params': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'random_state': 42
            }
        }

    def fit(self, features_df: pd.DataFrame, prices_df: pd.DataFrame,
            n_top: Optional[int] = None) -> List[str]:
        """
        Entraîne le sélecteur de features sur les données fournies.

        Crée une variable cible basée sur le rendement future 6h, entraîne
        un modèle XGBoost, et sélectionne les top-K features basé sur
        l'importance et les seuils de corrélation.

        Args:
            features_df: DataFrame avec toutes les features
            prices_df: DataFrame avec les prix (colonnes: 'close', 'timestamp')
            n_top: Nombre de top features à sélectionner (default: config.n_features)

        Returns:
            Liste des noms de features sélectionnées

        Raises:
            ValueError: Si les données sont invalides ou insuffisantes
            KeyError: Si les colonnes requises sont manquantes
        """
        try:
            # Validation des entrées
            if features_df.empty:
                raise ValueError("features_df ne peut pas être vide")
            if prices_df.empty:
                raise ValueError("prices_df ne peut pas être vide")

            if 'close' not in prices_df.columns:
                raise KeyError("prices_df doit contenir une colonne 'close'")

            # Configuration du nombre de features
            n_top = n_top or self.config.get('n_features', 30)

            logger.info("Début de l'entraînement avec %d features candidates",
                       features_df.shape[1])

            # Création de la variable cible: rendement futur 6h
            target = self._create_target(prices_df)

            # Alignement des indices
            common_idx = features_df.index.intersection(target.index)
            if len(common_idx) < len(target) * 0.8:
                logger.warning("Moins de 80%% des indices communs trouvés")

            X = features_df.loc[common_idx]
            y = target.loc[common_idx]

            logger.info("Données alignées: %d samples avec %d features",
                       X.shape[0], X.shape[1])

            # Gestion de la méthode de sélection
            method = self.config.get('method', 'xgboost')

            if method == 'xgboost':
                importance_scores = self._get_xgboost_importance(X, y)
            elif method == 'mutual_information':
                importance_scores = self._get_mutual_information(X, y)
            elif method == 'correlation_filter':
                importance_scores = self._get_correlation_importance(X)
            else:
                logger.warning("Méthode '%s' inconnue. Utilisation de xgboost.", method)
                importance_scores = self._get_xgboost_importance(X, y)

            # Création du DataFrame d'importance
            self.importance_df = pd.DataFrame({
                'feature': importance_scores.keys(),
                'importance': importance_scores.values()
            }).sort_values('importance', ascending=False)

            logger.info("Top 10 features par importance:\n%s",
                       self.importance_df.head(10).to_string())

            # Suppression des features hautement corrélées
            threshold = self.config.get('correlation_threshold', 0.95)
            selected = self._remove_correlated_features(
                X[self.importance_df['feature'].tolist()],
                threshold
            )

            # Sélection des top-K features
            top_features = selected[:min(n_top, len(selected))]
            self.selected_features = top_features.tolist()

            logger.info("Features sélectionnées: %d (top %d)",
                       len(self.selected_features), n_top)

            return self.selected_features

        except Exception as e:
            logger.error("Erreur lors de l'entraînement: %s", str(e))
            raise

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme les features en utilisant la sélection effectuée.

        Args:
            features_df: DataFrame avec toutes les features

        Returns:
            DataFrame filtré avec seulement les features sélectionnées

        Raises:
            RuntimeError: Si fit() n'a pas été appelé au préalable
            KeyError: Si des features sélectionnées manquent
        """
        if self.selected_features is None:
            raise RuntimeError("fit() doit être appelé avant transform()")

        missing_features = set(self.selected_features) - set(features_df.columns)
        if missing_features:
            raise KeyError(f"Features manquantes: {missing_features}")

        logger.debug("Transformation de %d features vers %d",
                    features_df.shape[1], len(self.selected_features))

        return features_df[self.selected_features]

    def fit_transform(self, features_df: pd.DataFrame, prices_df: pd.DataFrame,
                     n_top: Optional[int] = None) -> pd.DataFrame:
        """
        Entraîne le sélecteur et transforme les features en une seule opération.

        Args:
            features_df: DataFrame avec toutes les features
            prices_df: DataFrame avec les prix
            n_top: Nombre de top features à sélectionner

        Returns:
            DataFrame filtré avec features sélectionnées
        """
        self.fit(features_df, prices_df, n_top)
        return self.transform(features_df)

    def get_importance(self) -> pd.DataFrame:
        """
        Retourne les scores d'importance des features.

        Returns:
            DataFrame avec colonnes 'feature' et 'importance'

        Raises:
            RuntimeError: Si fit() n'a pas été appelé
        """
        if self.importance_df is None:
            raise RuntimeError("fit() doit être appelé avant get_importance()")

        return self.importance_df.copy()

    def save(self, path: str) -> None:
        """
        Sauvegarde l'état du sélecteur sur disque.

        Args:
            path: Chemin de destination pour la sauvegarde

        Raises:
            RuntimeError: Si fit() n'a pas été appelé
            IOError: En cas d'erreur d'écriture
        """
        if self.selected_features is None:
            raise RuntimeError("fit() doit être appelé avant save()")

        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            state = {
                'selected_features': self.selected_features,
                'importance_df': self.importance_df,
                'config': self.config,
                'model': self.model
            }

            with open(path, 'wb') as f:
                pickle.dump(state, f)

            logger.info("Sélecteur sauvegardé: %s", path)

        except IOError as e:
            logger.error("Erreur lors de la sauvegarde: %s", str(e))
            raise

    def load(self, path: str) -> None:
        """
        Charge l'état du sélecteur depuis le disque.

        Args:
            path: Chemin du fichier à charger

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            pickle.UnpicklingError: Si le fichier est corrompu
        """
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)

            self.selected_features = state['selected_features']
            self.importance_df = state['importance_df']
            self.config = state['config']
            self.model = state.get('model')

            logger.info("Sélecteur chargé: %s (%d features)",
                       path, len(self.selected_features))

        except FileNotFoundError:
            logger.error("Fichier non trouvé: %s", path)
            raise
        except Exception as e:
            logger.error("Erreur lors du chargement: %s", str(e))
            raise

    def _create_target(self, prices_df: pd.DataFrame) -> pd.Series:
        """
        Crée la variable cible basée sur le rendement futur 6h.

        Args:
            prices_df: DataFrame avec colonne 'close'

        Returns:
            Série binaire: 1 si rendement positif, 0 sinon
        """
        prices = prices_df['close'].copy()

        # Calcul du rendement futur (décalage de 6h = 6 périodes si données horaires)
        future_returns = prices.shift(-6) / prices - 1

        # Variable cible binaire
        target = (future_returns > 0).astype(int)

        logger.debug("Cible créée: %d positifs, %d négatifs, %d NaN",
                    (target == 1).sum(), (target == 0).sum(), target.isna().sum())

        return target

    def _get_xgboost_importance(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Calcule l'importance des features avec XGBoost.

        Entraîne un classifieur XGBoost et extrait l'importance basée sur le gain.

        Args:
            X: Features d'entraînement
            y: Variable cible

        Returns:
            Dictionnaire {nom_feature: importance}
        """
        # Suppression des NaN
        mask = y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        if len(y_clean) < 100:
            logger.warning("Peu de samples pour XGBoost: %d", len(y_clean))

        # Entraînement
        params = self.config.get('xgboost_params', {})
        self.model = xgb.XGBClassifier(**params)

        self.model.fit(X_clean, y_clean, verbose=0)

        # Extraction de l'importance (gain)
        importance_dict = self.model.get_booster().get_score(importance_type='gain')

        # Normalisation
        if importance_dict:
            total = sum(importance_dict.values())
            importance_dict = {k: v / total for k, v in importance_dict.items()}
        else:
            # Fallback vers importance weight
            importance_dict = self.model.get_booster().get_score(importance_type='weight')
            total = sum(importance_dict.values()) if importance_dict else 1
            importance_dict = {k: v / total for k, v in importance_dict.items()}

        # Assurer que toutes les features ont une importance
        for col in X.columns:
            if col not in importance_dict:
                importance_dict[col] = 0.0

        logger.info("Importance XGBoost calculée pour %d features", len(importance_dict))

        return importance_dict

    def _get_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Calcule l'importance basée sur l'information mutuelle.

        Args:
            X: Features d'entraînement
            y: Variable cible

        Returns:
            Dictionnaire {nom_feature: importance}
        """
        # Suppression des NaN
        mask = y.notna()
        X_clean = X[mask].copy()
        y_clean = y[mask].copy()

        # Normalisation des features
        X_normalized = self.scaler.fit_transform(X_clean)

        # Calcul de l'information mutuelle
        mi_scores = mutual_info_classif(X_normalized, y_clean, random_state=42)

        # Normalisation
        total = mi_scores.sum()
        importance_dict = {
            col: score / total
            for col, score in zip(X.columns, mi_scores)
        }

        logger.info("Importance mutual information calculée pour %d features",
                   len(importance_dict))

        return importance_dict

    def _get_correlation_importance(self, X: pd.DataFrame) -> dict:
        """
        Calcule l'importance basée sur la corrélation avec le target implicite.

        Args:
            X: Features d'entraînement

        Returns:
            Dictionnaire {nom_feature: importance}
        """
        # Calcul de la variance normalisée comme proxy d'importance
        variances = X.var()
        total = variances.sum()

        importance_dict = {
            col: var / total
            for col, var in variances.items()
        }

        logger.info("Importance corrélation calculée pour %d features",
                   len(importance_dict))

        return importance_dict

    def _remove_correlated_features(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.Index:
        """
        Supprime les features hautement corrélées, en conservant la plus importante.

        Itère sur les features triées par importance et exclut celles
        corrélées à > threshold avec une feature déjà sélectionnée.

        Args:
            X: Features triées par importance
            threshold: Seuil de corrélation absolue

        Returns:
            Index des features non corrélées
        """
        # Calcul de la matrice de corrélation
        corr_matrix = X.corr().abs()

        # Initialisation avec la première feature (la plus importante)
        selected = [X.columns[0]]
        remaining = set(X.columns[1:])

        # Itération sur les features restantes
        for feature in X.columns[1:]:
            # Vérification de la corrélation avec les features sélectionnées
            correlations = corr_matrix.loc[feature, selected]

            if (correlations > threshold).any():
                logger.debug("Feature '%s' supprimée (corrélée > %.2f)", feature, threshold)
            else:
                selected.append(feature)

        logger.info("Features après suppression corrélation: %d / %d",
                   len(selected), len(X.columns))

        return X.columns[X.columns.isin(selected)]


def main():
    """
    Exemple d'utilisation du FeatureSelector.
    """
    logging.basicConfig(level=logging.INFO)

    # Création d'un sélecteur
    selector = FeatureSelector()

    logger.info("FeatureSelector instancié avec succès")
    logger.info("Configuration: %s", selector.config)


if __name__ == "__main__":
    main()
