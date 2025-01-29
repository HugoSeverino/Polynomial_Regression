import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class PolynomialRegression:
    """
    Implémente la régression polynomiale avec normalisation des données
    et calcul des métriques de performance
    """
    
    def __init__(self, df, x_col, y_col):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.X = df[x_col].values  # Conversion en array numpy
        self.Y = df[y_col].values
        
        # Stockage des paramètres de normalisation
        self.X_mean = np.mean(self.X)
        self.X_std = np.std(self.X)

    def _normalize(self, X):
        """Normalisation Z-score des données X"""
        return (X - self.X_mean) / self.X_std if self.X_std != 0 else X - self.X_mean

    def fit(self, degree):
        """
        Entraîne un modèle polynomial et calcule les prédictions
        
        Args:
            degree (int): Degré du polynôme à ajuster
            
        Returns:
            dict: Résultats contenant les coefficients et métriques
        """
        
        # Normalisation des données
        X_norm = self._normalize(self.X)
        
        # Ajustement polynomial (algorithme des moindres carrés)
        coeffs = np.polyfit(X_norm, self.Y, degree)
        poly_eq = np.poly1d(coeffs)  # Création de la fonction polynomiale
        
        # Prédictions sur les données d'entraînement
        Y_pred = poly_eq(X_norm)
        
        # Génération de points interpolés étendus pour visualisation
        X_interp = np.linspace(min(self.X), max(self.X) * 1.1, 210)
        X_interp_norm = self._normalize(X_interp)
        Y_interp = poly_eq(X_interp_norm)
        
        # Calcul des métriques de performance
        mse = mean_squared_error(self.Y, Y_pred)
        r2 = r2_score(self.Y, Y_pred)
        
        return {
            "degree": degree,
            "coefficients": coeffs,
            "r2_score": r2,
            "mse": mse,
            "poly_eq": lambda x: poly_eq(self._normalize(x)),  # Fermeture pour conservation params
            "X_interp": X_interp,
            "Y_interp": Y_interp
        }