import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class PolynomialRegression:
    """Gère une régression polynomiale pour un ordre donné."""

    def __init__(self, df, x_col, y_col):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.X = df[x_col].values
        self.Y = df[y_col].values

    def fit(self, degree):
        """Ajuste un polynôme d'ordre `degree` et retourne les résultats."""
        coeffs = np.polyfit(self.X, self.Y, degree)  # Ajustement du polynôme
        poly_eq = np.poly1d(coeffs)  # Création de la fonction polynomiale
        
        Y_pred = poly_eq(self.X)  # Prédiction sur les points existants
        
        # Calculer R² et MSE
        r2 = r2_score(self.Y, Y_pred)
        mse = mean_squared_error(self.Y, Y_pred)

        # Générer des X interpolés pour lisser l'affichage
        X_new = np.linspace(min(self.X), max(self.X), 200)
        Y_new = poly_eq(X_new)

        return {
            "degree": degree,
            "coefficients": coeffs,
            "r2_score": r2,
            "mse": mse,
            "X_interp": X_new,
            "Y_interp": Y_new
        }
