import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class PolynomialRegression:
    """Gère une régression polynomiale pour un ordre donné et retourne les métriques associées."""

    def __init__(self, df, x_col, y_col):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.X = df[x_col].values
        self.Y = df[y_col].values

    def fit(self, degree):
        """Ajuste un polynôme d'ordre `degree` et retourne les coefficients, MSE et R²."""
        # Ajustement du polynôme
        coeffs = np.polyfit(self.X, self.Y, degree)
        poly_eq = np.poly1d(coeffs)

        # Prédiction sur les points d'origine
        Y_pred = poly_eq(self.X)

        # Calcul des métriques
        mse = mean_squared_error(self.Y, Y_pred)
        r2 = r2_score(self.Y, Y_pred)

        return {
            "degree": degree,
            "coefficients": coeffs,
            "r2_score": r2,
            "mse": mse,
            "poly_eq": poly_eq  # On stocke l'objet du polynôme pour l'interpolation future
        }
