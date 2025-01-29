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
        """Ajuste un polynôme d'ordre `degree`, génère les interpolations et retourne les métriques."""

        # 📌 Normalisation pour éviter les erreurs numériques
        X_mean = np.mean(self.X)
        X_std = np.std(self.X)
        X_norm = (self.X - X_mean) / X_std

        # Ajustement du polynôme sur les X normalisés
        coeffs = np.polyfit(X_norm, self.Y, degree)
        poly_eq = np.poly1d(coeffs)

        # Prédiction sur les X normalisés
        Y_pred = poly_eq(X_norm)

        # Génération des interpolations
        X_interp = np.linspace(min(self.X), max(self.X), 200)  # ✅ 200 points pour affichage
        X_interp_norm = (X_interp - X_mean) / X_std  # ✅ Normalisation des nouveaux points
        Y_interp = poly_eq(X_interp_norm)  # ✅ Calcul des Y correspondants

        # Calcul des métriques
        mse = mean_squared_error(self.Y, Y_pred)
        r2 = r2_score(self.Y, Y_pred)

        return {
            "degree": degree,
            "coefficients": coeffs,
            "r2_score": r2,
            "mse": mse,
            "poly_eq": lambda x: poly_eq((x - X_mean) / X_std),  # Ajustement de l'échelle
            "X_interp": X_interp,  # ✅ Correction : Assure 200 valeurs
            "Y_interp": Y_interp  # ✅ Correction : Assure 200 valeurs aussi
        }
