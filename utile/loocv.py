import numpy as np

class LOOCV:
    """Effectue une Leave-One-Out Cross-Validation pour une régression polynomiale d'un ordre donné."""

    def __init__(self, df, x_col, y_col):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.X = df[x_col].values
        self.Y = df[y_col].values

    def cross_validate(self, degree):
        """Effectue LOOCV en calculant la moyenne des erreurs quadratiques."""
        n = len(self.X)
        squared_errors = []

        for i in range(n):
            # Exclure un point pour l'entraînement
            X_train = np.delete(self.X, i)
            Y_train = np.delete(self.Y, i)
            X_test = self.X[i]
            Y_test = self.Y[i]

            # Ajuster le modèle polynomiale
            coeffs = np.polyfit(X_train, Y_train, degree)
            poly_eq = np.poly1d(coeffs)

            # Prédiction du point exclu
            Y_pred = poly_eq(X_test)

            # Erreur quadratique
            squared_error = (Y_test - Y_pred) ** 2
            squared_errors.append(squared_error)

        # Calcul de la moyenne des erreurs quadratiques (MSE)
        avg_mse = np.mean(squared_errors)

        return {
            "degree": degree,
            "avg_mse": avg_mse
        }
