import numpy as np
from utile.polynomial_regression import PolynomialRegression

class LOOCV:
    """Effectue une Leave-One-Out Cross-Validation pour plusieurs ordres de polynôme."""

    def __init__(self, df, x_col, y_col):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.X = df[x_col].values
        self.Y = df[y_col].values

    def cross_validate(self, degree):
        """Effectue LOOCV en retirant un point à la fois et calcule le MSE moyen."""
        n = len(self.X)
        squared_errors = []
        interpolations = []

        for i in range(n):
            df_train = self.df.drop(self.df.index[i])

            poly_reg = PolynomialRegression(df_train, self.x_col, self.y_col)
            result = poly_reg.fit(degree)

            Y_pred = result["poly_eq"](self.X[i])

            squared_error = (self.Y[i] - Y_pred) ** 2
            squared_errors.append(squared_error)

            # 📌 Prolongation de la courbe LOOCV
            X_interp = np.linspace(min(self.X), max(self.X) * 1.1, 210)
            Y_interp = result["poly_eq"](X_interp)

            interpolations.append({
                "X_interp": X_interp,
                "Y_interp": Y_interp
            })

        avg_mse = np.mean(squared_errors)

        return {
            "degree": degree,
            "avg_mse": avg_mse,
            "interpolations": interpolations
        }
