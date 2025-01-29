import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualization:
    """Affiche les interpolations polynomiales et l'évolution de R² et MSE en fonction de l'ordre."""
    
    def __init__(self, df, x_col, y_col):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col

    def generate_interpolations(self, results, extend_factor=1.2):
        """Génère des interpolations et les prolonge au-delà des données d'origine."""
        min_x, max_x = min(self.df[self.x_col]), max(self.df[self.x_col])
        range_x = max_x - min_x
        
        # Extrapoler en dehors du cadre
        min_x_extended = min_x - extend_factor * range_x / 10  
        max_x_extended = max_x + extend_factor * range_x / 10
        
        X_interp = np.linspace(min_x_extended, max_x_extended, 200)

        for res in results:
            res["X_interp"] = X_interp
            res["Y_interp"] = res["poly_eq"](X_interp)  # Appliquer le polynôme extrapolé

    def plot(self, results):
        """Affiche deux graphes : interpolations polynomiales et courbes de MSE/R²."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})

        # 📌 1️⃣ Graphique des interpolations polynomiales
        axes[0].scatter(self.df[self.x_col], self.df[self.y_col], color='red', label="Données d'origine")

        legend_texts = []
        for res in results:
            axes[0].plot(res["X_interp"], res["Y_interp"], label=f"Ordre {res['degree']}")
            legend_texts.append(f"Ordre {res['degree']}:\n  R²={res['r2_score']:.4f}, MSE={res['mse']:.4f}")

        legend_text = "\n\n".join(legend_texts)
        axes[0].text(0.98, 0.02, legend_text, fontsize=10, transform=axes[0].transAxes, 
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.6))

        axes[0].set_title("Interpolations polynomiales extrapolées")
        axes[0].set_xlabel(self.x_col)
        axes[0].set_ylabel(self.y_col)
        axes[0].legend()
        axes[0].grid(True)

        # 📌 2️⃣ Graphique des courbes de MSE et R² en fonction de l’ordre
        degrees = [res["degree"] for res in results]
        mse_values = [res["mse"] for res in results]
        r2_values = [res["r2_score"] for res in results]

        axes[1].plot(degrees, mse_values, marker='o', linestyle='-', color='blue', label="MSE")
        axes[1].plot(degrees, r2_values, marker='s', linestyle='-', color='green', label="R²")

        axes[1].set_title("Évolution de MSE et R² en fonction de l'ordre du polynôme")
        axes[1].set_xlabel("Ordre du polynôme")
        axes[1].set_ylabel("Valeur")
        axes[1].legend()
        axes[1].grid(True)

        # Afficher les deux graphes
        plt.tight_layout()
        plt.show()
