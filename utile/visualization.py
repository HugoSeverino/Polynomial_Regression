import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualization:
    """Affiche les interpolations et les données originales avec la légende des métriques."""
    
    def __init__(self, df, x_col, y_col):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col

    def generate_interpolations(self, results):
        """Génère des X interpolés et les Y correspondants pour chaque polynôme."""
        min_x, max_x = min(self.df[self.x_col]), max(self.df[self.x_col])
        X_interp = np.linspace(min_x, max_x, 200)

        for res in results:
            res["X_interp"] = X_interp
            res["Y_interp"] = res["poly_eq"](X_interp)  # Appliquer le polynôme sur X_interp

    def plot(self, results):
        """Affiche les données et les interpolations avec une légende détaillée."""
        plt.figure(figsize=(12, 8))
        
        # Afficher les données d'origine
        sns.scatterplot(x=self.df[self.x_col], y=self.df[self.y_col], color='red', label="Données d'origine")

        # Tracer chaque courbe interpolée
        legend_texts = []
        for res in results:
            plt.plot(res["X_interp"], res["Y_interp"], label=f"Ordre {res['degree']}")
            legend_texts.append(f"Ordre {res['degree']}:\n  R²={res['r2_score']:.4f}, MSE={res['mse']:.4f}")

        # Ajouter une légende détaillée
        legend_text = "\n\n".join(legend_texts)
        plt.text(0.98, 0.02, legend_text, fontsize=10, transform=plt.gca().transAxes, 
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.6))

        plt.title(f"Interpolations polynomiales")
        plt.xlabel(self.x_col)
        plt.ylabel(self.y_col)
        plt.legend()
        plt.grid(True)
        plt.show()
