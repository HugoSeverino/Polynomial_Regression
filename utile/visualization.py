import matplotlib.pyplot as plt
import numpy as np
import os

class Visualization:
    """Affiche les interpolations polynomiales et les résultats de LOOCV."""

    def __init__(self, df, x_col, y_col, output_dir="output"):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_interpolations(self, results):
        """Affiche et sauvegarde les interpolations classiques avec prolongation."""
        plt.figure(figsize=(12, 6))

        plt.scatter(self.df[self.x_col], self.df[self.y_col], color='red', label="Données d'origine")

        for res in results:
            plt.plot(res["X_interp"], res["Y_interp"], label=f"Ordre {res['degree']}")

        plt.title("Interpolations polynomiales avec prolongation")
        plt.xlabel(self.x_col)
        plt.ylabel(self.y_col)
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(self.output_dir, "interpolations.png")
        plt.savefig(output_path, dpi=300)
        print(f"✅ Graphique sauvegardé : {output_path}")

        plt.show()

    def plot_loocv(self, loocv_results):
        """Affiche et sauvegarde les interpolations LOOCV prolongées."""
        num_orders = len(loocv_results)
        num_cols = 3
        num_rows = -(-num_orders // num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = np.array(axes).reshape(num_rows, num_cols)

        colors = plt.cm.viridis(np.linspace(0, 1, num_orders))

        for i, (degree, res) in enumerate(loocv_results.items()):
            row, col = divmod(i, num_cols)
            ax = axes[row, col]

            ax.scatter(self.df[self.x_col], self.df[self.y_col], color='red', label="Données d'origine", alpha=0.7)

            for interp in res["interpolations"]:
                ax.plot(interp["X_interp"], interp["Y_interp"], color=colors[i], alpha=0.3)

            ax.set_title(f"LOOCV - Ordre {degree}")
            ax.set_xlabel(self.x_col)
            ax.set_ylabel(self.y_col)
            ax.grid(True)
            ax.legend(["LOOCV Régressions", "Données originales"])

        for i in range(num_orders, num_rows * num_cols):
            row, col = divmod(i, num_cols)
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "loocv.png")
        plt.savefig(output_path, dpi=300)
        print(f"✅ Graphique sauvegardé : {output_path}")

        plt.show()


    def plot_loocv_mse(self, loocv_results, results):

        # Configuration de la figure
        plt.figure(figsize=(12, 6))
        
        # Extraction des données
        degrees = list(loocv_results.keys())
        mse_loocv = [res["avg_mse"] for res in loocv_results.values()]
        mse_std = [res["mse"] for res in results]
        r2_values = [res["r2_score"] for res in results]
        
        # Premier axe Y pour les MSE
        ax1 = plt.gca()  # Get current axis
        ax1.plot(degrees, mse_loocv, 'b-o', label="MSE LOOCV")
        ax1.plot(degrees, mse_std, 'r--s', label="MSE Standard")
        ax1.set_xlabel("Degré du polynôme", fontsize=12)
        ax1.set_ylabel("MSE", color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Deuxième axe Y pour le R²
        ax2 = ax1.twinx()
        ax2.plot(degrees, r2_values, 'g-^', label="R² Standard")
        ax2.set_ylabel("R²", color='g', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Titre et légende
        plt.title("Évolution des Métriques par Degré Polynomial", fontsize=14, pad=20)
        ax1.legend(loc='upper left', bbox_to_anchor=(0.1, 1.15))
        ax2.legend(loc='upper right', bbox_to_anchor=(0.9, 1.15))
        
        # Sauvegarde et affichage
        output_path = os.path.join(self.output_dir, "mse_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé : {output_path}")
        plt.show()
        plt.close()
