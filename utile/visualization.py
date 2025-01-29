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
        os.makedirs(output_dir, exist_ok=True)  # 📌 Crée le dossier s'il n'existe pas

    def plot_interpolations(self, results):
        """Affiche et sauvegarde les interpolations classiques."""
        plt.figure(figsize=(12, 6))

        plt.scatter(self.df[self.x_col], self.df[self.y_col], color='red', label="Données d'origine")

        for res in results:
            plt.plot(res["X_interp"], res["Y_interp"], label=f"Ordre {res['degree']}")

        plt.title("Interpolations polynomiales")
        plt.xlabel(self.x_col)
        plt.ylabel(self.y_col)
        plt.legend()
        plt.grid(True)

        # 📌 Sauvegarde du fichier
        output_path = os.path.join(self.output_dir, "interpolations.png")
        plt.savefig(output_path, dpi=300)
        print(f"✅ Graphique sauvegardé : {output_path}")

        plt.show()

    def plot_loocv(self, loocv_results):
        """Affiche et sauvegarde les interpolations LOOCV dans une grille avec 3 graphes par ligne."""
        num_orders = len(loocv_results)
        num_cols = 3
        num_rows = -(-num_orders // num_cols)  # 📌 Equivalent à math.ceil(num_orders / num_cols)

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

        # 📌 Sauvegarde du fichier
        output_path = os.path.join(self.output_dir, "loocv.png")
        plt.savefig(output_path, dpi=300)
        print(f"✅ Graphique sauvegardé : {output_path}")

        plt.show()

    def plot_loocv_mse(self, loocv_results, results):
        """Affiche et sauvegarde l'évolution du MSE LOOCV, du MSE standard et du R² en fonction de l'ordre."""
        degrees = list(loocv_results.keys())
        mse_loocv_values = [res["avg_mse"] for res in loocv_results.values()]
        mse_values = [res["mse"] for res in results]
        r2_values = [res["r2_score"] for res in results]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(degrees, mse_loocv_values, marker='o', linestyle='-', color='blue', label="MSE LOOCV")
        ax1.plot(degrees, mse_values, marker='s', linestyle='--', color='orange', label="MSE standard")

        ax1.set_xlabel("Ordre du polynôme")
        ax1.set_ylabel("MSE", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(degrees, r2_values, marker='^', linestyle='-', color='green', label="R² standard")
        ax2.set_ylabel("R²", color="green")
        ax2.tick_params(axis='y', labelcolor="green")
        ax2.legend(loc="upper right")

        ax1.set_title("Évolution du MSE et R² en fonction de l'ordre du polynôme")
        ax1.grid(True)

        # 📌 Sauvegarde du fichier
        output_path = os.path.join(self.output_dir, "mse_comparison.png")
        plt.savefig(output_path, dpi=300)
        print(f"✅ Graphique sauvegardé : {output_path}")

        plt.show()
