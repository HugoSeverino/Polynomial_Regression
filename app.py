from utile.dataset_manager import DatasetManager
from utile.polynomial_regression import PolynomialRegression
from utile.visualization import Visualization
from utile.loocv import LOOCV

if __name__ == "__main__":
    file_path = "data/Position_salaries.csv"

    # Gérer le dataset
    dataset_manager = DatasetManager(file_path)
    dataset_manager.display_columns()

    # Sélection des colonnes
    x_col = input("\nChoisissez la colonne pour l'axe X : ")
    y_col = input("Choisissez la colonne pour l'axe Y : ")

    df_selected = dataset_manager.select_columns(x_col, y_col)

    if df_selected is not None:
        poly_reg = PolynomialRegression(df_selected, x_col, y_col)
        loocv = LOOCV(df_selected, x_col, y_col)

        max_degree = len(df_selected) - 1
        print(f"\n📌 Nombre maximum de points : {max_degree}")

        degree_min = int(input(f"Entrez l'ordre minimum du polynôme (>=1) : "))
        degree_max = int(input(f"Entrez l'ordre maximum du polynôme (<= {max_degree}) : "))

        # Stocker les résultats de régression classique
        results = [poly_reg.fit(degree) for degree in range(degree_min, degree_max + 1)]

        # Stocker les résultats LOOCV
        loocv_results = {}
        for degree in range(degree_min, degree_max + 1):
            loocv_results[degree] = loocv.cross_validate(degree)

        # 📌 Affichage des graphiques avec sauvegarde
        viz = Visualization(df_selected, x_col, y_col, output_dir="output")
        viz.plot_interpolations(results)  # 🔹 Graphique 1 : Interpolations classiques
        viz.plot_loocv(loocv_results)     # 🔹 Graphique 2 : LOOCV par ordre
        viz.plot_loocv_mse(loocv_results, results)  # 🔹 Graphique 3 : Comparaison MSE LOOCV vs Standard + R²
