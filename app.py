from utile.dataset_manager import DatasetManager
from utile.polynomial_regression import PolynomialRegression
from utile.visualization import Visualization

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

        # Définir les ordres de polynômes à tester
        max_degree = len(df_selected) - 1
        print(f"\n📌 Nombre maximum de points : {max_degree}")

        try:
            degree_min = int(input(f"Entrez l'ordre minimum du polynôme (>=1) : "))
            degree_max = int(input(f"Entrez l'ordre maximum du polynôme (<= {max_degree}) : "))

            if degree_min < 1 or degree_max > max_degree or degree_min > degree_max:
                raise ValueError("Valeurs incorrectes.")

        except ValueError:
            print("\n❌ Erreur : Entrée invalide.")
            exit()

        # Stocker les résultats
        results = [poly_reg.fit(degree) for degree in range(degree_min, degree_max + 1)]

        # Générer les interpolations et afficher les graphiques
        viz = Visualization(df_selected, x_col, y_col)
        viz.generate_interpolations(results)
        viz.plot(results)
