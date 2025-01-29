import matplotlib.pyplot as plt
import seaborn as sns
from utile.dataset_manager import DatasetManager
from utile.polynomial_regression import PolynomialRegression

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

        # Choisir une plage d'ordres de polynôme à tester
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
        results = []

        # Boucle sur les ordres choisis
        for degree in range(degree_min, degree_max + 1):
            result = poly_reg.fit(degree)
            results.append(result)

            # Affichage des résultats
            print(f"\n🔹 Polynôme d'ordre {degree}")
            print(f"  - Coefficients : {result['coefficients']}")
            print(f"  - R² Score : {result['r2_score']:.4f}")
            print(f"  - MSE : {result['mse']:.4f}")

        # Visualisation des interpolations
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df_selected[x_col], y=df_selected[y_col], color='red', label="Données d'origine")

        for res in results:
            plt.plot(res["X_interp"], res["Y_interp"], label=f"Ordre {res['degree']}")

        plt.title(f"Interpolations polynomiales d'ordre {degree_min} à {degree_max}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.grid(True)
        plt.show()
