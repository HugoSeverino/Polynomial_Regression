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
        # Effectuer les interpolations polynomiales
        poly_reg = PolynomialRegression(df_selected, x_col, y_col)
        poly_reg.fit_polynomials()
        results = poly_reg.get_results()

        # Affichage des résultats
        print("\n📊 Résultats des interpolations polynomiales :")
        for degree, res in results.items():
            print(f"\n🔹 Polynôme d'ordre {degree}")
            print(f"  - Coefficients : {res['coefficients']}")
            print(f"  - R² Score : {res['r2_score']:.4f}")
            print(f"  - MSE : {res['mse']:.4f}")

        # Visualisation des interpolations
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df_selected[x_col], y=df_selected[y_col], color='red', label="Données d'origine")

        for degree, res in results.items():
            plt.plot(res["X_interp"], res["Y_interp"], label=f"Ordre {degree}")

        plt.title(f"Interpolations polynomiales d'ordre 1 à {len(df_selected)-1}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.grid(True)
        plt.show()
