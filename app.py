"""
ENTRY POINT PRINCIPAL DE L'APPLICATION
Gère le flux d'exécution complet de l'analyse
"""

from utile.dataset_manager import DatasetManager
from utile.polynomial_regression import PolynomialRegression
from utile.visualization import Visualization
from utile.loocv import LOOCV

if __name__ == "__main__":
    # Configuration initiale
    file_path = "data/Position_salaries.csv"  # Chemin vers le dataset
    
    # Phase 1: Gestion des données
    dataset_manager = DatasetManager(file_path)
    dataset_manager.display_columns()  # Affiche les colonnes disponibles
    
    # Sélection interactive des colonnes
    x_col = input("\nChoisissez la colonne pour l'axe X : ")
    y_col = input("Choisissez la colonne pour l'axe Y : ")
    
    # Phase 2: Préparation des analyseurs
    df_selected = dataset_manager.select_columns(x_col, y_col)
    
    if df_selected is not None:
        # Initialisation des composants d'analyse
        poly_reg = PolynomialRegression(df_selected, x_col, y_col)
        loocv = LOOCV(df_selected, x_col, y_col)
        
        # Calcul du degré maximal possible
        max_degree = len(df_selected) - 1  # Théorème d'interpolation polynomiale
        
        # Configuration des paramètres d'analyse
        print(f"\n📌 Nombre maximum de points : {max_degree}")
        degree_min = int(input(f"Entrez l'ordre minimum du polynôme (>=1) : "))
        degree_max = int(input(f"Entrez l'ordre maximum du polynôme (<= {max_degree}) : "))
        
        # Phase 3: Calculs des régressions
        results = [poly_reg.fit(degree) for degree in range(degree_min, degree_max + 1)]
        
        # Phase 4: Validation croisée
        loocv_results = {
            degree: loocv.cross_validate(degree)
            for degree in range(degree_min, degree_max + 1)
        }
        
        # Phase 5: Visualisation et sauvegarde
        viz = Visualization(df_selected, x_col, y_col, output_dir="output")
        viz.plot_interpolations(results)  # Génère le graphique 1
        viz.plot_loocv(loocv_results)     # Génère le graphique 2
        viz.plot_loocv_mse(loocv_results, results)  # Génère le graphique 3