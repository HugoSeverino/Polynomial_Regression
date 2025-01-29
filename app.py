import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le fichier
df = pd.read_csv("data/Position_salaries.csv")

# Afficher les colonnes disponibles
print("Colonnes disponibles dans le dataset :")
print(df.columns.tolist())

# Demander à l'utilisateur de choisir les colonnes
x_col = input("\nChoisissez la colonne pour l'axe X : ")
y_col = input("Choisissez la colonne pour l'axe Y : ")

# Vérifier si les colonnes existent
if x_col in df.columns and y_col in df.columns:
    # Créer un nouveau dataset avec les colonnes choisies
    df_selected = df[[x_col, y_col]]
    
    # Afficher le dataset créé
    print("\nDataset créé avec les colonnes sélectionnées :")
    print(df_selected.head())

    # Sauvegarder le nouveau dataset si nécessaire
    df_selected.to_csv("data/dataset_selection.csv", index=False)

    # Visualiser les données avec Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_selected[x_col], y=df_selected[y_col])

    # Personnalisation du graphique
    plt.title(f"Relation entre {x_col} et {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()
else:
    print("\n❌ Erreur : Une des colonnes choisies n'existe pas dans le dataset.")

