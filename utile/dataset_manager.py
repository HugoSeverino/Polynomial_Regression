import pandas as pd

class DatasetManager:
    """Gestion du dataset : chargement, sélection des colonnes et affichage des informations."""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_data()
    
    def load_data(self):
        """Charge le fichier CSV dans un DataFrame."""
        try:
            df = pd.read_csv(self.file_path)
            print("\n✅ Fichier chargé avec succès.")
            return df
        except FileNotFoundError:
            print("\n❌ Erreur : Fichier introuvable.")
            return None

    def display_columns(self):
        """Affiche les colonnes disponibles."""
        if self.df is not None:
            print("\n📊 Colonnes disponibles :")
            print(self.df.columns.tolist())

    def select_columns(self, x_col, y_col):
        """Sélectionne les colonnes spécifiées et retourne un DataFrame filtré."""
        if x_col in self.df.columns and y_col in self.df.columns:
            df_selected = self.df[[x_col, y_col]].sort_values(by=x_col)
            print(f"\n✅ Dataset sélectionné avec {len(df_selected)} points.")
            return df_selected
        else:
            print("\n❌ Erreur : Colonnes invalides.")
            return None
