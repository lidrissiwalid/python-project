import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

# 1. Charger les données réelles et configurer les métadonnées
real_file = 'C:\\Users\\walid\\OneDrive\\Desktop\\walid\\Project\\dev\\python_inpt_project\\original_dataset.csv'
print(f"Chargement du dataset original : {real_file}...")
real_data = pd.read_csv(real_file)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
metadata.update_column(column_name='Label', sdtype='categorical')

# 2. Définir les fichiers synthétiques à évaluer
synthetic_files = {
    "CTGAN": "C:\\Users\\walid\\OneDrive\\Desktop\\walid\\Project\\dev\\python_inpt_project\\ctgan_augmented_dataset.csv",
    "TVAE": "C:\\Users\\walid\\OneDrive\\Desktop\\walid\\Project\\dev\\python_inpt_project\\tvae_augmented_dataset.csv",
    "TGAN": "C:\\Users\\walid\\OneDrive\\Desktop\\walid\\Project\\dev\\python_inpt_project\\tgan_augmented_dataset.csv"
}

# 3. Boucle d'évaluation
results = {}

for model_name, file_path in synthetic_files.items():
    print(f"\n========================================")
    print(f"Évaluation du dataset généré par {model_name}")
    print(f"========================================")
    
    try:
        # Charger les données générées
        synthetic_data = pd.read_csv(file_path)
        
        # Lancer l'évaluation
        quality_report = evaluate_quality(
            real_data,
            synthetic_data,
            metadata
        )
        
        # Récupérer et stocker le score
        score = quality_report.get_score() * 100
        results[model_name] = score
        print(f"\n=> Score global pour {model_name} : {score:.2f}%")
        
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} est introuvable. A-t-il bien été généré ?")
    except Exception as e:
        print(f"Erreur inattendue lors de l'évaluation de {model_name} : {e}")

# 4. Affichage du classement final
print("\n" + "="*40)
print("CLASSEMENT FINAL DES GÉNÉRATEURS (Fidélité Statistique)")
print("="*40)
# Trier les résultats du meilleur au moins bon
sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
for rank, (model, score) in enumerate(sorted_results, 1):
    print(f"{rank}. {model} : {score:.2f}%")
