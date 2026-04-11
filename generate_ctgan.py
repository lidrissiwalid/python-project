import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# 1. Chargement des données originales
print("Chargement du dataset original...")
# Attention aux espaces dans le nom exact de votre fichier
file_name = 'Ransomware and Goodware  File API Dataset.csv'
df = pd.read_csv(file_name)

# 2. Configuration des métadonnées (Crucial pour les GANs)
print("Analyse de la structure des données...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# On s'assure que le modèle comprend que 'Label' n'est pas un nombre
# mais bien une catégorie (Ransomware ou Goodware)
metadata.update_column(column_name='Label', sdtype='categorical')

# 3. Initialisation et Entraînement de CTGAN
print("Entraînement de CTGAN en cours (cela peut prendre quelques minutes)...")
# On limite à 50 epochs pour ce premier test de validation
synthesizer = CTGANSynthesizer(metadata, epochs=50) 
synthesizer.fit(df)

# 4. Génération de la donnée synthétique
print("Génération de 1000 nouvelles lignes d'appels d'API...")
synthetic_data = synthesizer.sample(num_rows=1000)

# Comme on compte des appels d'API, on s'assure de ne pas avoir de valeurs négatives
num_cols = synthetic_data.select_dtypes(include=['number']).columns
synthetic_data[num_cols] = synthetic_data[num_cols].clip(lower=0).round().astype(int)

# 5. Sauvegarde
output_file = 'ctgan_augmented_dataset.csv'
synthetic_data.to_csv(output_file, index=False)
print(f"Succès ! Le dataset synthétique a été sauvegardé sous : {output_file}")
