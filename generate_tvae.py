import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer

# 1. Chargement
df = pd.read_csv('Ransomware and Goodware  File API Dataset.csv')

# 2. Métadonnées
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
metadata.update_column(column_name='Label', sdtype='categorical')

# 3. Entraînement TVAE (Utilise des auto-encodeurs variationnels)
print("Entraînement de TVAE en cours...")
synthesizer = TVAESynthesizer(metadata, epochs=50)
synthesizer.fit(df)

# 4. Génération et nettoyage
print("Génération des données TVAE...")
synthetic_data = synthesizer.sample(num_rows=1000)
num_cols = synthetic_data.select_dtypes(include=['number']).columns
synthetic_data[num_cols] = synthetic_data[num_cols].clip(lower=0).round().astype(int)

# 5. Sauvegarde
synthetic_data.to_csv('tvae_augmented_dataset.csv', index=False)
print("Succès ! Dataset TVAE sauvegardé.")
