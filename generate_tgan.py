import pandas as pd
from tgan.model import TGANModel

print("Chargement du dataset pour TGAN...")
df = pd.read_csv('Ransomware and Goodware  File API Dataset.csv')

# TGAN a besoin de savoir quelles colonnes sont des variables continues.
# Le Label est à l'index 0 (catégoriel). On indique que le reste (index 1 à 63) est continu/numérique.
continuous_columns = list(range(1, len(df.columns)))

# Initialisation et entraînement (On garde un max_epoch bas pour tester d'abord)
print("Entraînement de TGAN en cours (peut être long)...")
tgan = TGANModel(continuous_columns, max_epoch=5, steps_per_epoch=100)
tgan.fit(df)

# Génération et nettoyage
print("Génération des données TGAN...")
synthetic_data = tgan.sample(1000)

num_cols = synthetic_data.select_dtypes(include=['number']).columns
synthetic_data[num_cols] = synthetic_data[num_cols].clip(lower=0).round().astype(int)

synthetic_data.to_csv('tgan_augmented_dataset.csv', index=False)
print("Succès ! Dataset TGAN sauvegardé.")
