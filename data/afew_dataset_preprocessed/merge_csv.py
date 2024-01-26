import pandas as pd

# Carica i file CSV
csv1 = pd.read_csv('./afew_0_200.csv')
csv2 = pd.read_csv('./afew_200_400.csv')
csv3 = pd.read_csv('./afew_400_600.csv')

# Combina i file
combined_csv = pd.concat([csv1, csv2, csv3])
len(combined_csv)

# Salva il file combinato
combined_csv.to_csv('afew_0_600.csv', index=False)
