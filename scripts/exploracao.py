import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv("cs-training-processed-final.csv")

# Exibir as primeiras 5 linhas do dataset
print("\n🔹 Primeiras linhas do dataset:")
print(df.head())

# Exibir informações sobre colunas e valores nulos
print("\n🔹 Informações gerais do dataset:")
print(df.info())

# Estatísticas básicas do dataset
print("\n🔹 Estatísticas básicas do dataset:")
print(df.describe())

# Contagem de valores nulos por coluna
print("\n🔹 Valores nulos por coluna:")
print(df.isnull().sum())
