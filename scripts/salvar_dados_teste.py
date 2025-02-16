import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Carregar os dados tratados
df = pd.read_csv("../data/cs-training-processed-final.csv")

# Separar features e target
X = df.drop(columns=["SeriousDlqin2yrs"]).values
y = df["SeriousDlqin2yrs"].values

# Dividir em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Criar DataFrame com os 20% de teste
dados_teste = pd.DataFrame(X_test, columns=df.drop(columns=["SeriousDlqin2yrs"]).columns)
dados_teste["SeriousDlqin2yrs"] = y_test  # Adicionar a coluna alvo

# Salvar o arquivo
dados_teste.to_csv("../data/dados_teste.csv", index=False)
print("Arquivo de teste salvo em ../datsa/dados_teste.csv")
