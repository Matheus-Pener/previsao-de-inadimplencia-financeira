import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# Carregar o modelo treinado
modelo = TabNetClassifier()
modelo.load_model("../models/tabnet_best_model.zip")

# Carregar os dados de teste (20% que deixamos de fora no treinamento)
dados_teste = pd.read_csv("../data/dados_teste.csv")

# Separar features e target
X_teste = dados_teste.drop(columns=["SeriousDlqin2yrs"]).values
y_true = dados_teste["SeriousDlqin2yrs"].values  # Valores reais de inadimplência

# Fazer previsões
probs = modelo.predict_proba(X_teste)[:, 1]  # Probabilidade de inadimplência

# Definir um threshold para classificar como inadimplente ou não
threshold = 0.6
previsoes = (probs >= threshold).astype(int)

# Adicionar as previsões e probabilidades ao DataFrame original
dados_teste["Probabilidade_Inadimplente"] = probs
dados_teste["Previsao"] = previsoes

# Exibir algumas previsões
print("Previsões nos 20% de teste:")
print(dados_teste.head(10))

# Salvar os resultados em formato XLSX
dados_teste.to_excel("../data/previsoes_teste.xlsx", index=False, engine="openpyxl")
print("✅ Resultados salvos em ../data/previsoes_teste.xlsx")
