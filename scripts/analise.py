import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Carregar os resultados das previsões
df_previsoes = pd.read_excel("previsoes_teste.xlsx")

# Obter valores reais e previstos
y_true = df_previsoes["valor_real"]
y_pred = df_previsoes["Previsao"]

# 1. Calcular a acurácia
acuracia = accuracy_score(y_true, y_pred)

# 2. Criar matriz de confusão
matriz_confusao = confusion_matrix(y_true, y_pred)

# 3. Relatório detalhado (precisão, recall, F1-score)
relatorio_classificacao = classification_report(y_true, y_pred)

# Exibir os resultados
print(f"Acurácia do modelo: {acuracia:.4%}")
print("\nMatriz de Confusão:")
print(matriz_confusao)
print("\nRelatório de Classificação:")
print(relatorio_classificacao)

# 📈 4. Visualizar a distribuição das probabilidades previstas
plt.figure(figsize=(8, 5))
sns.histplot(df_previsoes["Probabilidade_Inadimplente"], bins=50, kde=True)
plt.axvline(0.6, color='r', linestyle='--', label="Threshold = 0.6")
plt.title("Distribuição das Probabilidades de Inadimplência")
plt.xlabel("Probabilidade de Inadimplência")
plt.ylabel("Frequência")
plt.legend()
plt.show()

# 📈 3. Gráfico - Matriz de Confusão Visual
plt.figure(figsize=(6, 5))
sns.heatmap(matriz_confusao, annot=True, fmt="d", cmap="Blues", xticklabels=["Adimplente", "Inadimplente"], yticklabels=["Adimplente", "Inadimplente"])
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# Definir o threshold
threshold = 0.6

# Carregar os resultados das previsões
df_previsoes = pd.read_excel("previsoes_teste.xlsx")

# Criar gráfico de distribuição de probabilidades por classe
plt.figure(figsize=(8, 5))
sns.kdeplot(df_previsoes[df_previsoes["valor_real"] == 0]["Probabilidade_Inadimplente"], label="Adimplente", fill=True)
sns.kdeplot(df_previsoes[df_previsoes["valor_real"] == 1]["Probabilidade_Inadimplente"], label="Inadimplente", fill=True)
plt.axvline(threshold, color='r', linestyle='--', label=f"Threshold = {threshold}")
plt.title("Distribuição das Probabilidades por Classe")
plt.xlabel("Probabilidade de Inadimplência")
plt.ylabel("Densidade")
plt.legend()
plt.show()

# Criar categorias de faixas salariais com base nos quartis
df_previsoes["Faixa_Salarial"] = pd.cut(
    df_previsoes["MonthlyIncome"],
    bins=[0, 0.0012, 0.0018, 0.0023, 0.005, 0.01, 0.02, df_previsoes["MonthlyIncome"].max()],
    labels=["Muito Baixa", "Baixa", "Média-Baixa", "Média", "Alta", "Muito Alta", "Extrema"]
)
# Criar boxplot para visualizar a relação entre faixa salarial e probabilidade de inadimplência
plt.figure(figsize=(8, 5))
sns.boxplot(x="Faixa_Salarial", y="Probabilidade_Inadimplente", data=df_previsoes)
plt.title("Probabilidade de Inadimplência por Faixa Salarial")
plt.xlabel("Faixa Salarial (Normalizada)")
plt.ylabel("Probabilidade de Inadimplência")
plt.xticks(rotation=45)
plt.show()
