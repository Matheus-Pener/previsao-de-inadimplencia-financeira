import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Carregar os dados
df = pd.read_csv("cs-training-processed-final.csv")

# 1. Remover coluna irrelevante
df.drop(columns=["Unnamed: 0"], inplace=True)

# 2. Tratamento de valores ausentes
imputer = SimpleImputer(strategy="median")
df["MonthlyIncome"] = imputer.fit_transform(df[["MonthlyIncome"]])
df["NumberOfDependents"] = imputer.fit_transform(df[["NumberOfDependents"]])

# 3. Correção de outliers
df = df[df["age"] >= 18]  # Remover idades inválidas

# Filtrar valores absurdos em RevolvingUtilizationOfUnsecuredLines (máximo razoável = 1.5)
df = df[df["RevolvingUtilizationOfUnsecuredLines"] <= 1.5]

# 4. Normalização dos dados (MinMaxScaler)
scaler = MinMaxScaler()
features = df.drop(columns=["SeriousDlqin2yrs"])  # Variável alvo não entra na normalização
df_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
df_scaled["SeriousDlqin2yrs"] = df["SeriousDlqin2yrs"].values  # Recolocar a variável alvo

# 5. Balanceamento das classes (SMOTE)
X = df_scaled.drop(columns=["SeriousDlqin2yrs"])
y = df_scaled["SeriousDlqin2yrs"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Recriar o dataframe após o balanceamento
df_final = pd.DataFrame(X_resampled, columns=X.columns)
df_final["SeriousDlqin2yrs"] = y_resampled

# Salvar dataset tratado
df_final.to_csv("cs-training-processed.csv", index=False)

print("✅ Tratamento de dados concluído. Arquivo salvo como 'cs-training-processed.csv'.")