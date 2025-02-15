import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, classification_report
import torch

# Carregar os dados tratados
df = pd.read_csv("../data/cs-training-processed-final.csv")

# Separar features e target
X = df.drop(columns=["SeriousDlqin2yrs"]).values
y = df["SeriousDlqin2yrs"].values

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ajustar pesos para classes desbalanceadas
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Criar e configurar o modelo TabNet com ajustes de estabilidade
model = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr': 1e-3},  # Reduzindo ainda mais a taxa de aprendizado
    scheduler_params={"step_size":15, "gamma":0.7},  # Ajustando a taxa de decaimento do learning rate
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',  # Melhor que softmax para seleção de features
    lambda_sparse=1e-3,  # Maior regularização para evitar overfitting
    gamma=1.0  # Reduzindo o impacto do balanceamento interno
)

# Treinar o modelo com pesos ajustados
print("Treinando o modelo TabNet com ajustes otimizados...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_name=["valid"],
    eval_metric=["auc"],
    batch_size=1024,
    virtual_batch_size=128,
    max_epochs=1000,  # Permitindo mais épocas para melhor aprendizado
    patience=20,  # Aumentando paciência para evitar early stopping prematuro
    num_workers=0,
    drop_last=False,
    weights=sample_weights  # Aplicando pesos às classes
)

# Fazer previsões
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# Avaliação do modelo
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC-ROC Score:", auc_score)
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, zero_division=1))

# Salvar o modelo treinado
model.save_model("../models/tabnet_model.zip")
print("Modelo salvo em ../models/tabnet_model.zip")
