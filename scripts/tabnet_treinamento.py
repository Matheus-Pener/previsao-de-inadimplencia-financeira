import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from skopt import gp_minimize
from skopt.space import Real, Integer
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

# Definir o espaço de busca para Bayesian Optimization
search_space = [
    Real(1e-4, 1e-2, "log-uniform"),  # Learning rate
    Real(1e-4, 1e-2, "log-uniform"),  # Lambda sparse
    Real(0.5, 2.0, "uniform"),        # Gamma
    Integer(512, 2048)                 # Batch size
]

# Função de avaliação para o otimizador Bayesian
def objective(params):
    lr, lambda_sparse, gamma, batch_size = params
    print(f"Testando: lr={lr}, lambda_sparse={lambda_sparse}, gamma={gamma}, batch_size={batch_size}")
    
    model = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params={'lr': lr},
        scheduler_params={"step_size":15, "gamma":0.7},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        lambda_sparse=lambda_sparse,
        gamma=gamma
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_name=["valid"],
        eval_metric=["auc"],
        batch_size=int(batch_size),
        virtual_batch_size=128,
        max_epochs=500,
        patience=20,
        num_workers=0,
        drop_last=False,
        weights=sample_weights
    )
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC Score: {auc_score}")
    
    return -auc_score  # Minimização (invertido porque queremos maximizar AUC)

# Rodar otimização bayesiana
print("Iniciando otimização bayesiana de hiperparâmetros...")
res = gp_minimize(objective, search_space, n_calls=20, random_state=42)

# Melhor combinação de hiperparâmetros encontrada
best_params = res.x
best_auc = -res.fun
print(f"Melhores hiperparâmetros: lr={best_params[0]}, lambda_sparse={best_params[1]}, gamma={best_params[2]}, batch_size={int(best_params[3])}")
print(f"Melhor AUC-ROC: {best_auc}")

# Treinar o modelo final com os melhores hiperparâmetros
final_model = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr': best_params[0]},
    scheduler_params={"step_size":15, "gamma":0.7},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',
    lambda_sparse=best_params[1],
    gamma=best_params[2]
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_name=["valid"],
    eval_metric=["auc"],
    batch_size=int(best_params[3]),
    virtual_batch_size=128,
    max_epochs=500,
    patience=20,
    num_workers=0,
    drop_last=False,
    weights=sample_weights
)

# Salvar o melhor modelo
final_model.save_model("../models/tabnet_best_model.zip")
print("Melhor modelo salvo em ../models/tabnet_best_model.zip")