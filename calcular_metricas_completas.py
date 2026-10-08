import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# ==============================================================================
# TEMA 4: REGRESIÓN MATRICIAL, VARIANZA Y SESGO
# ==============================================================================

# Cargar dataset
df_t4 = pd.read_csv('sensor_humedad_temp.csv')
v = df_t4['voltaje'].values
temp = df_t4['temperatura'].values
y = df_t4['humedad_referencia'].values
n_total = len(v)

# Train/Test Split (12% test, 88% train, seed 42)
indices = np.arange(n_total)
idx_tr, idx_te = train_test_split(indices, test_size=0.12, random_state=42)

v_tr, v_te = v[idx_tr], v[idx_te]
temp_tr, temp_te = temp[idx_tr], temp[idx_te]
y_tr, y_te = y[idx_tr], y[idx_te]

# Tarea 1: Modelo Matricial Bivariado
X_tr = np.column_stack([np.ones(len(v_tr)), v_tr, temp_tr])
X_te = np.column_stack([np.ones(len(v_te)), v_te, temp_te])

w_mat, resids, rank_X, s_X = np.linalg.lstsq(X_tr, y_tr, rcond=None)
b_mat, w1_mat, w2_mat = w_mat[0], w_mat[1], w_mat[2]

y_pred_te_mat = X_te @ w_mat
y_pred_tr_mat = X_tr @ w_mat

rmse_te_mat = float(np.sqrt(np.mean((y_te - y_pred_te_mat)**2)))
rmse_tr_mat = float(np.sqrt(np.mean((y_tr - y_pred_tr_mat)**2)))
mae_te_mat  = float(np.mean(np.abs(y_te - y_pred_te_mat)))

# Modelo univariado (Semana 3) sobre los mismos datos
p_univ = np.polyfit(v_tr, y_tr, 1)
y_pred_te_univ = np.polyval(p_univ, v_te)
y_pred_tr_univ = np.polyval(p_univ, v_tr)
rmse_te_univ = float(np.sqrt(np.mean((y_te - y_pred_te_univ)**2)))
rmse_tr_univ = float(np.sqrt(np.mean((y_tr - y_pred_tr_univ)**2)))

# Tarea 2: Diagnóstico de Varianza y Sesgo (Polinomios 1, 3, 10 en Voltaje)
poly_results = {}
for deg in [1, 3, 10]:
    p_deg = np.polyfit(v_tr, y_tr, deg)
    y_pr_tr = np.polyval(p_deg, v_tr)
    y_pr_te = np.polyval(p_deg, v_te)
    r_tr = float(np.sqrt(np.mean((y_tr - y_pr_tr)**2)))
    r_te = float(np.sqrt(np.mean((y_te - y_pr_te)**2)))
    poly_results[f'grado_{deg}'] = {
        'rmse_train': r_tr,
        'rmse_test': r_te,
        'brecha': r_te - r_tr,
        'coeficientes': p_deg.tolist()
    }

# Tarea 3: PSO para regresión
def costo_particulas(P, x1, x2, y, lam=0.0):
    w1 = P[:, 0:1]
    w2 = P[:, 1:2]
    b  = P[:, 2:3]
    x1_row = x1.reshape(1, -1)
    x2_row = x2.reshape(1, -1)
    y_row  = y.reshape(1, -1)
    y_pred = w1 @ x1_row + w2 @ x2_row + b
    ecm = np.mean((y_pred - y_row)**2, axis=1)
    if lam > 0:
        return ecm + lam * (P[:, 0]**2 + P[:, 1]**2)
    return ecm

def pso_regresion(x1, x2, y, n_particulas=50, n_iter=300, lam=0.0, seed=42):
    np.random.seed(seed)
    lb = np.array([5.0, -2.0, -10.0])
    ub = np.array([25.0, 5.0, 30.0])
    P = np.random.uniform(lb, ub, (n_particulas, 3))
    V = np.random.uniform(-(ub-lb)*0.1, (ub-lb)*0.1, (n_particulas, 3))
    costos = costo_particulas(P, x1, x2, y, lam=lam)
    pbest_pos = P.copy()
    pbest_val = costos.copy()
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    w_inertia, c1, c2 = 0.7, 1.5, 1.5
    for it in range(n_iter):
        r1, r2 = np.random.rand(n_particulas, 3), np.random.rand(n_particulas, 3)
        V = w_inertia * V + c1 * r1 * (pbest_pos - P) + c2 * r2 * (gbest_pos - P)
        P = P + V
        costos = costo_particulas(P, x1, x2, y, lam=lam)
        mejores = costos < pbest_val
        pbest_pos[mejores] = P[mejores]
        pbest_val[mejores] = costos[mejores]
        min_idx = np.argmin(pbest_val)
        if pbest_val[min_idx] < gbest_val:
            gbest_val = pbest_val[min_idx]
            gbest_pos = pbest_pos[min_idx].copy()
    return float(gbest_pos[0]), float(gbest_pos[1]), float(gbest_pos[2]), float(gbest_val)

w1_pso, w2_pso, b_pso, ecm_pso_tr = pso_regresion(v_tr, temp_tr, y_tr, n_particulas=60, n_iter=350)
y_pred_te_pso = w1_pso * v_te + w2_pso * temp_te + b_pso
rmse_te_pso = float(np.sqrt(np.mean((y_te - y_pred_te_pso)**2)))

# Reto de Extensión Tema 4: Ridge con PSO
ridge_results = {}
for lam in [0.0, 0.1, 1.0, 10.0]:
    w1_r, w2_r, b_r, _ = pso_regresion(v_tr, temp_tr, y_tr, lam=lam, n_particulas=60, n_iter=350)
    y_pred_r = w1_r * v_te + w2_r * temp_te + b_r
    rmse_r = float(np.sqrt(np.mean((y_te - y_pred_r)**2)))
    norm_w = float(np.sqrt(w1_r**2 + w2_r**2))
    ridge_results[f'lambda_{lam}'] = {
        'w1': w1_r,
        'w2': w2_r,
        'b': b_r,
        'norm_w': norm_w,
        'rmse_test': rmse_r
    }

# ==============================================================================
# TEMA 5: ÁLGEBRA LINEAL Y REGRESIÓN COMO PROYECCIÓN
# ==============================================================================

df_t5 = pd.read_csv('tres_sensores.csv')
v5 = df_t5['voltaje'].values
temp5 = df_t5['temperatura'].values
cond5 = df_t5['conductividad'].values
y5 = df_t5['humedad_referencia'].values

# Tarea 1: Diagnóstico de Redundancia con Rango
X_sensores = np.column_stack([v5, temp5, cond5])
rango_X_sensores = int(np.linalg.matrix_rank(X_sensores))
corr_v_c = float(np.corrcoef(v5, cond5)[0, 1])
corr_v_t = float(np.corrcoef(v5, temp5)[0, 1])
corr_t_c = float(np.corrcoef(temp5, cond5)[0, 1])

# Tarea 2: Covarianza y Vectores/Valores Propios (2x2 entre V y C)
cov_vc = np.cov(v5, cond5)
eigvals_raw, eigvecs_raw = np.linalg.eig(cov_vc)
idx_sort = np.argsort(eigvals_raw)[::-1]
eigvals = np.real(eigvals_raw[idx_sort]).tolist()
eigvecs = np.real(eigvecs_raw[:, idx_sort]).tolist()

# Tarea 3: Verificación Automática de Calidad (QA) por Ortogonalidad
X_no_red = np.column_stack([np.ones(len(v5)), v5, temp5])
w_mco_t5, _, _, _ = np.linalg.lstsq(X_no_red, y5, rcond=None)

def verificar_ajuste_qa(X, y, w, tol=1e-6):
    e = y - X @ w
    dot_products = X.T @ e
    escala = np.linalg.norm(X, axis=0) * np.linalg.norm(y)
    criterio = np.abs(dot_products) < (tol * escala + 1e-12)
    es_valido = bool(np.all(criterio))
    
    # Reto extensión: R^2 geométrico por proyección ortogonal
    y_hat = X @ w
    y_bar = np.mean(y)
    norm_sc_reg = float(np.sum((y_hat - y_bar)**2))
    norm_sc_tot = float(np.sum((y - y_bar)**2))
    norm_sc_err = float(np.sum(e**2))
    r2_proj = float(norm_sc_reg / norm_sc_tot)
    
    return {
        'valido': es_valido,
        'dot_products': dot_products.tolist(),
        'residuos_norm': float(np.linalg.norm(e)),
        'r2': r2_proj,
        'stc': norm_sc_tot,
        'scr': norm_sc_reg,
        'sce': norm_sc_err
    }

qa_optimo = verificar_ajuste_qa(X_no_red, y5, w_mco_t5)
qa_malo   = verificar_ajuste_qa(X_no_red, y5, w_mco_t5 * 1.05)

resultados_completos = {
    'tema4': {
        'n_train': len(idx_tr),
        'n_test': len(idx_te),
        'modelo_univariado': {
            'w': float(p_univ[0]),
            'b': float(p_univ[1]),
            'rmse_train': rmse_tr_univ,
            'rmse_test': rmse_te_univ
        },
        'modelo_matricial_bivariado': {
            'b': float(b_mat),
            'w1_voltaje': float(w1_mat),
            'w2_temperatura': float(w2_mat),
            'rmse_train': rmse_tr_mat,
            'rmse_test': rmse_te_mat,
            'mae_test': mae_te_mat
        },
        'tarea2_polinomios': poly_results,
        'tarea3_pso': {
            'w1': w1_pso,
            'w2': w2_pso,
            'b': b_pso,
            'rmse_test': rmse_te_pso,
            'diff_w1': abs(w1_pso - w1_mat),
            'diff_w2': abs(w2_pso - w2_mat),
            'diff_b': abs(b_pso - b_mat),
            'diff_rmse': abs(rmse_te_pso - rmse_te_mat)
        },
        'reto_ridge': ridge_results
    },
    'tema5': {
        'tarea1_rango': {
            'rango': rango_X_sensores,
            'corr_vc': corr_v_c,
            'corr_vt': corr_v_t,
            'corr_tc': corr_t_c
        },
        'tarea2_eigen': {
            'matriz_covarianza': cov_vc.tolist(),
            'autovalores': eigvals,
            'autovectores': eigvecs,
            'ratio_autovalores': float(eigvals[0] / (eigvals[1] if eigvals[1] > 1e-12 else 1e-12))
        },
        'tarea3_qa': {
            'w_optimo': w_mco_t5.tolist(),
            'qa_optimo': qa_optimo,
            'qa_malo': qa_malo
        }
    }
}

with open('metricas_calculadas.json', 'w', encoding='utf-8') as f:
    json.dump(resultados_completos, f, indent=2)

print('Metricas calculadas y guardadas exitosamente en metricas_calculadas.json')
