import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Cargar datos
df_temp = pd.read_csv('sensor_humedad_temp.csv')
v = df_temp['voltaje'].values
temp = df_temp['temperatura'].values
y = df_temp['humedad_referencia'].values
x1_tr, x1_te, x2_tr, x2_te, y_tr, y_te = train_test_split(v, temp, y, test_size=0.12, random_state=42)

# FIGURA 4: Tema 5 Dispersión Voltaje vs Conductividad y Autovectores
df_tres = pd.read_csv('tres_sensores.csv')
v_sens = df_tres['voltaje'].values
c_sens = df_tres['conductividad'].values

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.scatter(v_sens, c_sens, color='#0055B9', alpha=0.45, s=22, edgecolors='none', label='Mediciones de Campo (1000 puntos)')

mean_v = float(np.mean(v_sens))
mean_c = float(np.mean(c_sens))
cov = np.cov(v_sens, c_sens)
eigvals, eigvecs = np.linalg.eig(cov)

# Tomar parte real
eigvals = np.real(eigvals)
eigvecs = np.real(eigvecs)

# Ordenar por autovalor decreciente
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Escalar flechas por 2 * sqrt(lambda) (2 desviaciones estándar)
scale1 = float(2.0 * np.sqrt(eigvals[0]))
v1 = eigvecs[:, 0] * scale1
scale2 = float(2.0 * np.sqrt(max(0.0, eigvals[1])))
if scale2 < 0.1:
    scale2 = 0.35
v2 = eigvecs[:, 1] * scale2

ax.arrow(mean_v, mean_c, float(v1[0]), float(v1[1]), head_width=0.08, head_length=0.1, fc='#FFC800', ec='#FFC800', linewidth=2.5,
         label=r'Dirección Propia 1 ($\lambda_1 = ' + f'{eigvals[0]:.3f}' + r'$)', zorder=5)
ax.arrow(mean_v, mean_c, float(v2[0]), float(v2[1]), head_width=0.08, head_length=0.1, fc='#00D2FF', ec='#00D2FF', linewidth=2.5,
         label=r'Dirección Propia 2 ($\lambda_2 = ' + f'{eigvals[1]:.2e}' + r'$)', zorder=5)
ax.scatter([mean_v], [mean_c], color='#0B1B3D', s=80, marker='X', zorder=6, label=f'Centroide (V={mean_v:.2f}, C={mean_c:.2f})')

ax.set_xlabel('Voltaje Sensor Principal (V)', fontweight='bold', color='#0B1B3D')
ax.set_ylabel('Conductividad Eléctrica Aparente (EC_a)', fontweight='bold', color='#0B1B3D')
ax.set_title('CARACTERIZACIÓN DE REDUNDANCIA POR COMPONENTES PRINCIPALES', fontweight='bold', color='#0B1B3D', pad=12)
ax.legend(frameon=True, facecolor='white', edgecolor='#0055B9', loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('fig_tema5_dispersion_autovectores.png')
plt.close()
print('Fig 4 guardada')

# FIGURA 5: Verificación de Ortogonalidad de Residuos (QA)
fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
X_qa = np.column_stack([np.ones(len(v)), v, temp])
w_opt, _, _, _ = np.linalg.lstsq(X_qa, y, rcond=None)
e_opt = y - X_qa @ w_opt
dots_opt = np.abs(X_qa.T @ e_opt)

w_bad = w_opt * 1.05
e_bad = y - X_qa @ w_bad
dots_bad = np.abs(X_qa.T @ e_bad)

vars_labels = ['Col 0: Intercepto (1)', 'Col 1: Voltaje (V)', 'Col 2: Temp (T)']
x_indices = np.arange(len(vars_labels))
b_width = 0.35

ax.bar(x_indices - b_width/2, dots_opt, b_width, label=r'Ajuste MCO Óptimo ($|X_j^\top e|$)', color='#0055B9', edgecolor='#00D2FF', linewidth=1.2)
ax.bar(x_indices + b_width/2, dots_bad, b_width, label=r'Ajuste Erróneo $+5\%$ ($|X_j^\top e|$)', color='#FFC800', edgecolor='#0B1B3D', linewidth=1.2)

ax.set_yscale('log')
ax.set_ylabel('Magnitud del Producto Punto $|X_j^T e|$ (Escala Log)', fontweight='bold', color='#0B1B3D')
ax.set_title('PRUEBA AUTOMÁTICA DE CONTROL DE CALIDAD (QA) POR ORTOGONALIDAD', fontweight='bold', color='#0B1B3D', pad=12)
ax.set_xticks(x_indices)
ax.set_xticklabels(vars_labels, fontweight='bold')
ax.axhline(1e-6, color='red', linestyle='--', linewidth=1.5, label=r'Umbral de Tolerancia $\tau = 10^{-6}$')
ax.legend(frameon=True, facecolor='white', edgecolor='#0055B9')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('fig_tema5_ortogonalidad.png')
plt.close()
print('Fig 5 guardada exitosamente')
