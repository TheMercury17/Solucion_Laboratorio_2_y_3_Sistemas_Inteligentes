# Solución Guía de Laboratorio 2 y 3: Sistemas Inteligentes
## Regresión Multivariada, Álgebra Lineal y Regresión como Proyección
### Caso de Estudio Agroindustrial Transversal: AgroSwarm S.A.S.

**Universidad Sergio Arboleda**  
**Escuela de Ingeniería**  
**Programa de Ciencias de la Computación e Inteligencia Artificial**  
**Docente Titular:** Ph.D. Ricardo Andrés Fonseca Perdomo  
**Semestre:** 2026-II  

---

### Grupo de Trabajo: Edeméridos (Grupo 5)
* **Andrés Sebastián Coral Vallejo** — Ciencias de la Computación e Inteligencia Artificial
* **Paula Alejandra Ortiz** — Ciencias de la Computación e Inteligencia Artificial

---

### Descripción del Repositorio
Este repositorio contiene la solución completa, rigurosa y paso a paso para los **Laboratorios 2 y 3 (Temas 4 y 5)** de la asignatura **Sistemas Inteligentes**:
* **Tema 4:** Regresión Matricial Multivariable (Voltaje y Temperatura), Diagnóstico de Varianza y Sesgo (Polinomios de Grado 1, 3 y 10), Optimización Bio-Inspirada con Enjambre de Partículas (PSO), Recomendación de Ingeniería para Sistemas Embebidos (Microcontroladores de 8 bits) y Reto de Extensión de Regresión Ridge ($) con PSO.
* **Tema 5:** Álgebra Lineal y Detección de Redundancia de Sensores con Rango Matricial ($\text{rango}=2$), Caracterización de Covarianza con Valores y Vectores Propios (PCA y Elipse de Dispersión Colapsada), Verificación Automática de Control de Calidad (QA) por Ortogonalidad de Residuos ($\mathbf{X}^\top \mathbf{e} = \mathbf{0}$) y Reto de Extensión de Coeficiente de Determinación ^2$ derivado de la Proyección Ortogonal sobre $\operatorname{col}(\mathbf{X})$.

---

### Documento Principal Entregable
El informe técnico completo de 33 páginas, diseñado con estética visual de interfaz geométrica moderna (paleta de colorimetría azul noche, cobalto UI, cian neón, dorado y fondo blanco) utilizando las fuentes oficiales Faktos, Fontsona 5 Royal y Cambria Math, se encuentra disponible en formato PDF:

📄 **[Descargar Documento PDF: Solucion_Laboratorio_2_y_3_Sistemas_Inteligentes.pdf](./Solucion_Laboratorio_2_y_3_Sistemas_Inteligentes.pdf)**

---

### Estructura de Archivos
`	ext
├── Solucion_Laboratorio_2_y_3_Sistemas_Inteligentes.pdf  # Informe técnico completo compilado (33 páginas)
├── Solucion_Laboratorio_2_y_3_Sistemas_Inteligentes.tex  # Código fuente en LaTeX con diseño Persona 3 Reload UI
├── README.md                                             # Descripción general del repositorio
├── calcular_metricas_completas.py                        # Script matemático y computacional con todos los cálculos
├── generar_graficas.py                                   # Script generador de figuras de alta resolución (300 DPI)
├── sensor_humedad_temp.csv                              # Dataset ampliado para Tema 4 (Voltaje, Temp, Humedad)
├── tres_sensores.csv                                    # Dataset de instrumentación para Tema 5 (3 sensores)
├── fig_tarea2_sesgo_varianza.png                        # Gráfica de diagnóstico de sesgo y varianza
├── fig_tarea3_convergencia_pso.png                      # Curva de convergencia logarítmica de PSO vs MCO
├── fig_reto_ridge_coeficientes.png                      # Trayectoria de coeficientes en regularización Ridge
├── fig_tema5_dispersion_autovectores.png                # Diagrama de dispersión y direcciones propias (PCA)
└── fig_tema5_ortogonalidad.png                          # Validación automática de control de calidad (QA)
`
