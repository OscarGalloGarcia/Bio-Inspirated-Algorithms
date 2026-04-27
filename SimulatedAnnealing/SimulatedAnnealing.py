# Algoritmo: Simulated Annealing N-Dimensional
# Oscar Alberto Gallo García

import matplotlib.pyplot as plot
import numpy as np


def simulated_annealing(funcion, dimension, tam_paso, T_inicial, T_min,
                         factor_enfriamiento, iter_maximas, limites, minimizar):

    lim_inf, lim_sup = limites
    signo = 1 if minimizar else -1

    x_actual = np.random.uniform(lim_inf, lim_sup, dimension)
    x_mejor  = x_actual.copy()

    T = T_inicial
    historial_T = []

    # Curva de la función (fija durante toda la animación)
    x_vals = np.linspace(lim_inf, lim_sup, 500)
    y_vals = [funcion(np.array([x])) for x in x_vals]

    plot.ion()
    _, (ax1, ax2) = plot.subplots(1, 2, figsize=(12, 4))

    while T > T_min:
        for _ in range(iter_maximas):

            x_nuevo = np.clip(x_actual + np.random.uniform(-tam_paso, tam_paso, dimension),
                              lim_inf, lim_sup)

            delta = signo * (funcion(x_nuevo) - funcion(x_actual))

            if delta < 0 or np.random.random() < np.exp(-delta / T):
                x_actual = x_nuevo

            if signo * funcion(x_actual) < signo * funcion(x_mejor):
                x_mejor = x_actual.copy()

            historial_T.append(T)

        # Actualizar gráfica al final de cada nivel de temperatura
        ax1.cla()
        ax1.plot(x_vals, y_vals, color="steelblue")
        ax1.scatter(x_actual[0], funcion(x_actual), color="orange", s=80, zorder=5, label="actual")
        ax1.scatter(x_mejor[0],  funcion(x_mejor),  color="red",    s=120, marker="*", zorder=6,
                    label=f"mejor = {x_mejor[0]:.3f}")
        ax1.set_xlabel("x"); ax1.set_ylabel("f(x)")
        ax1.set_title(f"Temperatura actual: {T:.4f}")
        ax1.legend()

        ax2.cla()
        ax2.plot(historial_T, color="tomato")
        ax2.set_xlabel("Iteración"); ax2.set_ylabel("Temperatura")
        ax2.set_title("Enfriamiento")

        plot.tight_layout()
        plot.pause(0.05)

        T *= factor_enfriamiento

    plot.ioff()
    plot.show()

    return x_mejor, historial_T


if __name__ == "__main__":

    np.random.seed(7)

    # Minimización:
    # f(x) = 10 + x² − 10·cos(2πx) 
    def rastrigin(v):
        return 10 + v[0]**2 - 10 * np.cos(2 * np.pi * v[0])

    x_opt, hist = simulated_annealing(
        funcion             = rastrigin,
        dimension           = 1,
        tam_paso            = 0.5,
        T_inicial           = 100.0,
        T_min               = 0.01,
        factor_enfriamiento = 0.95,
        iter_maximas        = 50,
        limites             = [-5.12, 5.12],
        minimizar           = True
    )
    print(f"Minimización | x* = {np.round(x_opt, 4)} | f(x*) = {rastrigin(x_opt):.6f}")

    # Maximización: 
    # máximo global en x=1 (altura 2), locales en x=-2, x=3, x=-1
    def gaussianas(v):
        x = v[0]
        return (2.0 * np.exp(-(x - 1)**2) +
                1.5 * np.exp(-(x + 2)**2) +
                1.0 * np.exp(-(x - 3)**2) +
                0.8 * np.exp(-(x + 1)**2))

    x_opt, hist = simulated_annealing(
        funcion             = gaussianas,
        dimension           = 1,
        tam_paso            = 0.4,
        T_inicial           = 5.0,
        T_min               = 0.001,
        factor_enfriamiento = 0.97,
        iter_maximas        = 50,
        limites             = [-5.0, 5.0],
        minimizar           = False
    )
    print(f"Maximización | x* = {np.round(x_opt, 4)} | f(x*) = {gaussianas(x_opt):.6f}")
