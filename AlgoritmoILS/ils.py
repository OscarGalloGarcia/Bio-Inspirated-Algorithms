# Algoritmo: Iterated Local Search (ILS)
# Oscar Alberto Gallo García
# MSc Robótica e Inteligencia Artificial

import math
import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(7)
np.random.seed(7)

os.makedirs('figs', exist_ok=True)


def busqueda_local(f, x, limites, paso, n_local, maximizar):
    x_best = list(x)
    f_best = f(x_best)
    n = len(x_best)
    for _ in range(n_local):
        x_new = [max(limites[0], min(limites[1], x_best[d] + random.uniform(-paso, paso)))
                 for d in range(n)]
        f_new = f(x_new)
        if f_new > f_best if maximizar else f_new < f_best:
            x_best = x_new
            f_best = f_new
    return x_best


def perturbacion(x, limites, fuerza):
    return [max(limites[0], min(limites[1], x[d] + random.uniform(-fuerza, fuerza)))
            for d in range(len(x))]


def ils(funcion_objetivo, dimension, limites, max_iter,
        paso, n_local, fuerza, maximizar=False):
    x = [random.uniform(limites[0], limites[1]) for _ in range(dimension)]
    x_best = busqueda_local(funcion_objetivo, x, limites, paso, n_local, maximizar)
    f_best = funcion_objetivo(x_best)

    for _ in range(max_iter):
        x_pert = perturbacion(x_best, limites, fuerza)
        x_new = busqueda_local(funcion_objetivo, x_pert, limites, paso, n_local, maximizar)
        f_new = funcion_objetivo(x_new)
        if f_new > f_best if maximizar else f_new < f_best:
            x_best = x_new
            f_best = f_new

    return x_best


def rastrigin(x):
    n = len(x)
    return 10 * n + sum(xi * xi - 10 * math.cos(2 * math.pi * xi) for xi in x)


def ackley(x):
    n = len(x)
    s1 = sum(xi * xi for xi in x)
    s2 = sum(math.cos(2 * math.pi * xi) for xi in x)
    return -20 * math.exp(-0.2 * math.sqrt(s1 / n)) - math.exp(s2 / n) + 20 + math.e


def neg_rastrigin(x): return -rastrigin(x)
def neg_ackley(x): return -ackley(x)


def graficar_2d(funcion_objetivo, limites, titulo, mejor, archivo):
    n = 80
    x = np.linspace(limites[0], limites[1], n)
    y = np.linspace(limites[0], limites[1], n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            Z[i, j] = funcion_objetivo([X[i, j], Y[i, j]])

    fig = plt.figure(figsize=(12, 4.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85)
    ax1.set_title(titulo + ' (superficie)')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x)')

    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(X, Y, Z, levels=30, cmap='viridis')
    plt.colorbar(cs, ax=ax2)
    ax2.plot(mejor[0], mejor[1], 'r*', markersize=18,
             markeredgecolor='white', label='mejor')
    ax2.legend()
    ax2.set_title(titulo + ' (contorno)')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')

    plt.tight_layout()
    plt.savefig('figs/' + archivo, dpi=130, bbox_inches='tight')
    plt.close()


def correr(funcion, dim, limites, maximizar, nombre, archivo=None):
    if dim <= 2:
        max_iter, n_local = 50, 500
    else:
        max_iter, n_local = 100, 1000
    sol = ils(funcion, dim, limites, max_iter, 0.1, n_local, 2.0, maximizar)
    valor = funcion(sol)
    tipo = 'maximizacion' if maximizar else 'minimizacion'
    print('---', nombre, '(', tipo, ', dim =', dim, ')---')
    print('Mejor solucion:', [round(v, 5) for v in sol])
    print('f(mejor) =', valor)
    print()
    if archivo is not None and dim == 2:
        graficar_2d(funcion, limites, nombre, sol, archivo)
    return sol, valor


def main():
    print(' MINIMIZACION 2D ')
    correr(rastrigin, 2, [-5.12, 5.12], False, 'Rastrigin 2D', 'ILS_min_rastrigin.png')

    print(' MAXIMIZACION 2D ')
    correr(neg_rastrigin, 2, [-5.12, 5.12], True, 'Rastrigin negativa 2D', 'ILS_max_rastrigin.png')

    print(' MINIMIZACION N-DIM ')
    n = 4
    correr(rastrigin, n, [-5.12, 5.12], False, 'Rastrigin 4D')
    correr(ackley,    n, [-32, 32],     False, 'Ackley 4D')

    print(' MAXIMIZACION N-DIM ')
    correr(neg_rastrigin, n, [-5.12, 5.12], True, '-Rastrigin 4D')
    correr(neg_ackley,    n, [-32, 32],     True, '-Ackley 4D')


if __name__ == '__main__':
    main()
