# Algoritmo: Particle Swarm Optimization (PSO)
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


def pso(funcion_objetivo, tam_poblacion, dimension, limites,
        max_iter, w, c1, c2, maximizar=False):
    x = [[random.uniform(limites[d][0], limites[d][1]) for d in range(dimension)]
         for _ in range(tam_poblacion)]

    v = [[random.uniform(-(limites[d][1] - limites[d][0]),
                          (limites[d][1] - limites[d][0])) for d in range(dimension)]
         for _ in range(tam_poblacion)]

    y = [list(xi) for xi in x]
    f_y = [funcion_objetivo(yi) for yi in y]

    if maximizar:
        idx = f_y.index(max(f_y))
    else:
        idx = f_y.index(min(f_y))
    y_gorro = list(y[idx])
    f_y_gorro = f_y[idx]

    for _ in range(max_iter):
        for i in range(tam_poblacion):
            f_xi = funcion_objetivo(x[i])
            if maximizar:
                if f_xi > f_y[i]:
                    y[i] = list(x[i])
                    f_y[i] = f_xi
                if f_y[i] > f_y_gorro:
                    y_gorro = list(y[i])
                    f_y_gorro = f_y[i]
            else:
                if f_xi < f_y[i]:
                    y[i] = list(x[i])
                    f_y[i] = f_xi
                if f_y[i] < f_y_gorro:
                    y_gorro = list(y[i])
                    f_y_gorro = f_y[i]

        for i in range(tam_poblacion):
            for j in range(dimension):
                r1 = random.random()
                r2 = random.random()
                v[i][j] = (w * v[i][j]
                           + c1 * r1 * (y[i][j] - x[i][j])
                           + c2 * r2 * (y_gorro[j] - x[i][j]))
                x[i][j] = x[i][j] + v[i][j]
                x[i][j] = max(limites[j][0], min(limites[j][1], x[i][j]))

    return y_gorro


def esfera(x):
    return sum(xi * xi for xi in x)


def rastrigin(x):
    n = len(x)
    total = 10 * n
    for xi in x:
        total += xi * xi - 10 * math.cos(2 * math.pi * xi)
    return total


def rosenbrock(x):
    total = 0
    for i in range(len(x) - 1):
        total += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return total


def ackley(x):
    n = len(x)
    s1 = sum(xi * xi for xi in x)
    s2 = sum(math.cos(2 * math.pi * xi) for xi in x)
    return -20 * math.exp(-0.2 * math.sqrt(s1 / n)) - math.exp(s2 / n) + 20 + math.e


def neg_esfera(x):
    return -esfera(x)


def neg_rastrigin(x):
    return -rastrigin(x)


def neg_ackley(x):
    return -ackley(x)


def seno_coseno(x):
    return math.sin(x[0]) * math.cos(x[1])


def easom(x):
    return math.cos(x[0]) * math.cos(x[1]) * math.exp(
        -((x[0] - math.pi) ** 2) - ((x[1] - math.pi) ** 2))


def graficar_2d(funcion, limites, titulo, mejor, archivo):
    n = 80
    x = np.linspace(limites[0][0], limites[0][1], n)
    y = np.linspace(limites[1][0], limites[1][1], n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            Z[i, j] = funcion([X[i, j], Y[i, j]])

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
        tam_pob = 50
        max_it = 200
    else:
        tam_pob = 100
        max_it = 1000
    sol = pso(funcion, tam_pob, dim, limites, max_it, 0.7, 1.5, 1.5, maximizar)
    valor = funcion(sol)
    tipo = 'maximizacion' if maximizar else 'minimizacion'
    print('---', nombre, '(', tipo, ', dim =', dim, ')---')
    print('Mejor solucion:', [round(v, 5) for v in sol])
    print('f(mejor) =', valor)
    print()
    if archivo is not None and dim == 2:
        graficar_2d(funcion, limites, nombre, sol, archivo)
    return sol, valor


print('==================== MINIMIZACION 2D ====================')
correr(esfera,     2, [(-5, 5), (-5, 5)],                False, 'Esfera 2D',     'PSO_min_esfera.png')
correr(rastrigin,  2, [(-5.12, 5.12), (-5.12, 5.12)],   False, 'Rastrigin 2D',  'PSO_min_rastrigin.png')
correr(rosenbrock, 2, [(-2, 2), (-2, 2)],                False, 'Rosenbrock 2D', 'PSO_min_rosenbrock.png')

print('==================== MAXIMIZACION 2D ====================')
correr(neg_esfera,  2, [(-5, 5), (-5, 5)],                           True, 'Esfera negativa 2D', 'PSO_max_neg_esfera.png')
correr(seno_coseno, 2, [(-math.pi, math.pi), (-math.pi, math.pi)],   True, 'sin(x1)*cos(x2)',    'PSO_max_seno_coseno.png')
correr(easom,       2, [(0, 6), (0, 6)],                             True, 'Easom 2D',            'PSO_max_easom.png')

print('==================== MINIMIZACION N-DIM ====================')
n = 10
correr(esfera,    n, [(-5, 5)] * n,            False, 'Esfera 10D')
correr(rastrigin, n, [(-5.12, 5.12)] * n,      False, 'Rastrigin 10D')
correr(ackley,    n, [(-32, 32)] * n,          False, 'Ackley 10D')

print('==================== MAXIMIZACION N-DIM ====================')
correr(neg_esfera,    n, [(-5, 5)] * n,          True, '-Esfera 10D')
correr(neg_rastrigin, n, [(-5.12, 5.12)] * n,    True, '-Rastrigin 10D')
correr(neg_ackley,    n, [(-32, 32)] * n,        True, '-Ackley 10D')
