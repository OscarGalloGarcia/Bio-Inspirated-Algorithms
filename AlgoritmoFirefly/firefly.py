# Algoritmo Firefly (Xin-She Yang, 2008)
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


def firefly(funcion_objetivo, tam_poblacion, dimension, limites,
            max_iter, alpha, beta0, gamma, maximizar=False):
    x = [[random.uniform(limites[0], limites[1]) for _ in range(dimension)]
         for _ in range(tam_poblacion)]

    fitness = [funcion_objetivo(xi) for xi in x]

    for _ in range(max_iter):
        for i in range(tam_poblacion):
            for j in range(tam_poblacion):
                mejor_j = fitness[j] > fitness[i] if maximizar else fitness[j] < fitness[i]
                if mejor_j:
                    r2 = sum((x[i][d] - x[j][d]) ** 2 for d in range(dimension))
                    beta = beta0 * math.exp(-gamma * r2)
                    for d in range(dimension):
                        x[i][d] += beta * (x[j][d] - x[i][d]) + alpha * (random.random() - 0.5)
                        x[i][d] = max(limites[0], min(limites[1], x[i][d]))
                    fitness[i] = funcion_objetivo(x[i])

    idx = fitness.index(max(fitness)) if maximizar else fitness.index(min(fitness))
    return x[idx]


def esfera(x):
    return sum(xi * xi for xi in x)


def rastrigin(x):
    n = len(x)
    return 10 * n + sum(xi * xi - 10 * math.cos(2 * math.pi * xi) for xi in x)


def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def ackley(x):
    n = len(x)
    s1 = sum(xi * xi for xi in x)
    s2 = sum(math.cos(2 * math.pi * xi) for xi in x)
    return -20 * math.exp(-0.2 * math.sqrt(s1 / n)) - math.exp(s2 / n) + 20 + math.e


def neg_esfera(x): return -esfera(x)
def neg_rastrigin(x): return -rastrigin(x)
def neg_ackley(x): return -ackley(x)


def seno_coseno(x):
    return math.sin(x[0]) * math.cos(x[1])


def easom(x):
    return math.cos(x[0]) * math.cos(x[1]) * math.exp(
        -((x[0] - math.pi) ** 2) - ((x[1] - math.pi) ** 2))


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
    tam_pob, max_it = (40, 200) if dim <= 2 else (80, 500)
    sol = firefly(funcion, tam_pob, dim, limites, max_it, 0.2, 1.0, 1.0, maximizar)
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
    correr(esfera,     2, [-5, 5],       False, 'Esfera 2D',     'FA_min_esfera.png')
    correr(rastrigin,  2, [-5.12, 5.12], False, 'Rastrigin 2D',  'FA_min_rastrigin.png')
    correr(rosenbrock, 2, [-2, 2],       False, 'Rosenbrock 2D', 'FA_min_rosenbrock.png')

    print(' MAXIMIZACION 2D ')
    correr(neg_esfera,  2, [-5, 5],             True, 'Esfera negativa 2D', 'FA_max_neg_esfera.png')
    correr(seno_coseno, 2, [-math.pi, math.pi], True, 'sin(x1)*cos(x2)',   'FA_max_seno_coseno.png')
    correr(easom,       2, [0, 6],              True, 'Easom 2D',           'FA_max_easom.png')

    print(' MINIMIZACION N-DIM ')
    n = 4
    correr(esfera,    n, [-5, 5],       False, 'Esfera 4D')
    correr(rastrigin, n, [-5.12, 5.12], False, 'Rastrigin 4D')
    correr(ackley,    n, [-32, 32],     False, 'Ackley 4D')

    print(' MAXIMIZACION N-DIM ')
    correr(neg_esfera,    n, [-5, 5],       True, '-Esfera 4D')
    correr(neg_rastrigin, n, [-5.12, 5.12], True, '-Rastrigin 4D')
    correr(neg_ackley,    n, [-32, 32],     True, '-Ackley 4D')


if __name__ == '__main__':
    main()
