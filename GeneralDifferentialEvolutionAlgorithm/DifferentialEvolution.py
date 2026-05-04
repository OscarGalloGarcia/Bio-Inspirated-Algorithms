import random


def evolucion_diferencial(funcion_objetivo, tam_poblacion, dimension, limites,
                          max_iter, F, Cr, maximizar=False):
    poblacion = []
    for _ in range(tam_poblacion):
        individuo = [random.uniform(limites[d][0], limites[d][1])
                     for d in range(dimension)]
        poblacion.append(individuo)

    fitness = [funcion_objetivo(ind) for ind in poblacion]

    for _ in range(max_iter):
        for i in range(tam_poblacion):
            indices = list(range(tam_poblacion))
            indices.remove(i)
            r1, r2, r3 = random.sample(indices, 3)
            x1 = poblacion[r1]
            x2 = poblacion[r2]
            x3 = poblacion[r3]

            mutante = []
            for d in range(dimension):
                valor = x1[d] + F * (x2[d] - x3[d])
                if valor < limites[d][0]:
                    valor = limites[d][0]
                if valor > limites[d][1]:
                    valor = limites[d][1]
                mutante.append(valor)

            j_rand = random.randint(0, dimension - 1)
            hijo = []
            for d in range(dimension):
                if random.random() < Cr or d == j_rand:
                    hijo.append(mutante[d])
                else:
                    hijo.append(poblacion[i][d])

            fit_hijo = funcion_objetivo(hijo)
            if maximizar:
                mejor = fit_hijo > fitness[i]
            else:
                mejor = fit_hijo < fitness[i]

            if mejor:
                poblacion[i] = hijo
                fitness[i] = fit_hijo

    if maximizar:
        idx = fitness.index(max(fitness))
    else:
        idx = fitness.index(min(fitness))
    return poblacion[idx]
