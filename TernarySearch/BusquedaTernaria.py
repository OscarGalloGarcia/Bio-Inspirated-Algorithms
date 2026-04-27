# Algoritmo: Busqueda Ternaria
# Oscar Alberto Gallo García

import matplotlib.pyplot as plot

#Funcion 1
def f(x):
    return (x + 5)**2 + 7

#Funcion 2
def f2(x):
    return (x)**2 + 2*x

#Algoritmo de busqueda ternaria
def BusquedaTernaria(x,fx,a0,b0,e,fn):

    no_iteracion = 0

    #Grafica inicial
    plot.plot(x, fx)
    plot.axvline(a0)
    plot.axvline(b0)
    plot.scatter(a0, fn(a0), label="a0")
    plot.scatter(b0, fn(b0), label="b0")
    plot.title(f"Iteración {no_iteracion}")
    plot.xlabel("x")
    plot.ylabel("f(x)")
    plot.pause(10)

    while (b0 - a0) > e:
        L = ((b0-a0)/3)
        a1 = a0 + L
        b1 = b0 - L

        fa = fn(a1)
        fb = fn(b1)

        if fa < fb:
            b0 = b1
        else:
            a0 = a1

        no_iteracion += 1
        # Actualizar los puntos de la grafica
        plot.clf()
        plot.plot(x, fx)
        plot.axvline(a0)
        plot.axvline(b0)
        plot.scatter(a0, fn(a0), label="a0")
        plot.scatter(b0, fn(b0), label="b0")
        plot.title(f"Iteración {no_iteracion}")
        plot.xlabel("x")
        plot.ylabel("f(x)")
        #Pausa para ver la grafica
        plot.pause(5) 
    
    min = (a0 + b0)/2
    print(f"Minimo de la funcion: {min}")
    plot.show()


if __name__ == "__main__":
    x = [i * 0.1 for i in range(-150, 51)]
    fx = [f(xi) for xi in x]

    # Intervalo de búsqueda
    a0 = -8
    b0 = 0
    e = 0.1 # Condicion de paro

    #Algoritmo de busqueda ternaria funcion 1
    BusquedaTernaria(x,fx,a0,b0,e, fn=f)

    # Funcion 2 #
    x2 = [i * 0.1 for i in range(-150, 151)]
    fx2 = [f2(xi) for xi in x2]
    # Intervalo inicial para f2 
    a0_2 = -10
    b0_2 = 10
    #Algoritmo de busqueda ternaria para f2
    BusquedaTernaria(x2, fx2, a0=a0_2, b0=b0_2, e=e, fn=f2)





