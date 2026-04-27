# Algoritmo: Busqueda Ternaria
# Oscar Alberto Gallo García

import matplotlib.pyplot as plot
import os

# Crear carpeta para guardar imágenes
carpeta = "imagenes_busqueda_ternaria"
if not os.path.exists(carpeta):
    os.makedirs(carpeta)

# Funcion 1
def f(x):
    return (x + 5)**2 + 7

# Funcion 2
def f2(x):
    return (x)**2 + 2*x

# Algoritmo de busqueda ternaria
def BusquedaTernaria(x, fx, a0, b0, e, fn, nombre_funcion):

    contador = 0  # Contador de iteraciones

    while (b0 - a0) > e:
        L = (b0 - a0) / 3
        a1 = a0 + L
        b1 = b0 - L

        fa = fn(a1)
        fb = fn(b1)

        # Graficar
        plot.figure()
        plot.plot(x, fx)
        plot.axvline(a0, linestyle="--")
        plot.axvline(b0, linestyle="--")
        plot.scatter(a1, fa)
        plot.scatter(b1, fb)
        plot.title(f"Iteración {contador}")
        plot.xlabel("x")
        plot.ylabel("f(x)")

        # Guardar imagen
        plot.savefig(f"{carpeta}/{nombre_funcion}_iter_{contador}.png")
        plot.close()

        # Actualizar intervalo
        if fa < fb:
            b0 = b1
        else:
            a0 = a1

        contador += 1

    minimo = (a0 + b0) / 2

    # Guardar gráfica final
    plot.figure()
    plot.plot(x, fx)
    plot.axvline(a0, linestyle="--")
    plot.axvline(b0, linestyle="--")
    plot.scatter(minimo, fn(minimo))
    plot.title("Resultado Final")
    plot.xlabel("x")
    plot.ylabel("f(x)")
    plot.savefig(f"{carpeta}/{nombre_funcion}_resultado_final.png")
    plot.close()

    print(f"Mínimo aproximado de la función {nombre_funcion}: {minimo}")


if __name__ == "__main__":

    e = 0.1  # Condición de paro

    # -------- Funcion 1 --------
    x = [i * 0.1 for i in range(-150, 51)]
    fx = [f(xi) for xi in x]
    BusquedaTernaria(x, fx, a0=-8, b0=0, e=e, fn=f, nombre_funcion="f1")

    # -------- Funcion 2 --------
    x2 = [i * 0.1 for i in range(-150, 151)]
    fx2 = [f2(xi) for xi in x2]
    BusquedaTernaria(x2, fx2, a0=-10, b0=10, e=e, fn=f2, nombre_funcion="f2")
