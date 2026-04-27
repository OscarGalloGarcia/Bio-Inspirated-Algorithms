# Algoritmo: Gradiente 1D
# Oscar Alberto Gallo García

import matplotlib.pyplot as plot
import os

# Crear carpeta para guardar imagenes
carpeta = "imagenes_gradiente"
if not os.path.exists(carpeta):
    os.makedirs(carpeta)

# Funcion para gradiente descendente:
def f(x):
    return x**2 + 2*x + 3

# Funcion para gradiente ascendente:
def g(x):
    return -(x - 2)**2 + 9

# --- Derivada numerica ---
def derivada(fn, x, h=1e-5):
    return (fn(x + h) - fn(x - h)) / (2 * h)

# --- Algoritmo gradiente 1D ---
def gradiente(x0, fn, alpha, e, modo, x_vals, fx_vals, titulo, nombre):
    #x0     : punto inicial
    #fn     : funcion
    #alpha  : paso
    #e      : condicion de paro
    #modo   : 'descenso' o 'ascenso'
    #nombre : prefijo para los archivos guardados

    x_actual = x0
    trayectoria_x = [x_actual]
    trayectoria_y = [fn(x_actual)]

    # --- Correr todas las iteraciones ---
    while True:
        grad = derivada(fn, x_actual)

        if abs(grad) < e:
            break

        if modo == "descenso":
            x_actual = x_actual - alpha * grad
        else:
            x_actual = x_actual + alpha * grad

        trayectoria_x.append(x_actual)
        trayectoria_y.append(fn(x_actual))

    resultado = x_actual
    valor_optimo = fn(resultado)
    total = len(trayectoria_x)

    print(f"\n[{titulo}]")
    print(f"  Iteraciones   : {total - 1}")
    print(f"  x optimo      : {resultado:.6f}")
    print(f"  f(x optimo)   : {valor_optimo:.6f}")

    # --- Guardar 3 imagenes intermedias (inicio, 1/3, 2/3 del recorrido) ---
    snapshots = [0, 4, 25]

    for i, idx in enumerate(snapshots):
        plot.figure()
        plot.plot(x_vals, fx_vals, label="f(x)")
        plot.plot(trayectoria_x[:idx+1], trayectoria_y[:idx+1],
                  "r--o", markersize=4, label="Trayectoria")
        plot.scatter(trayectoria_x[idx], trayectoria_y[idx],
                     color="red", zorder=5,
                     label=f"x = {trayectoria_x[idx]:.4f}")
        plot.title(f"{titulo} | Iteracion {idx}")
        plot.xlabel("x")
        plot.ylabel("f(x)")
        plot.legend()
        plot.savefig(f"{carpeta}/{nombre}_iter_{i+1}.png")
        plot.close()

    # --- Guardar imagen del resultado final ---
    plot.figure()
    plot.plot(x_vals, fx_vals, label="f(x)")
    plot.plot(trayectoria_x, trayectoria_y, "r--o", markersize=4, label="Trayectoria")
    plot.scatter(resultado, valor_optimo, color="green", s=100, zorder=5,
                 label=f"Optimo: x={resultado:.4f}, f={valor_optimo:.4f}")
    plot.title(f"{titulo} | Resultado final")
    plot.xlabel("x")
    plot.ylabel("f(x)")
    plot.legend()
    plot.savefig(f"{carpeta}/{nombre}_resultado_final.png")
    plot.close()

    print(f"  Imagenes guardadas en: {carpeta}/")

    return resultado, valor_optimo


if __name__ == "__main__":

    alpha = 0.1
    e = 1e-4  # Condicion de paro

    # -------- Ejemplo 1: Gradiente DESCENDENTE --------
    # f(x) = x^2 + 2x + 3  minimo en x = -1,  f(-1) = 2
    x1 = [i * 0.1 for i in range(-50, 40)]
    fx1 = [f(xi) for xi in x1]

    gradiente(x0=3.0, fn=f, alpha=alpha, e=e,
              modo="descenso",
              x_vals=x1, fx_vals=fx1,
              titulo="Gradiente Descendente: f(x) = x^2 + 2x + 3",
              nombre="descenso_f")

    # -------- Ejemplo 2: Gradiente ASCENDENTE --------
    # g(x) = -(x-2)^2 + 9   maximo en x = 2,  g(2) = 9
    x2 = [i * 0.1 for i in range(-30, 70)]
    fx2 = [g(xi) for xi in x2]

    gradiente(x0=-2.0, fn=g, alpha=alpha, e=e,
              modo="ascenso",
              x_vals=x2, fx_vals=fx2,
              titulo="Gradiente Ascendente: g(x) = -(x-2)^2 + 9",
              nombre="ascenso_g")
