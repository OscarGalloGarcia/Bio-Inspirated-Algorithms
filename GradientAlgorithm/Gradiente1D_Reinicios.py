# Algoritmo: Gradiente 1D Con Reinicios
# Oscar Alberto Gallo García

import matplotlib.pyplot as plot
import random
import os

os.makedirs("figs", exist_ok=True)

# Funcion para gradiente descendente:
def f(x):
    return x**6 - 10*x**4 + 28*x**2 - x

# Funcion para gradiente ascendente:
def g(x):
    return -x**6 + 10*x**4 - 28*x**2 + x

# --- Derivada numerica ---
def derivada(fn, x, h=1e-5):
    return (fn(x + h) - fn(x - h)) / (2 * h)

# --- Algoritmo gradiente 1D ---
def gradiente(fn, alpha, e, modo, N, x_vals, nombre="resultado", carpeta="figs"):

    x_actual = random.uniform(x_vals[0], x_vals[-1])
    x_asterizco = x_actual
    fx_vals = [fn(x) for x in x_vals]

    trayectoria_x = []
    trayectoria_y = []

    for i in range(N):
        while True:
            grad = derivada(fn, x_actual)
            if abs(grad) < e:
                break
            if modo == "descenso":
                x_actual = x_actual - alpha * grad
            else:
                x_actual = x_actual + alpha * grad

            plot.clf()
            plot.plot(x_vals, fx_vals)
            plot.scatter(x_actual, fn(x_actual), color="red", zorder=5, label=f"Reinicio {i+1}")
            plot.scatter(x_asterizco, fn(x_asterizco), color="green", s=150, marker="*", zorder=6, label=f"x* = {x_asterizco:.4f}")
            plot.legend()
            plot.pause(0.01)

        if modo == "descenso":
            if fn(x_actual) < fn(x_asterizco):
                x_asterizco = x_actual
        else:
            if fn(x_actual) > fn(x_asterizco):
                x_asterizco = x_actual

        trayectoria_x.append(x_actual)
        trayectoria_y.append(fn(x_actual))

        x_actual = random.uniform(x_vals[0], x_vals[-1])

    # --- Guardar snapshots ---
    snapshots = [0, 4, N - 1]

    for i, idx in enumerate(snapshots):
        idx = min(idx, len(trayectoria_x) - 1)
        plot.figure()
        plot.plot(x_vals, fx_vals, label="f(x)")
        plot.scatter(trayectoria_x[:idx + 1], trayectoria_y[:idx + 1],
                     color="red", zorder=5, label="Convergencias")
        plot.scatter(trayectoria_x[idx], trayectoria_y[idx],
                     color="red", s=80, zorder=6,
                     label=f"x = {trayectoria_x[idx]:.4f}")
        plot.title(f"{nombre} | Reinicio {idx + 1}")
        plot.xlabel("x")
        plot.ylabel("f(x)")
        plot.legend()
        plot.savefig(f"{carpeta}/{nombre}_iter_{i + 1}.png", dpi=150, bbox_inches="tight")
        plot.close()

    # --- Guardar resultado final ---
    plot.figure()
    plot.plot(x_vals, fx_vals, label="f(x)")
    plot.scatter(trayectoria_x, trayectoria_y, color="red", zorder=5, label="Convergencias")
    plot.scatter(x_asterizco, fn(x_asterizco), color="green", s=200, marker="*",
                 zorder=6, label=f"x* = {x_asterizco:.4f}\nf(x*) = {fn(x_asterizco):.4f}")
    plot.title(f"{nombre} | Resultado final")
    plot.xlabel("x")
    plot.ylabel("f(x)")
    plot.legend()
    plot.savefig(f"{carpeta}/{nombre}_resultado_final.png", dpi=150, bbox_inches="tight")
    plot.close()

    print(f"Valor optimo: {x_asterizco}")
    return x_asterizco


if __name__ == "__main__":

    alpha = 0.01
    e = 1e-4  # Condicion de paro
    N = 20 # Numero de reinicios

    # Funcion 1. Gradiente descendente
    x1 = [i * 0.05 for i in range(-60, 60)]

    gradiente(fn=f, alpha=alpha, e=e,
              modo="descenso", N=N, x_vals=x1, nombre="reinicios_descenso")

    # Funcion 2. Gradiente acendente
    x2 = [i * 0.05 for i in range(-60, 60)]

    gradiente(fn=g, alpha=alpha, e=e,
              modo="ascenso", N=N, x_vals=x2, nombre="reinicios_ascenso")
