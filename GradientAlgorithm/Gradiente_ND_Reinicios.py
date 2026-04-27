# Algoritmo: Gradiente N-D Con Reinicios
# Oscar Alberto Gallo García

import matplotlib.pyplot as plot
import numpy as np
import os

os.makedirs("figs", exist_ok=True)


# --- Gradiente numérico (diferencias finitas centradas) ---
def gradiente_numerico(fn, x, h=1e-5):
    """Calcula ∇f(x) para x ∈ R^n usando diferencias finitas centradas."""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_mas = x.copy(); x_mas[i] += h
        x_menos = x.copy(); x_menos[i] -= h
        grad[i] = (fn(x_mas) - fn(x_menos)) / (2 * h)
    return grad


# --- Algoritmo gradiente N-D con reinicios ---
def gradiente_nd(fn, alpha, e, modo, N, bounds,
                 max_pasos=10000, nombre="resultado", carpeta="figs"):
    """
    Gradiente descendente/ascendente en n dimensiones con reinicios.

    Parámetros
    ----------
    fn        : función objetivo f: R^n -> R
    alpha     : tasa de aprendizaje
    e         : criterio de paro  (||∇f(x)|| < e)
    modo      : "descenso" o "ascenso"
    N         : número de reinicios
    bounds    : lista de tuplas [(min0,max0), (min1,max1), ...] por dimensión
    max_pasos : máximo de pasos por reinicio (evita bucles infinitos)
    nombre    : prefijo para las figuras guardadas
    carpeta   : directorio de salida
    """
    n_dims = len(bounds)
    os.makedirs(carpeta, exist_ok=True)

    def punto_aleatorio():
        return np.array([np.random.uniform(b[0], b[1]) for b in bounds])

    x_actual  = punto_aleatorio()
    x_asterisco = x_actual.copy()

    trayectoria   = []   # punto convergido al final de cada reinicio
    f_trayectoria = []   # f(x) en cada convergencia

    # ---- Configuración de animación en tiempo real (solo 2D) ----
    es_2d = (n_dims == 2)
    if es_2d:
        res = 80
        x0v = np.linspace(bounds[0][0], bounds[0][1], res)
        x1v = np.linspace(bounds[1][0], bounds[1][1], res)
        X0, X1 = np.meshgrid(x0v, x1v)
        Z = np.vectorize(lambda a, b: fn(np.array([a, b])))(X0, X1)

        plot.ion()
        fig_anim, ax_anim = plot.subplots(figsize=(7, 6))

    # ---- Ciclo principal de reinicios ----
    for i in range(N):
        pasos = 0
        while pasos < max_pasos:
            grad = gradiente_numerico(fn, x_actual)
            if np.linalg.norm(grad) < e:
                break
            if modo == "descenso":
                x_actual = x_actual - alpha * grad
            else:
                x_actual = x_actual + alpha * grad

            # Mantener dentro de los límites del espacio de búsqueda
            for d in range(n_dims):
                x_actual[d] = np.clip(x_actual[d], bounds[d][0], bounds[d][1])
            pasos += 1

            # Animación paso a paso (solo 2D)
            if es_2d:
                ax_anim.cla()
                ax_anim.contourf(X0, X1, Z, levels=30, cmap="plasma", alpha=0.75)
                ax_anim.contour(X0, X1, Z, levels=15, colors="white",
                                linewidths=0.4, alpha=0.5)
                if trayectoria:
                    prev = np.array(trayectoria)
                    ax_anim.scatter(prev[:, 0], prev[:, 1], color="red", s=50,
                                   zorder=5, label="Convergencias")
                ax_anim.scatter(x_actual[0], x_actual[1],
                                color="yellow", s=80, zorder=6, label="Actual")
                ax_anim.scatter(x_asterisco[0], x_asterisco[1],
                                color="lime", s=180, marker="*", zorder=7,
                                label=f"x* = ({x_asterisco[0]:.3f}, {x_asterisco[1]:.3f})")
                ax_anim.set_title(f"{nombre} | Reinicio {i+1}, paso {pasos}")
                ax_anim.set_xlabel("x\u2080"); ax_anim.set_ylabel("x\u2081")
                ax_anim.legend(loc="upper right", fontsize=8)
                plot.pause(0.001)

        # Actualizar mejor solución encontrada
        if modo == "descenso":
            if fn(x_actual) < fn(x_asterisco):
                x_asterisco = x_actual.copy()
        else:
            if fn(x_actual) > fn(x_asterisco):
                x_asterisco = x_actual.copy()

        trayectoria.append(x_actual.copy())
        f_trayectoria.append(fn(x_actual))

        # Reinicio: nuevo punto aleatorio
        x_actual = punto_aleatorio()

    if es_2d:
        plot.ioff()
        plot.close(fig_anim)

    # ---- Guardar figura de resultados ----
    tray = np.array(trayectoria)

    fig, (ax1, ax2) = plot.subplots(1, 2, figsize=(13, 5))

    if es_2d:
        # Contour plot con puntos de convergencia
        cf = ax1.contourf(X0, X1, Z, levels=30, cmap="plasma", alpha=0.8)
        plot.colorbar(cf, ax=ax1)
        ax1.scatter(tray[:, 0], tray[:, 1], color="red", s=60, zorder=5,
                    label="Convergencias")
        ax1.scatter(x_asterisco[0], x_asterisco[1],
                    color="lime", s=200, marker="*", zorder=7,
                    label=f"x* = ({x_asterisco[0]:.3f}, {x_asterisco[1]:.3f})\n"
                          f"f(x*) = {fn(x_asterisco):.4f}")
        ax1.set_title(f"{nombre} | Trayectoria 2D")
        ax1.set_xlabel("x\u2080"); ax1.set_ylabel("x\u2081")
        ax1.legend(fontsize=8)
    else:
        # Proyección en las primeras 2 dimensiones (para n > 2)
        sc = ax1.scatter(tray[:, 0], tray[:, 1], c=f_trayectoria,
                         cmap="plasma", s=70, zorder=5)
        plot.colorbar(sc, ax=ax1, label="f(x)")
        ax1.scatter(x_asterisco[0], x_asterisco[1],
                    color="lime", s=200, marker="*", zorder=7,
                    label=f"x*: x\u2080={x_asterisco[0]:.3f}, x\u2081={x_asterisco[1]:.3f}\n"
                          f"f(x*) = {fn(x_asterisco):.4f}")
        ax1.set_title(f"{nombre} | Proyección x\u2080–x\u2081  ({n_dims}D)")
        ax1.set_xlabel("x\u2080"); ax1.set_ylabel("x\u2081")
        ax1.legend(fontsize=8)

    # Curva de convergencia (aplica siempre)
    ax2.plot(range(1, N + 1), f_trayectoria, "o-", color="steelblue",
             label="f(x) por reinicio")
    ax2.axhline(fn(x_asterisco), color="green", linestyle="--",
                label=f"Mejor f(x*) = {fn(x_asterisco):.4f}")
    ax2.set_title(f"{nombre} | Curva de convergencia")
    ax2.set_xlabel("Reinicio"); ax2.set_ylabel("f(x)")
    ax2.legend(fontsize=8)

    plot.tight_layout()
    plot.savefig(f"{carpeta}/{nombre}_resultado_final.png", dpi=150, bbox_inches="tight")
    plot.close()

    print(f"\n[{nombre}]")
    print(f"  Dimensiones : {n_dims}D")
    print(f"  Modo        : {modo}")
    print(f"  x*          = {np.round(x_asterisco, 5)}")
    print(f"  f(x*)       = {fn(x_asterisco):.6f}")
    return x_asterisco


# ======================================================================
# EJEMPLOS
# ======================================================================
if __name__ == "__main__":

    alpha = 0.01
    e     = 1e-4
    N     = 20

    # ------------------------------------------------------------------
    # Funciones de ejemplo: extensión directa de la función 1D al espacio n-D
    #
    #   Descenso:  f(x) = Σᵢ [ xᵢ⁶ − 10xᵢ⁴ + 28xᵢ² − xᵢ ]
    #   Ascenso:   g(x) = Σᵢ [ −xᵢ⁶ + 10xᵢ⁴ − 28xᵢ² + xᵢ ]
    # ------------------------------------------------------------------
    def f_nd(v):
        return sum( xi**6 - 10*xi**4 + 28*xi**2 - xi  for xi in v)

    def g_nd(v):
        return sum(-xi**6 + 10*xi**4 - 28*xi**2 + xi  for xi in v)

    # --- 2D Gradiente ASCENDENTE ---
    print("=" * 50)
    print("Ejemplo 1: 2D — Gradiente ascendente")
    gradiente_nd(fn=g_nd, alpha=alpha, e=e, modo="ascenso", N=N,
                 bounds=[(-3, 3), (-3, 3)], nombre="2D_ascenso")

    # --- 2D Gradiente DESCENDENTE ---
    print("=" * 50)
    print("Ejemplo 2: 2D — Gradiente descendente")
    gradiente_nd(fn=f_nd, alpha=alpha, e=e, modo="descenso", N=N,
                 bounds=[(-3, 3), (-3, 3)], nombre="2D_descenso")

    # --- 5D Gradiente DESCENDENTE ---
    print("=" * 50)
    print("Ejemplo 3: 5D — Gradiente descendente")
    gradiente_nd(fn=f_nd, alpha=alpha, e=e, modo="descenso", N=N,
                 bounds=[(-3, 3)] * 5, nombre="5D_descenso")

    # --- 5D Gradiente ASCENDENTE ---
    print("=" * 50)
    print("Ejemplo 4: 5D — Gradiente ascendente")
    gradiente_nd(fn=g_nd, alpha=alpha, e=e, modo="ascenso", N=N,
                 bounds=[(-3, 3)] * 5, nombre="5D_ascenso")

    # --- 10D Gradiente DESCENDENTE ---
    print("=" * 50)
    print("Ejemplo 5: 10D — Gradiente descendente")
    gradiente_nd(fn=f_nd, alpha=alpha, e=e, modo="descenso", N=N,
                 bounds=[(-3, 3)] * 10, nombre="10D_descenso")

    # --- 10D Gradiente ASCENDENTE ---
    print("=" * 50)
    print("Ejemplo 6: 10D — Gradiente ascendente")
    gradiente_nd(fn=g_nd, alpha=alpha, e=e, modo="ascenso", N=N,
                 bounds=[(-3, 3)] * 10, nombre="10D_ascenso")
