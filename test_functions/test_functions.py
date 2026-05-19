#Oscar Alberto Gallo García
#Algoritmos Bio-inspirados: Funciones de Prueba


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# ─── Funciones de prueba

def spherical(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def quadric(x: np.ndarray) -> float:
    n = len(x)
    return float(sum(np.sum(x[:j + 1]) ** 2 for j in range(n)))


def ackley(x: np.ndarray) -> float:
    n = len(x)
    a = -0.2 * np.sqrt(np.sum(x ** 2) / n)
    b = np.sum(np.cos(2 * np.pi * x)) / n
    return float(-20 * np.exp(a) - np.exp(b) + 20 + np.e)


def bohachevsky1(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    return float(
        x1 ** 2 + 2 * x2 ** 2
        - 0.3 * np.cos(3 * np.pi * x1)
        - 0.4 * np.cos(4 * np.pi * x2)
        + 0.7
    )


def colville(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return float(
        100 * (x2 - x1 ** 2) ** 2
        + (1 - x1) ** 2
        + 90 * (x4 - x3 ** 2) ** 2
        + (1 - x3) ** 2
        + 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
        + 19.8 * (x2 - 1) * (x4 - 1)
    )


def easom(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    return float(
        -np.cos(x1) * np.cos(x2)
        * np.exp(-((x1 - np.pi) ** 2) - (x2 - np.pi) ** 2)
    )


def griewank(x: np.ndarray) -> float:
    j = np.arange(1, len(x) + 1)
    return float(1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(j))))


def hyperellipsoid(x: np.ndarray) -> float:
    j = np.arange(1, len(x) + 1)
    return float(np.sum((j * x) ** 2))


def rastrigin(x: np.ndarray) -> float:
    return float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10))


def rosenbrock(x: np.ndarray) -> float:
    n = len(x)
    total = 0.0
    for j in range(n // 2):
        x2j_1 = x[2 * j]
        x2j   = x[2 * j + 1]
        total += 100 * (x2j - x2j_1 ** 2) ** 2 + (1 - x2j_1) ** 2
    return total


def schwefel(x: np.ndarray) -> float:
    # Forma estándar: 418.9829·n − Σ xⱼ·sin(√|xⱼ|), f* = 0 en x_j ≈ 420.9687
    n = len(x)
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


FUNCTIONS = {
    "Spherical":      {"fn": spherical,      "bounds": (-100, 100),    "dim": None, "f_star": 0,  "x_star": lambda n: np.zeros(n)},
    "Quadric":        {"fn": quadric,        "bounds": (-100, 100),    "dim": None, "f_star": 0,  "x_star": lambda n: np.zeros(n)},
    "Ackley":         {"fn": ackley,         "bounds": (-30,  30),     "dim": None, "f_star": 0,  "x_star": lambda n: np.zeros(n)},
    "Bohachevsky1":   {"fn": bohachevsky1,   "bounds": (-50,  50),     "dim": 2,    "f_star": 0,  "x_star": lambda n: np.zeros(2)},
    "Colville":       {"fn": colville,       "bounds": (-10,  10),     "dim": 4,    "f_star": 0,  "x_star": lambda n: np.ones(4)},
    "Easom":          {"fn": easom,          "bounds": (-100, 100),    "dim": 2,    "f_star": -1, "x_star": lambda n: np.array([np.pi, np.pi])},
    "Griewank":       {"fn": griewank,       "bounds": (-600, 600),    "dim": None, "f_star": 0,  "x_star": lambda n: np.zeros(n)},
    "Hyperellipsoid": {"fn": hyperellipsoid, "bounds": (-1,   1),      "dim": None, "f_star": 0,  "x_star": lambda n: np.zeros(n)},
    "Rastrigin":      {"fn": rastrigin,      "bounds": (-5.12, 5.12),  "dim": None, "f_star": 0,  "x_star": lambda n: np.zeros(n)},
    "Rosenbrock":     {"fn": rosenbrock,     "bounds": (-2.048, 2.048),"dim": None, "f_star": 0,  "x_star": lambda n: np.ones(n)},
    "Schwefel":       {"fn": schwefel,       "bounds": (-500, 500),    "dim": None, "f_star": 0,  "x_star": lambda n: np.full(n, 420.9687)},
}


# ─── Graficación individual

def plot_individual_functions(resolution: int = 150) -> None:
    os.makedirs("figs", exist_ok=True)
    plottable = {k: v for k, v in FUNCTIONS.items() if k != "Colville"}

    for name, info in plottable.items():
        lo, hi = info["bounds"]
        lo_p, hi_p = (0, 2 * np.pi + 1) if name == "Easom" else (lo, hi)

        x1 = np.linspace(lo_p, hi_p, resolution)
        x2 = np.linspace(lo_p, hi_p, resolution)
        X1, X2 = np.meshgrid(x1, x2)

        Z = np.array([
            [info["fn"](np.array([X1[i, j], X2[i, j]])) for j in range(resolution)]
            for i in range(resolution)
        ])

        fig = plt.figure(figsize=(12, 5))
        fig.suptitle(
            f"{name}   |   dominio $[{lo},\\,{hi}]^n$   |   $f^* = {info['f_star']}$",
            fontsize=11
        )

        # Superficie 3D
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        surf = ax1.plot_surface(X1, X2, Z, cmap=cm.viridis, alpha=0.85, linewidth=0)
        fig.colorbar(surf, ax=ax1, shrink=0.5, pad=0.1)

        x_opt = info["x_star"](2)
        in_view = lo_p <= x_opt[0] <= hi_p and lo_p <= x_opt[1] <= hi_p
        if in_view:
            f_opt = info["fn"](x_opt)
            ax1.scatter([x_opt[0]], [x_opt[1]], [f_opt],
                        color="red", s=60, zorder=10, label=f"$f^*={info['f_star']}$")
            ax1.legend(fontsize=9)

        ax1.set_title("Superficie 3D", fontsize=10)
        ax1.set_xlabel("$x_1$", fontsize=9)
        ax1.set_ylabel("$x_2$", fontsize=9)
        ax1.set_zlabel("$f(x)$", fontsize=9)
        ax1.tick_params(labelsize=7)

        # Curvas de nivel
        ax2 = fig.add_subplot(1, 2, 2)
        cp = ax2.contourf(X1, X2, Z, levels=40, cmap=cm.viridis)
        ax2.contour(X1, X2, Z, levels=40, colors="white", linewidths=0.3, alpha=0.4)
        fig.colorbar(cp, ax=ax2)

        if in_view:
            ax2.plot(x_opt[0], x_opt[1], "r*", markersize=12,
                     label=f"$x^* = {np.round(x_opt, 3)}$")
            ax2.legend(fontsize=8)

        ax2.set_title("Curvas de nivel", fontsize=10)
        ax2.set_xlabel("$x_1$", fontsize=9)
        ax2.set_ylabel("$x_2$", fontsize=9)
        ax2.tick_params(labelsize=7)

        plt.tight_layout()
        fname = f"figs/{name.lower()}.png"
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Guardado: {fname}")


# ─── Tabla de evaluación

def evaluate_all_functions(dims: list[int] = [5, 10, 20]) -> None:
    col_w = [18, 6, 35, 14, 8]
    header = (
        f"{'Función':<{col_w[0]}} {'N':>{col_w[1]}} "
        f"{'x (truncado)':<{col_w[2]}} {'f(x)':>{col_w[3]}} {'f*':>{col_w[4]}}"
    )
    sep = "-" * sum(col_w + [len(col_w) * 1 + 4])

    print("\n" + sep)
    print(header)
    print(sep)

    for name, info in FUNCTIONS.items():
        fn      = info["fn"]
        f_star  = info["f_star"]
        fixed_n = info["dim"]

        eval_dims = [fixed_n] if fixed_n is not None else dims

        for n in eval_dims:
            x = info["x_star"](n)
            fx = fn(x)

            if len(x) <= 4:
                x_str = "[" + ", ".join(f"{v:.4g}" for v in x) + "]"
            else:
                x_str = "[" + ", ".join(f"{v:.4g}" for v in x[:3]) + f", ... ({n}D)]"

            fx_str = f"{fx:.6e}"
            print(
                f"{name:<{col_w[0]}} {n:>{col_w[1]}} "
                f"{x_str:<{col_w[2]}} {fx_str:>{col_w[3]}} {f_star:>{col_w[4]}}"
            )

        print(sep)


def main():
    print("=" * 70)
    print("Funciones de Prueba para Optimización")
    print("=" * 70)

    print("\n[NOTA] Colville es una función fija de 4D — omitida en la gráfica 2D.\n")

    evaluate_all_functions(dims=[5, 10, 20])

    print("\nGenerando imágenes individuales...")
    plot_individual_functions(resolution=150)


if __name__ == "__main__":
    main()
