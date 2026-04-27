# Gradiente N-Dimensional
# Oscar Alberto Gallo Garcia

import matplotlib.pyplot as plot

# f(x1, x2) = (x1+2)^2 + (x2-3)^2  ->  minimo en (-2, 3)
def f(x):
    return (x[0] + 2)**2 + (x[1] - 3)**2

# g(x1, x2) = -(x1-1)^2 - (x2+1)^2 + 10  ->  maximo en (1, -1)
def g(x):
    return -(x[0] - 1)**2 - (x[1] + 1)**2 + 10


# Gradiente numerico
def gradiente_numerico(fn, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_mas = x[:]
        x_menos = x[:]
        x_mas[i] = x_mas[i] + h
        x_menos[i] = x_menos[i] - h
        grad.append((fn(x_mas) - fn(x_menos)) / (2 * h))
    return grad


# Norma del gradiente para la condicion de paro
def norma(v):
    return sum(vi**2 for vi in v) ** 0.5


def gradiente_nd(x0, fn, alpha, e, modo, titulo):
    x_actual = x0[:]
    trayectoria = [x_actual[:]]
    valores_f = [fn(x_actual)]

    while True:
        grad = gradiente_numerico(fn, x_actual)

        if norma(grad) < e:
            break

        if modo == "descenso":
            x_actual = [x_actual[i] - alpha * grad[i] for i in range(len(x_actual))]
        else:
            x_actual = [x_actual[i] + alpha * grad[i] for i in range(len(x_actual))]

        trayectoria.append(x_actual[:])
        valores_f.append(fn(x_actual))

    resultado = x_actual
    valor_optimo = fn(resultado)
    total = len(trayectoria)

    print(f"\n{titulo}")
    print(f"  Iteraciones: {total - 1}")
    print(f"  x optimo:    {[round(v, 6) for v in resultado]}")
    print(f"  f(x optimo): {valor_optimo:.6f}")

    # Graficas de contorno con trayectoria (solo 2D)
    if len(x0) == 2:
        x_min = min(t[0] for t in trayectoria) - 2
        x_max = max(t[0] for t in trayectoria) + 2
        y_min = min(t[1] for t in trayectoria) - 2
        y_max = max(t[1] for t in trayectoria) + 2

        paso = 0.2
        x_grid = [x_min + i * paso for i in range(int((x_max - x_min) / paso) + 1)]
        y_grid = [y_min + j * paso for j in range(int((y_max - y_min) / paso) + 1)]
        Z = [[fn([xr, yr]) for xr in x_grid] for yr in y_grid]

        # Para capturar iteraciones 0, 4 y 25
        for idx in [0, 4, 25]:
            idx = min(idx, total - 1)
            tray_x = [t[0] for t in trayectoria[:idx + 1]]
            tray_y = [t[1] for t in trayectoria[:idx + 1]]

            plot.figure()
            contorno = plot.contour(x_grid, y_grid, Z, levels=20)
            plot.colorbar(contorno)
            plot.plot(tray_x, tray_y, "r--o", markersize=4, label="Trayectoria")
            plot.scatter(trayectoria[idx][0], trayectoria[idx][1], color="red", zorder=5,
                         label=f"x = {[round(v, 3) for v in trayectoria[idx]]}")
            plot.title(f"{titulo} | Iteracion {idx}")
            plot.xlabel("x1")
            plot.ylabel("x2")
            plot.legend()
            plot.show()

        # Resultado final
        tray_x = [t[0] for t in trayectoria]
        tray_y = [t[1] for t in trayectoria]

        plot.figure()
        contorno = plot.contour(x_grid, y_grid, Z, levels=20)
        plot.colorbar(contorno)
        plot.plot(tray_x, tray_y, "r--o", markersize=3, label="Trayectoria")
        plot.scatter(resultado[0], resultado[1], color="green", s=100, zorder=5,
                     label=f"Optimo: {[round(v, 4) for v in resultado]}")
        plot.title(f"{titulo} | Resultado final")
        plot.xlabel("x1")
        plot.ylabel("x2")
        plot.legend()
        plot.show()

    # Grafica de convergencia
    plot.figure()
    plot.plot(range(len(valores_f)), valores_f, "b-o", markersize=3)
    plot.title(f"{titulo} | Convergencia")
    plot.xlabel("Iteracion")
    plot.ylabel("f(x)")
    plot.show()

    return resultado, valor_optimo


if __name__ == "__main__":

    alpha = 0.1
    e = 1e-4

    # Funcion 1: descenso -> minimo de f en (-2, 3)
    gradiente_nd(x0=[4.0, 6.0], fn=f, alpha=alpha, e=e,
                 modo="descenso", titulo="Gradiente Descendente 2D")

    # Funcion 2: ascenso -> maximo de g en (1, -1)
    gradiente_nd(x0=[-3.0, 4.0], fn=g, alpha=alpha, e=e,
                 modo="ascenso", titulo="Gradiente Ascendente 2D")
