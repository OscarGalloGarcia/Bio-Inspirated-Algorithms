import math, random

SERVICIOS = {
    # EC2
    "ec2_small":    {"precio": 30,  "latencia": 40, "disponibilidad": 0.99},
    "ec2_medium":   {"precio": 60,  "latencia": 25, "disponibilidad": 0.999},
    "ec2_large":    {"precio": 110, "latencia": 15, "disponibilidad": 0.9999},
    "ec2_xlarge":   {"precio": 180, "latencia": 8,  "disponibilidad": 0.9999},

    # RDS 
    "rds_small":    {"precio": 50,  "latencia": 20, "disponibilidad": 0.99},
    "rds_medium":   {"precio": 80,  "latencia": 12, "disponibilidad": 0.999},
    "rds_large":    {"precio": 140, "latencia": 6,  "disponibilidad": 0.9999},

    # ElastiCache 
    "cache_small":  {"precio": 25,  "latencia": 2,  "disponibilidad": 0.99},
    "cache_medium": {"precio": 45,  "latencia": 1,  "disponibilidad": 0.999},

    # Load Balancer
    "lb_basic":     {"precio": 20,  "latencia": 5,  "disponibilidad": 0.999},
    "lb_premium":   {"precio": 50,  "latencia": 2,  "disponibilidad": 0.9999},
}

TIPOS = ["ec2", "rds", "cache", "lb"]
UMBRAL = {"max_latencia": 60, "min_disponibilidad": 0.99}
LAMBDAS = {"latencia": 10, "disponibilidad": 5000}

# Función de costoo
def costo(config):
    precio = sum(SERVICIOS[config[t]]["precio"] for t in TIPOS)
    latencia = sum(SERVICIOS[config[t]]["latencia"] for t in TIPOS)
    disponibilidad = math.prod(SERVICIOS[config[t]]["disponibilidad"] for t in TIPOS)

    infactibilidad_latencia = max(0, latencia - UMBRAL["max_latencia"])
    infactibilidad_disponibilidad = max(0, UMBRAL["min_disponibilidad"] - disponibilidad)

    return precio + LAMBDAS["latencia"] * infactibilidad_latencia \
                  + LAMBDAS["disponibilidad"] * infactibilidad_disponibilidad

def vecino(config):
    nuevo  = config.copy()
    tipo = random.choice(TIPOS)
    opciones = [k for k in SERVICIOS if k.startswith(tipo)]
    nuevo[tipo] = random.choice(opciones)
    return nuevo

# Simulated Annealing
def simulated_annealing(T=1000, alpha=0.95, iterations=1000, T_min=0.1, sin_mejora_max=50):
    x    = {"ec2": "ec2_medium", "rds": "rds_small", "cache": "cache_small", "lb": "lb_basic"}
    mejor = x.copy()
    sin_mejora = 0

    for i in range(iterations):
        if T < T_min:
            print(f"  Paro: temperatura mínima en iteración {i}")
            break
        if sin_mejora >= sin_mejora_max:
            print(f"  Paro: convergencia en iteración {i}")
            break

        y     = vecino(x)
        delta = costo(y) - costo(x)
        if delta < 0 or random.random() < math.exp(-delta / T):
            x = y

        if costo(x) < costo(mejor):
            mejor = x.copy()
            sin_mejora = 0
        else:
            sin_mejora += 1

        T *= alpha

    return mejor, costo(mejor)


mejor_config, mejor_costo = simulated_annealing()
print(f"\nMejor configuración encontrada:")
for k, v in mejor_config.items():
    print(f"  {k:8} → {v}")
print(f"\ncosto mensual: ${mejor_costo:.2f}")