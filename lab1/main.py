import requests
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
response = requests.get(url)
data = response.json()
results = data["results"]

n = len(results)
print("Кількість вузлів:", n)

print("\nТабуляція вузлів:")
print(" i |  Latitude | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:8.2f}")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(d_lon / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


distances = [0]
elevations = [results[0]["elevation"]]

for i in range(1, n):
    elevations.append(results[i]["elevation"])
    d = haversine(results[i - 1]["latitude"], results[i - 1]["longitude"],
                  results[i]["latitude"], results[i]["longitude"])
    distances.append(distances[i - 1] + d)

print("\nТабуляція (відстань, висота):")
print(" i | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

def my_spline(x, y):
    num = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(num)]

    al = np.zeros(num + 1)
    be = np.zeros(num + 1)
    ga = np.zeros(num + 1)
    de = np.zeros(num + 1)

    be[0] = 1
    de[0] = 0

    for i in range(1, num):
        al[i] = h[i - 1]
        be[i] = 2 * (h[i - 1] + h[i])
        ga[i] = h[i]
        de[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    al[num] = h[num - 1]
    be[num] = 2 * h[num - 1]
    de[num] = 3 * ((y[num] - y[num - 1]) / h[num - 1])

    A = np.zeros(num + 1)
    B = np.zeros(num + 1)
    A[0] = -ga[0] / be[0]
    B[0] = de[0] / be[0]
    for i in range(1, num + 1):
        den = al[i] * A[i - 1] + be[i]
        if i < num:
            A[i] = -ga[i] / den
        B[i] = (de[i] - al[i] * B[i - 1]) / den

    c = np.zeros(num + 1)
    c[num] = B[num]
    for i in range(num - 1, -1, -1):
        c[i] = A[i] * c[i + 1] + B[i]

    a_coeffs = y[:-1]
    b_coeffs = np.zeros(num)
    d_coeffs = np.zeros(num)
    for i in range(num):
        d_coeffs[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b_coeffs[i] = (y[i + 1] - y[i]) / h[i] - (h[i] / 3) * (c[i + 1] + 2 * c[i])

    return a_coeffs, b_coeffs, c[:-1], d_coeffs

def evaluate_spline(x_eval, x_nodes, a, b, c, d):
    y_eval = []
    for x in x_eval:
        i = 0
        while i < len(x_nodes) - 2 and x > x_nodes[i + 1]:
            i += 1
        dx = x - x_nodes[i]
        y_eval.append(a[i] + b[i] * dx + c[i] * (dx ** 2) + d[i] * (dx ** 3))
    return np.array(y_eval)

counts = [10, 15, 20]

fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax1.plot(distances, elevations, label="21 вузол (еталон)")

print("\n")
for c_nodes in counts:
    idx = np.linspace(0, n - 1, c_nodes).astype(int)
    xs = np.array(distances)[idx]
    ys = np.array(elevations)[idx]

    a, b, c, d = my_spline(xs, ys)

    y_pred = evaluate_spline(distances, xs, a, b, c, d)
    errors = np.abs(np.array(elevations) - y_pred)

    print(f"{c_nodes} вузлів")
    print(f"Максимальна похибка: {np.max(errors)}")
    print(f"Середня похибка: {np.mean(errors)}")

    x_fine = np.linspace(xs[0], xs[-1], 200)
    y_fine = evaluate_spline(x_fine, xs, a, b, c, d)
    ax1.plot(x_fine, y_fine, label=f"{c_nodes} вузлів")

    ax2.plot(distances, errors, label=f"{c_nodes} вузлів")

ax1.set_title("Вплив кількості вузлів")
ax1.legend()
ax1.grid()

ax2.set_title("Похибка апроксимації")
ax2.legend()
ax2.grid()

print("\n--- ДОДАТКОВО: ХАРАКТЕРИСТИКИ МАРШРУТУ ---")

total_dist = distances[-1]
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))

print(f"Загальна довжина маршруту (м): {total_dist:.2f}")
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
print(f"Сумарний спуск (м): {total_descent:.2f}")

grad_full = np.gradient(elevations, distances) * 100
print("\nАналіз градієнта:")
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

mass = 80
g = 9.81
energy_J = mass * g * total_ascent

print("\nМеханічна енергія підйому:")
print(f"Механічна робота (Дж): {energy_J:.2f}")
print(f"Механічна робота (кДж): {energy_J / 1000:.2f}")
print(f"Енергія (ккал): {energy_J / 4184:.2f}")

f = open("results.txt", "w", encoding="utf-8")
f.write("РЕЗУЛЬТАТИ\n")
f.write(f"Загальна довжина (м): {total_dist:.2f}\n")
f.write(f"Сумарний набір висоти (м): {total_ascent:.2f}\n")
f.write(f"Сумарний спуск (м): {total_descent:.2f}\n")
f.close()
plt.show()