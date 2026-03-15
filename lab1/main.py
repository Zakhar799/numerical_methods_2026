import requests
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
response = requests.get(url)
data = response.json()
results = data["results"]

n = len(results)
print("Кількість вузлів:", n)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(d_lon / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


distances = [0]
elevations = []
for i in range(n):
    elevations.append(results[i]["elevation"])

for i in range(1, n):
    d = haversine(results[i - 1]["latitude"], results[i - 1]["longitude"],
                  results[i]["latitude"], results[i]["longitude"])
    distances.append(distances[i - 1] + d)

print("\nТабуляція (відстань, висота):")
print(" i | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

def my_spline(x, y):
    num = len(x) - 1
    h = []
    for i in range(num):
        h.append(x[i + 1] - x[i])

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


counts = [10, 15, 20]
plt.figure(figsize=(10, 6))

for c_nodes in counts:
    idx = np.linspace(0, n - 1, c_nodes).astype(int)
    xs = np.array(distances)[idx]
    ys = np.array(elevations)[idx]

    a, b, c, d = my_spline(xs, ys)

    x_fine = []
    y_fine = []
    for i in range(len(xs) - 1):
        step = np.linspace(xs[i], xs[i + 1], 20)
        for val in step:
            dx = val - xs[i]
            y_val = a[i] + b[i] * dx + c[i] * (dx ** 2) + d[i] * (dx ** 3)  
            x_fine.append(val)
            y_fine.append(y_val)

    plt.plot(x_fine, y_fine, label=f"Вузлів: {c_nodes}")

    if c_nodes == 20:
        print("\nКоефіцієнти сплайна (перші 3 інтервали):")
        print(" i |    a    |    b    |    c    |    d")
        for i in range(3):
            print(f"{i:2d} | {a[i]:7.2f} | {b[i]:7.4f} | {c[i]:7.6f} | {d[i]:7.8f}")

plt.scatter(distances, elevations, color='red', s=10, label="Дані GPS")
plt.legend()
plt.grid()
plt.title("Графік висоти (сплайни)")
plt.show()

ascent = 0
for i in range(1, n):
    diff = elevations[i] - elevations[i - 1]
    if diff > 0:
        ascent += diff

descent = 0
for i in range(1, n):
    diff = elevations[i - 1] - elevations[i]
    if diff > 0:
        descent += diff

print("\n--- ХАРАКТЕРИСТИКИ МАРШРУТУ ---")
print("Загальна довжина (м):", round(distances[-1], 2))
print("Сумарний набір висоти (м):", round(ascent, 2))
print("Сумарний спуск (м):", round(descent, 2))

work = 80 * 9.81 * ascent
print("Механічна робота (кДж):", round(work / 1000, 2))

f = open("results.txt", "w", encoding="utf-8")
f.write("РЕЗУЛЬТАТИ\n")
f.write(f"Довжина: {distances[-1]}\n")
f.write(f"Набір: {ascent}\n")
f.write(f"Спуск: {descent}\n")
f.close()