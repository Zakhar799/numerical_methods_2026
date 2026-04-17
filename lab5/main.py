import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)

a = 0
b = 24

x_graph = np.linspace(a, b, 1000)
y_graph = f(x_graph)
plt.figure(figsize=(10, 6))
plt.plot(x_graph, y_graph, label=r'$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$')
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True)
plt.legend()
plt.show()

def simpson_method(f, a, b, N):
    if N % 2 != 0:
        N = N + 1
    h = (b - a) / N

    suma = f(a) + f(b)

    for i in range(1, N, 2):
        xi = a + i * h
        suma = suma + 4 * f(xi)

    for i in range(2, N - 1, 2):
        xi = a + i * h
        suma = suma + 2 * f(xi)

    return (h / 3) * suma

I0 = simpson_method(f, a, b, 1000000)
print("Точне значення I0 =", I0)

N_values = []
errors = []
N_opt = 0
target_eps = 1e-12

for n in range(10, 1002, 2):
    In = simpson_method(f, a, b, n)
    eps = abs(In - I0)
    N_values.append(n)
    errors.append(eps)
    if eps <= target_eps and N_opt == 0:
        N_opt = n

print("Оптимальне N_opt =", N_opt)

plt.figure(figsize=(10, 5))
plt.plot(N_values, errors, color='red')
plt.yscale('log')
plt.title('Залежність похибки від N')
plt.xlabel('N')
plt.ylabel('Похибка')
plt.grid(True)
plt.show()

N0 = int((N_opt / 10) // 8 * 8)
if N0 < 8: N0 = 8
eps0 = abs(simpson_method(f, a, b, N0) - I0)
print(f"При N0 = {N0} похибка eps0 = {eps0}")

I_N0 = simpson_method(f, a, b, N0)
I_half_N0 = simpson_method(f, a, b, N0 // 2)
IR = I_N0 + (I_N0 - I_half_N0) / 15
epsR = abs(IR - I0)
print(f"Похибка Рунге-Ромберга epsR = {epsR}")

I1 = simpson_method(f, a, b, N0 // 4)
I2 = simpson_method(f, a, b, N0 // 2)
I3 = I_N0
IA = (I2 ** 2 - I1 * I3) / (2 * I2 - (I1 + I3))
epsA = abs(IA - I0)
print(f"Похибка Ейткена epsA = {epsA}")

p = (1 / np.log(2)) * np.log(abs((I3 - I2) / (I2 - I1)))
print(f"Оцінений порядок точності p = {p}")

calls_count = 0


def adaptive_simpson(f, a, b, tol):
    global calls_count
    c = (a + b) / 2
    I1 = simpson_method(f, a, b, 2)
    I2 = simpson_method(f, a, c, 2) + simpson_method(f, c, b, 2)
    calls_count += 5
    if abs(I1 - I2) < 15 * tol:
        return I2
    else:
        return adaptive_simpson(f, a, c, tol / 2) + adaptive_simpson(f, c, b, tol / 2)


print("\n9. Результати адаптивного алгоритму:")
print(f"{'Задана точність':<15} | {'Значення інтегралу':<20} | {'Кількість викликів f(x)'}")
print("-" * 65)

for t in [1e-3, 1e-6, 1e-9]:
    calls_count = 0
    res = adaptive_simpson(f, a, b, t)
    print(f"{t:<15} | {res:<20.12f} | {calls_count}")