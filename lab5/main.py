import numpy as np
import matplotlib.pyplot as plt

# Функція
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)

a, b = 0, 24

# --- Графік ---
x_graph = np.linspace(a, b, 1000)
plt.plot(x_graph, f(x_graph))
plt.title('Графік функції')
plt.grid()
plt.show()

# --- Метод Сімпсона ---
def simpson_method(f, a, b, N):
    if N % 2 != 0:
        N += 1
    h = (b - a) / N

    s = f(a) + f(b)

    for i in range(1, N, 2):
        s += 4 * f(a + i * h)

    for i in range(2, N, 2):
        s += 2 * f(a + i * h)

    return s * h / 3

# "Еталонне" значення
I0 = simpson_method(f, a, b, 200000)
print("Еталонне значення I0 =", I0)

# --- Похибка ---
N_values = []
errors = []

target_eps = 1e-7
N_opt = None

for n in range(10, 10000, 2):
    In = simpson_method(f, a, b, n)
    eps = abs(In - I0)

    N_values.append(n)
    errors.append(eps)

    if eps < target_eps and N_opt is None:
        N_opt = n

print("N_opt =", N_opt)

# Графік похибки
plt.plot(N_values, errors)
plt.yscale('log')
plt.title('Похибка')
plt.grid()
plt.show()

# --- N0 ---
N0 = (N_opt // 8) * 8
if N0 < 8:
    N0 = 8

I_N0 = simpson_method(f, a, b, N0)
eps0 = abs(I_N0 - I0)

print(f"N0 = {N0}, eps0 = {eps0}")

# --- Рунге-Ромберг ---
I_half = simpson_method(f, a, b, N0 // 2)
IR = I_N0 + (I_N0 - I_half) / 15
print("Runge error:", abs(IR - I0))

# --- Ейткен ---
I1 = simpson_method(f, a, b, N0 // 4)
I2 = simpson_method(f, a, b, N0 // 2)
I3 = I_N0

IA = (I2**2 - I1 * I3) / (2 * I2 - I1 - I3)
epsA = abs(IA - I0)

p = np.log(abs((I3 - I2) / (I2 - I1))) / np.log(2)

print("Aitken error:", epsA)
print("Order p =", p)

# --- Адаптивний метод ---
calls = 0

def f_count(x):
    global calls
    calls += 1
    return f(x)

def adaptive_simpson(f, a, b, eps):
    c = (a + b) / 2

    I1 = simpson_method(f, a, b, 2)
    I2 = simpson_method(f, a, c, 2) + simpson_method(f, c, b, 2)

    if abs(I1 - I2) < 15 * eps:
        return I2
    return adaptive_simpson(f, a, c, eps / 2) + adaptive_simpson(f, c, b, eps / 2)

print("\nАдаптивний метод:")

for tol in [1e-3, 1e-6, 1e-9]:
    calls = 0
    result = adaptive_simpson(f_count, a, b, tol)
    print(f"eps={tol}, I={result}, calls={calls}")