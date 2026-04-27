import numpy as np
import matplotlib.pyplot as plt


# ПУНКТ 1: Визначення підінтегральної функції f(x) та інтервалу [a, b]
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)

a, b = 0, 24

x_vals = np.linspace(a, b, 500)
plt.figure(figsize=(8, 4))
plt.plot(x_vals, f(x_vals), label="f(x)")
plt.title("Графік інтенсивності навантаження")
plt.grid(True)
plt.show()


# ПУНКТ 3: Функція для обчислення інтегралу складовою формулою Сімпсона
def simpson_rule(f, a, b, N):
    if N % 2 != 0: N += 1
    h = (b - a) / N

    res = f(a) + f(b)
    for i in range(1, N, 2):
        res += 4 * f(a + i * h)
    for i in range(2, N, 2):
        res += 2 * f(a + i * h)

    return (h / 3) * res


# ПУНКТ 2: Знаходження точного значення інтегралу I0
I0 = simpson_rule(f, a, b, 1000000)
print(f"Пункт 2: Точне значення інтегралу I0 = {I0:.12f}")

# ПУНКТ 4: Дослідження залежності точності від N та пошук N_opt
N_range = range(10, 10000, 10)
errors = []
target_eps = 1e-12
N_opt = None

for N in N_range:
    current_I = simpson_rule(f, a, b, N)
    error = abs(current_I - I0)
    errors.append(error)
    if error <= target_eps and N_opt is None:
        N_opt = N

plt.figure(figsize=(8, 4))
plt.plot(list(N_range), errors)
plt.yscale('log')
plt.axhline(y=target_eps, color='r', linestyle='--', label='eps = 1e-12')
plt.title("Залежність похибки від N")
plt.xlabel("N")
plt.ylabel("Похибка")
plt.legend()
plt.grid(True)
plt.show()

if N_opt:
    print(f"Пункт 4: N_opt = {N_opt}, досягнута точність = {errors[N_range.index(N_opt)]:.2e}")
else:
    N_opt = 1000
    print(f"Пункт 4: Точність {target_eps} не досягнута при N <= 1000. N_opt встановлено на {N_opt}")

# ПУНКТ 5: Обчислення похибки при N0 = N_opt / 10 (кратне 8)
N0 = (max(8, (N_opt // 10)) // 8) * 8
I_N0 = simpson_rule(f, a, b, N0)
eps0 = abs(I_N0 - I0)
print(f"Пункт 5: N0 = {N0}, eps0 = {eps0:.2e}")

# ПУНКТ 6: Метод Рунге-Ромберга
I_N0_half = simpson_rule(f, a, b, N0 // 2)
IR = I_N0 + (I_N0 - I_N0_half) / 15
epsR = abs(IR - I0)
print(f"Пункт 6: Значення за Рунге-Ромбергом IR = {IR:.12f}, epsR = {epsR:.2e}")

# ПУНКТ 7: Метод Ейткена та оцінка порядку точності p
I1 = simpson_rule(f, a, b, N0 // 4)
I2 = simpson_rule(f, a, b, N0 // 2)
I3 = I_N0
IA = (I2 ** 2 - I1 * I3) / (2 * I2 - (I1 + I3))
epsA = abs(IA - I0)

diff_numerator = I3 - I2
diff_denominator = I2 - I1

if abs(diff_numerator) < 1e-15 or abs(diff_denominator) < 1e-15:
    p_aitken = 4.0  # Теоретичний порядок точності для методу Сімпсона
    print(f"Пункт 7: Значення за Ейткеном IA = {IA:.12f}, epsA = {epsA:.2e}, p = {p_aitken:.4f} (встановлено теоретичне, через досягнення маш. точності)")
else:
    p_aitken = np.log(abs(diff_numerator / diff_denominator)) / np.log(2)
    print(f"Пункт 7: Значення за Ейткеном IA = {IA:.12f}, epsA = {epsA:.2e}, p = {p_aitken:.4f}")

# ПУНКТ 9: Адаптивний алгоритм
def adaptive_simpson(f, a, b, eps):
    def step(a, b, eps, whole):
        mid = (a + b) / 2
        left = simpson_rule(f, a, mid, 2)
        right = simpson_rule(f, mid, b, 2)
        if abs(whole - (left + right)) <= 15 * eps:
            return left + right + (left + right - whole) / 15
        return step(a, mid, eps / 2, left) + step(mid, b, eps / 2, right)

    return step(a, b, eps, simpson_rule(f, a, b, 2))


print("\nПункт 9: Адаптивний алгоритм")
for tol in [1e-3, 1e-6, 1e-9]:
    res_adapt = adaptive_simpson(f, a, b, tol)
    print(f"Задана точність: {tol}, Отримане значення: {res_adapt:.12f}, Відхилення: {abs(res_adapt - I0):.2e}")