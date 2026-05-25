import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# ВХІДНІ ДАНІ ТА ТЕСТОВЕ ДИФЕРЕНЦІАЛЬНЕ РІВНЯННЯ (Задачі Коші)
# =====================================================================
def f(x, y):
    return y - x


def exact_sol(x):
    return 0.5 * np.exp(x) + x + 1


x0, xN = 0.0, 2.0
y0 = 1.5
h_fixed = 0.1
eps = 1e-5


def d2f_dx2(x, y):
    return f(x, y) - 1


# =====================================================================
# ДОПОМІЖНІ МЕТОДИ
# =====================================================================
def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# =====================================================================
# ЧАСТИНА 1: МЕТОД ПРОГНОЗУ ТА КОРЕКЦІЇ АДАМСА 2-ГО ПОРЯДКУ
# =====================================================================
def adams_pc_fixed(x0, xN, y0, h):
    x_list = [x0]
    y_list = [y0]

    x_list.append(x0 + h)
    y_list.append(rk4_step(x0, y0, h))

    steps = int((xN - x0) / h)

    for n in range(1, steps):
        x_n = x_list[n]
        x_nm1 = x_list[n - 1]
        y_n = y_list[n]
        y_nm1 = y_list[n - 1]

        f_n = f(x_n, y_n)
        f_nm1 = f(x_nm1, y_nm1)
        y_next_pred = y_n + (h / 2.0) * (3 * f_n - f_nm1)

        y_corr = y_next_pred
        for _ in range(2):
            y_corr = y_n + (h / 2.0) * (f(x_n + h, y_corr) + f_n)

        x_list.append(x_n + h)
        y_list.append(y_corr)

    return np.array(x_list), np.array(y_list)


def adams_pc_auto(x0, xN, y0, eps):
    x_list = [x0]
    y_list = [y0]
    h_list = []

    h = 0.05
    x = x0
    y = y0

    x_next = x + h
    y_next = rk4_step(x, y, h)
    x_list.append(x_next)
    y_list.append(y_next)
    h_list.append(h)

    while x_list[-1] < xN:
        x_n = x_list[-1]
        x_nm1 = x_list[-2]
        y_n = y_list[-1]
        y_nm1 = y_list[-2]

        if x_n + h > xN:
            h = xN - x_n

        f_n = f(x_n, y_n)
        f_nm1 = f(x_nm1, y_nm1)
        y_pred = y_n + (h / 2.0) * (3 * f_n - f_nm1)

        y_corr = y_pred
        for _ in range(2):
            y_corr = y_n + (h / 2.0) * (f(x_n + h, y_corr) + f_n)

        y_triple_prime = d2f_dx2(x_n, y_n)
        err_est = abs(-(h ** 3) / 12.0 * y_triple_prime)

        if err_est > eps:
            h /= 2.0
            x_list[-1] = x_list[-2] + h
            y_list[-1] = rk4_step(x_list[-2], y_list[-2], h)
        else:
            x_list.append(x_n + h)
            y_list.append(y_corr)
            h_list.append(h)
            if err_est < eps / 8.0:
                h *= 2.0

    return np.array(x_list), np.array(y_list), np.array(h_list)


# =====================================================================
# ЧАСТИНА 2: МЕТОД РУНГЕ-КУТТА 4-ГО ПОРЯДКУ (РК4)
# =====================================================================
def rk4_fixed(x0, xN, y0, h):
    steps = int((xN - x0) / h)
    x_list = [x0]
    y_list = [y0]

    for _ in range(steps):
        x = x_list[-1]
        y = y_list[-1]
        y_next = rk4_step(x, y, h)
        x_list.append(x + h)
        y_list.append(y_next)

    return np.array(x_list), np.array(y_list)


def rk4_auto(x0, xN, y0, eps):
    x_list = [x0]
    y_list = [y0]
    h_list = [0.0]

    h = 0.1
    x = x0
    y = y0

    while x < xN:
        if x + h > xN:
            h = xN - x

        y_h = rk4_step(x, y, h)

        y_h2_mid = rk4_step(x, y, h / 2.0)
        y_h2 = rk4_step(x + h / 2.0, y_h2_mid, h / 2.0)

        err = abs(y_h - y_h2) / 15.0

        if err <= eps:
            x += h
            y = y_h2
            x_list.append(x)
            y_list.append(y)
            h_list.append(h)

            if err < eps / 32.0:
                h *= 2.0
        else:
            h /= 2.0

    return np.array(x_list), np.array(y_list), np.array(h_list)


# =====================================================================
# ОБЧИСЛЕННЯ ТА ПОБУДОВА ГРАФІКІВ
# =====================================================================

x_exact = np.linspace(x0, xN, 200)
y_exact = exact_sol(x_exact)

x_adams, y_adams = adams_pc_fixed(x0, xN, y0, h_fixed)
x_rk4, y_rk4 = rk4_fixed(x0, xN, y0, h_fixed)

err_adams_exact = abs(y_adams - exact_sol(x_adams))
err_rk4_exact = abs(y_rk4 - exact_sol(x_rk4))
err_adams_theo = abs(-(h_fixed ** 3) / 12.0 * d2f_dx2(x_adams, y_adams))

plt.figure(figsize=(14, 10))

# Графік 1
plt.subplot(2, 2, 1)
plt.plot(x_exact, y_exact, 'k-', label="Аналітичний розв'язок", linewidth=2)
plt.plot(x_adams, y_adams, 'ro--', label="Адамс (П-К) 2-го пор.")
plt.plot(x_rk4, y_rk4, 'b*:', label="Рунге-Кутта 4-го пор.")
plt.title('Порівняння чисельних методів ($h = 0.1$)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True)

# Графік 2
plt.subplot(2, 2, 2)
plt.plot(x_adams, err_adams_exact, 'r-o', label='Фактична похибка')
plt.plot(x_adams, err_adams_theo, 'g--', label='Теоретична оцінка ($R_2^{kop}$)')
plt.title('Локальна похибка методу Адамса 2-го пор.')
plt.xlabel('$x$')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

# Графік 3
x_adams_auto, _, h_adams_auto = adams_pc_auto(x0, xN, y0, eps)
x_rk4_auto, _, h_rk4_auto = rk4_auto(x0, xN, y0, eps)

plt.subplot(2, 2, 3)
plt.step(x_adams_auto[1:], h_adams_auto, 'r-', where='post', label='Крок Адамса (П-К)')
plt.step(x_rk4_auto, h_rk4_auto, 'b-', where='post', label='Крок Рунге-Кутта 4')
plt.title(fr'Залежність величини кроку $h(x)$ від координати ($\epsilon = {eps}$)')
plt.xlabel('$x$')
plt.ylabel('Величина кроку $h$')
plt.legend()
plt.grid(True)

# Графік 4
plt.subplot(2, 2, 4)
plt.plot(x_rk4, err_rk4_exact, 'b-s', label='Фактична похибка РК4')
plt.title('Локальна похибка методу РК4 ($h = 0.1$)')
plt.xlabel('$x$')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("--- Результати в кінцевій точці xN = 2.0 ---")
print(f"Точний розв'язок:                {exact_sol(xN):.8f}")
print(f"Метод Адамса (сталий крок):      {y_adams[-1]:.8f} (Похибка: {err_adams_exact[-1]:.2e})")
print(f"Метод РК4 (сталий крок):         {y_rk4[-1]:.8f} (Похибка: {err_rk4_exact[-1]:.2e})")
print(f"Кількість точок Адамса (авто):  {len(x_adams_auto)}")
print(f"Кількість точок РК4 (авто):     {len(x_rk4_auto)}")