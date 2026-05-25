""" 
Лабораторна робота №10 
Ч.1: Метод прогнозу та корекції Адамса (2-го порядку) 
Ч.2: Метод Рунге-Кутта 4-го порядку 
Рівняння: y' = f(x, y)  на [x0, xN] з початковою умовою y(x0) = y0 

Варіант 26: 
  y' = y - 2x/y        (нелінійне рівняння) 
  Аналітичний розв'язок: y(x) = sqrt(1 + 2x) 
  Відрізок: [0, 1],  y(0) = 1 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# РІВНЯННЯ ТА ТОЧНИЙ РОЗВ'ЯЗОК 
# ───────────────────────────────────────────────────────────── 

def f(x, y):
    """Права частина ОДУ: y' = y - 2x/y"""
    return y - 2 * x / y


def exact(x):
    """Точний розв'язок: y(x) = sqrt(1 + 2x)"""
    return np.sqrt(1 + 2 * x)


X0, XN = 0.0, 1.0  # відрізок
Y0 = 1.0  # початкова умова y(0) = 1
EPS = 1e-5  # задана точність


# ─────────────────────────────────────────────────────────────
# ЗАВДАННЯ 1: Аналітичний розв'язок 
# ───────────────────────────────────────────────────────────── 

def task1_analytical():
    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 1: Аналітичний розв'язок")
    print("=" * 60)
    print("  Рівняння : y' = y - 2x/y")
    print("  y(0) = 1")
    print("  Розв'язок: y = sqrt(1 + 2x)")
    print()

    x_vals = np.linspace(X0, XN, 300)
    y_vals = exact(x_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    ax.plot(x_vals, y_vals, color='#4fc3f7', lw=2.5, label=r"$y = \sqrt{1+2x}$")
    ax.scatter([X0], [Y0], color='#ffd54f', s=100, zorder=5, label=f"y({X0}) = {Y0}")

    ax.set_title("Завдання 1 — Аналітичний розв'язок y' = y − 2x/y",
                 color='white', fontsize=13)
    ax.set_xlabel('x', color='#aaa');
    ax.set_ylabel('y', color='#aaa')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=11)
    ax.grid(color='#2a2d3e', lw=0.6)
    plt.tight_layout()
    plt.savefig('task1_analytical.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Графік збережено: task1_analytical.png")


# ───────────────────────────────────────────────────────────── 
# МЕТОД РУНГЕ-КУТТА 4-го порядку (стартер для Адамса) 
# ───────────────────────────────────────────────────────────── 

def runge_kutta4(f, x0, y0, h, n_steps):
    """ 
    RK4: робить n_steps кроків від x0 з кроком h. 
    Повертає масиви x, y. 
    """
    x = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    x[0], y[0] = x0, y0
    for i in range(n_steps):
        xi, yi = x[i], y[i]
        k1 = h * f(xi, yi)
        k2 = h * f(xi + h / 2, yi + k1 / 2)
        k3 = h * f(xi + h / 2, yi + k2 / 2)
        k4 = h * f(xi + h, yi + k3)
        y[i + 1] = yi + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[i + 1] = xi + h
    return x, y


# ───────────────────────────────────────────────────────────── 
# Ч.1 ЗАВДАННЯ 2: Метод прогнозу та корекції Адамса 2-го порядку 
# ───────────────────────────────────────────────────────────── 

def adams2(f, x0, y0, xN, h, max_iter=50, tol=1e-12):
    """ 
    Метод прогнозу та корекції Адамса 2-го порядку. 

    Прогноз:  y*_{n+1} = y_n + h/2*(3f_n - f_{n-1}) 
    Корекція: y_{n+1}  = y_n + h/2*(f(x_{n+1}, y*) + f_n) 
    """
    N = int(round((xN - x0) / h))
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    h_arr = np.zeros(N)

    # Перший крок — RK4 
    x_rk, y_rk = runge_kutta4(f, x0, y0, h, 1)
    x[0], y[0] = x_rk[0], y_rk[0]
    x[1], y[1] = x_rk[1], y_rk[1]

    for i in range(1, N):
        xi, yi = x[i], y[i]
        xi1 = xi + h

        fi = f(xi, yi)
        fi_1 = f(x[i - 1], y[i - 1])

        # Прогноз (Адамс екстраполяція 2-го порядку) 
        y_pred = yi + h / 2 * (3 * fi - fi_1)

        # Корекція (ітерації) 
        y_corr = y_pred
        for _ in range(max_iter):
            y_new = yi + h / 2 * (f(xi1, y_corr) + fi)
            if abs(y_new - y_corr) < tol:
                y_corr = y_new
                break
            y_corr = y_new

        x[i + 1] = xi1
        y[i + 1] = y_corr
        h_arr[i] = h

    return x, y


# ───────────────────────────────────────────────────────────── 
# Ч.1 ЗАВДАННЯ 2: чисельний розв'язок Адамса 
# ───────────────────────────────────────────────────────────── 

def task2_adams_solve(h=0.1):
    print("\n" + "=" * 60)
    print(f"ЗАВДАННЯ 2: Метод Адамса 2-го порядку (h = {h})")
    print("=" * 60)
    x, y = adams2(f, X0, Y0, XN, h)
    y_ex = exact(x)

    print(f"  {'x':>8}  {'y_adams':>14}  {'y_exact':>14}  {'|помилка|':>14}")
    print("  " + "-" * 56)
    for xi, yi, ye in zip(x, y, y_ex):
        print(f"  {xi:>8.4f}  {yi:>14.8f}  {ye:>14.8f}  {abs(yi - ye):>14.2e}")
    return x, y


# ───────────────────────────────────────────────────────────── 
# Ч.1 ЗАВДАННЯ 3: Графік локальної похибки (точне значення) 
# ───────────────────────────────────────────────────────────── 

def task3_local_error_exact(h=0.1):
    print("\n" + "=" * 60)
    print(f"ЗАВДАННЯ 3: Локальна похибка Адамса (точне значення, h={h})")
    print("=" * 60)
    x, y = adams2(f, X0, Y0, XN, h)
    err = np.abs(y - exact(x))

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')
    ax.semilogy(x, err + 1e-20, color='#f48fb1', lw=2, marker='o', markersize=4)
    ax.set_title(f'Завдання 3 — Локальна похибка Адамса 2 (h={h})',
                 color='white', fontsize=12)
    ax.set_xlabel('x', color='#aaa');
    ax.set_ylabel('|y_adams − y_exact|', color='#aaa')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.grid(color='#2a2d3e', lw=0.6, which='both')
    plt.tight_layout()
    plt.savefig('task3_adams_error_exact.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Графік збережено: task3_adams_error_exact.png")
    return x, err


# ───────────────────────────────────────────────────────────── 
# Ч.1 ЗАВДАННЯ 4: Оцінка похибки по Рунге (Адамс) 
# ───────────────────────────────────────────────────────────── 

def runge_error_adams(f, x0, y0, xN, h, order=2):
    """Оцінка локальної похибки методом Рунге для Адамса."""
    x1, y1 = adams2(f, x0, y0, xN, h)
    x2, y2 = adams2(f, x0, y0, xN, h / 2)
    # Інтерполюємо y2 на сітку x1 
    y2_interp = np.interp(x1, x2, y2)
    err = np.abs(y1 - y2_interp) / (2 ** order - 1)
    return x1, err, y1


def task4_adams_runge_error(h=0.1):
    print("\n" + "=" * 60)
    print(f"ЗАВДАННЯ 4: Оцінка похибки по Рунге (Адамс, h={h})")
    print("=" * 60)
    x, err_runge, y = runge_error_adams(f, X0, Y0, XN, h)
    err_exact = np.abs(y - exact(x))

    print(f"  Макс. похибка (Рунге)  : {err_runge.max():.3e}")
    print(f"  Макс. похибка (точна)  : {err_exact.max():.3e}")
    print(f"  Оптимальний крок ~ {h * (EPS / err_runge.max()) ** (1 / 2):.5f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')
    ax.semilogy(x, err_runge + 1e-20, color='#4fc3f7', lw=2, label='Рунге оцінка')
    ax.semilogy(x, err_exact + 1e-20, color='#f48fb1', lw=2, linestyle='--', label='Точна похибка')
    ax.set_title(f'Завдання 4 — Порівняння оцінок похибки Адамса (h={h})',
                 color='white', fontsize=12)
    ax.set_xlabel('x', color='#aaa');
    ax.set_ylabel('похибка', color='#aaa')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=10)
    ax.grid(color='#2a2d3e', lw=0.6, which='both')
    plt.tight_layout()
    plt.savefig('task4_adams_runge_error.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Графік збережено: task4_adams_runge_error.png")


# ───────────────────────────────────────────────────────────── 
# Ч.1 ЗАВДАННЯ 5: Автоматичний вибір кроку (Адамс) 
# ───────────────────────────────────────────────────────────── 

def adams2_adaptive(f, x0, y0, xN, eps=EPS, h_init=0.1, order=2):
    """Адамс 2-го порядку з автоматичним вибором кроку (метод Рунге)."""
    C = 0.9  # безпечний множник 
    x_list = [x0]
    y_list = [y0]
    h_list = []
    h = h_init
    x_cur, y_cur = x0, y0

    while x_cur < xN - 1e-12:
        h = min(h, xN - x_cur)

        # Один крок з кроком h 
        x1s, y1s = adams2(f, x_cur, y_cur, x_cur + h, h)
        y_h = y1s[-1]

        # Два кроки з кроком h/2 
        x2s, y2s = adams2(f, x_cur, y_cur, x_cur + h, h / 2)
        y_h2 = y2s[-1]

        err = abs(y_h - y_h2) / (2 ** order - 1)

        if err < eps:
            x_cur += h
            y_cur = y_h2  # краща оцінка
            x_list.append(x_cur)
            y_list.append(y_cur)
            h_list.append(h)
            if err < eps / 4 and h < (xN - x_cur):
                h *= 2  # збільшуємо крок
        else:
            h /= 2  # зменшуємо крок

    return (np.array(x_list), np.array(y_list), np.array(h_list))


def task5_adams_adaptive():
    print("\n" + "=" * 60)
    print(f"ЗАВДАННЯ 5: Автоматичний вибір кроку (Адамс, eps={EPS:.0e})")
    print("=" * 60)
    x, y, h_arr = adams2_adaptive(f, X0, Y0, XN, eps=EPS)
    err = np.abs(y - exact(x))
    print(f"  Кількість вузлів: {len(x)}")
    print(f"  Макс. похибка   : {err.max():.3e}")
    print(f"  Мін. крок       : {h_arr.min():.5f}")
    print(f"  Макс. крок      : {h_arr.max():.5f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0f1117')

        # Розв'язок
    x_fine = np.linspace(X0, XN, 400)
    ax1.plot(x_fine, exact(x_fine), color='#ffd54f', lw=2, label='Точний розв\'язок')
    ax1.plot(x, y, 'o-', color='#4fc3f7', lw=1.5, markersize=4, label='Адамс адаптив.')
    ax1.set_title('Розв\'язок (адаптивний крок)', color='white', fontsize=11)
    ax1.set_xlabel('x', color='#aaa');
    ax1.set_ylabel('y', color='#aaa')
    ax1.tick_params(colors='#aaa')
    for sp in ax1.spines.values(): sp.set_edgecolor('#444')
    ax1.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=9)
    ax1.grid(color='#2a2d3e', lw=0.6)

    # Крок 
    ax2.step(x[1:], h_arr, where='pre', color='#69f0ae', lw=2)
    ax2.set_title('Залежність кроку h(x)', color='white', fontsize=11)
    ax2.set_xlabel('x', color='#aaa');
    ax2.set_ylabel('h', color='#aaa')
    ax2.tick_params(colors='#aaa')
    for sp in ax2.spines.values(): sp.set_edgecolor('#444')
    ax2.grid(color='#2a2d3e', lw=0.6)

    plt.suptitle(f'Завдання 5 — Адамс адаптивний (eps={EPS:.0e})',
                 color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('task5_adams_adaptive.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Графік збережено: task5_adams_adaptive.png")


# ───────────────────────────────────────────────────────────── 
# Ч.2 ЗАВДАННЯ 6: Рунге-Кутта 4-го порядку 
# ───────────────────────────────────────────────────────────── 

def task6_rk4_solve(h=0.1):
    print("\n" + "=" * 60)
    print(f"ЗАВДАННЯ 6: Рунге-Кутта 4-го порядку (h={h})")
    print("=" * 60)
    N = int(round((XN - X0) / h))
    x, y = runge_kutta4(f, X0, Y0, h, N)
    y_ex = exact(x)

    print(f"  {'x':>8}  {'y_rk4':>14}  {'y_exact':>14}  {'|помилка|':>14}")
    print("  " + "-" * 56)
    for xi, yi, ye in zip(x, y, y_ex):
        print(f"  {xi:>8.4f}  {yi:>14.8f}  {ye:>14.8f}  {abs(yi - ye):>14.2e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')
    x_fine = np.linspace(X0, XN, 400)
    ax.plot(x_fine, exact(x_fine), color='#ffd54f', lw=2.5, label='Точний розв\'язок')
    ax.plot(x, y, 'o-', color='#4fc3f7', lw=1.8, markersize=5, label='RK4')
    ax.set_title(f'Завдання 6 — RK4 (h={h})', color='white', fontsize=12)
    ax.set_xlabel('x', color='#aaa');
    ax.set_ylabel('y', color='#aaa')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=11)
    ax.grid(color='#2a2d3e', lw=0.6)
    plt.tight_layout()
    plt.savefig('task6_rk4_solve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Графік збережено: task6_rk4_solve.png")
    return x, y


# ───────────────────────────────────────────────────────────── 
# Ч.2 ЗАВДАННЯ 7: Локальна похибка RK4 (точне значення) 
# ───────────────────────────────────────────────────────────── 

def task7_rk4_error_exact():
    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 7: Локальна похибка RK4 (залежність від h)")
    print("=" * 60)
    h_values = [0.2, 0.1, 0.05, 0.025]
    colors = ['#f48fb1', '#4fc3f7', '#69f0ae', '#ffd54f']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f1117')
    for ax in (ax1, ax2): ax.set_facecolor('#0f1117')

    for h, c in zip(h_values, colors):
        N = int(round((XN - X0) / h))
        x, y = runge_kutta4(f, X0, Y0, h, N)
        err = np.abs(y - exact(x))
        ax1.semilogy(x, err + 1e-20, color=c, lw=1.8, marker='o', markersize=3,
                     label=f'h={h}')
        print(f"  h={h:.3f}  max_err={err.max():.3e}")

    ax1.set_title('Похибка RK4 при різних h', color='white', fontsize=11)
    ax1.set_xlabel('x', color='#aaa');
    ax1.set_ylabel('|y_rk4 − y_exact|', color='#aaa')
    ax1.tick_params(colors='#aaa')
    for sp in ax1.spines.values(): sp.set_edgecolor('#444')
    ax1.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=9)
    ax1.grid(color='#2a2d3e', lw=0.6, which='both')

    # Порядок збіжності 
    max_errs = []
    for h in h_values:
        N = int(round((XN - X0) / h))
        x, y = runge_kutta4(f, X0, Y0, h, N)
        max_errs.append(np.abs(y - exact(x)).max())

    ax2.loglog(h_values, max_errs, 'o-', color='#4fc3f7', lw=2, markersize=7, label='RK4')
    # Теоретична лінія O(h^4) 
    h_ref = np.array(h_values)
    ax2.loglog(h_ref, max_errs[0] * (h_ref / h_values[0]) ** 4, '--',
               color='#ffd54f', lw=1.5, label='O(h⁴) теорія')
    ax2.set_title('Порядок збіжності RK4', color='white', fontsize=11)
    ax2.set_xlabel('h', color='#aaa');
    ax2.set_ylabel('max|помилка|', color='#aaa')
    ax2.tick_params(colors='#aaa')
    for sp in ax2.spines.values(): sp.set_edgecolor('#444')
    ax2.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=9)
    ax2.grid(color='#2a2d3e', lw=0.6, which='both')

    plt.suptitle('Завдання 7 — Дослідження похибки RK4',
                 color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('task7_rk4_error.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Графік збережено: task7_rk4_error.png")


# ───────────────────────────────────────────────────────────── 
# Ч.2 ЗАВДАННЯ 8: Оцінка похибки RK4 по методу Рунге 
# ───────────────────────────────────────────────────────────── 

def runge_error_rk4(f, x0, y0, xN, h, order=4):
    """Оцінка локальної похибки RK4 методом Рунге."""
    N1 = int(round((xN - x0) / h))
    N2 = int(round((xN - x0) / (h / 2)))
    x1, y1 = runge_kutta4(f, x0, y0, h, N1)
    x2, y2 = runge_kutta4(f, x0, y0, h / 2, N2)
    y2_interp = np.interp(x1, x2, y2)
    err = np.abs(y1 - y2_interp) / (2 ** order - 1)
    return x1, err, y1


def task8_rk4_runge_error(h=0.1):
    print("\n" + "=" * 60)
    print(f"ЗАВДАННЯ 8: Оцінка похибки RK4 по Рунге (h={h})")
    print("=" * 60)
    x, err_runge, y = runge_error_rk4(f, X0, Y0, XN, h)
    err_exact = np.abs(y - exact(x))
    h_opt = h * (EPS / (err_runge.max() + 1e-30)) ** (1 / 4)

    print(f"  Макс. похибка (Рунге) : {err_runge.max():.3e}")
    print(f"  Макс. похибка (точна) : {err_exact.max():.3e}")
    print(f"  Оцінка оптимального h : {h_opt:.6f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')
    ax.semilogy(x, err_runge + 1e-20, color='#4fc3f7', lw=2, label='Рунге оцінка')
    ax.semilogy(x, err_exact + 1e-20, color='#f48fb1', lw=2, linestyle='--', label='Точна похибка')
    ax.axhline(EPS, color='#ffd54f', lw=1.5, linestyle=':', label=f'ε={EPS:.0e}')
    ax.set_title(f'Завдання 8 — Похибка RK4 по Рунге (h={h})',
                 color='white', fontsize=12)
    ax.set_xlabel('x', color='#aaa');
    ax.set_ylabel('похибка', color='#aaa')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=10)
    ax.grid(color='#2a2d3e', lw=0.6, which='both')
    plt.tight_layout()
    plt.savefig('task8_rk4_runge_error.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Графік збережено: task8_rk4_runge_error.png")


# ───────────────────────────────────────────────────────────── 
# Ч.2 ЗАВДАННЯ 9: Автоматичний вибір кроку для RK4 
# ───────────────────────────────────────────────────────────── 

def rk4_adaptive(f, x0, y0, xN, eps=EPS, h_init=0.2, order=4):
    """RK4 з автоматичним вибором кроку (метод Рунге)."""
    C = 0.9
    x_list = [x0];
    y_list = [y0];
    h_list = []
    h = h_init
    x_cur, y_cur = x0, y0

    while x_cur < xN - 1e-12:
        h = min(h, xN - x_cur)

        # Крок h 
        x1s, y1s = runge_kutta4(f, x_cur, y_cur, h, 1)
        y_h = y1s[-1]

        # Два кроки h/2 
        x2s, y2s = runge_kutta4(f, x_cur, y_cur, h / 2, 2)
        y_h2 = y2s[-1]

        err = abs(y_h - y_h2) / (2 ** order - 1)

        if err == 0:
            err = 1e-30

            # Новий крок
        h_new = C * h * (eps / err) ** (1 / (order + 1))

        if err <= eps:
            x_cur += h
            y_cur = y_h2
            x_list.append(x_cur)
            y_list.append(y_cur)
            h_list.append(h)
            h = min(h_new, 2 * h)  # не збільшуємо більше ніж вдвічі
        else:
            h = max(h_new, h / 4)  # не зменшуємо більше ніж вчетверо

    return np.array(x_list), np.array(y_list), np.array(h_list)


def task9_rk4_adaptive():
    print("\n" + "=" * 60)
    print(f"ЗАВДАННЯ 9: RK4 з автоматичним вибором кроку (eps={EPS:.0e})")
    print("=" * 60)
    x, y, h_arr = rk4_adaptive(f, X0, Y0, XN, eps=EPS)
    err = np.abs(y - exact(x))
    print(f"  Кількість вузлів: {len(x)}")
    print(f"  Макс. похибка   : {err.max():.3e}")
    print(f"  Мін. крок       : {h_arr.min():.6f}")
    print(f"  Макс. крок      : {h_arr.max():.6f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f1117')
    for ax in (ax1, ax2): ax.set_facecolor('#0f1117')

    x_fine = np.linspace(X0, XN, 400)
    ax1.plot(x_fine, exact(x_fine), color='#ffd54f', lw=2.5, label='Точний розв\'язок')
    ax1.plot(x, y, 'o-', color='#4fc3f7', lw=1.8, markersize=4, label='RK4 адаптив.')
    ax1.set_title('Розв\'язок (адаптивний крок)', color='white', fontsize=11)
    ax1.set_xlabel('x', color='#aaa');
    ax1.set_ylabel('y', color='#aaa')
    ax1.tick_params(colors='#aaa')
    for sp in ax1.spines.values(): sp.set_edgecolor('#444')
    ax1.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=9)
    ax1.grid(color='#2a2d3e', lw=0.6)

    ax2.step(x[1:], h_arr, where='pre', color='#69f0ae', lw=2)
    ax2.set_title('Залежність кроку h(x)', color='white', fontsize=11)
    ax2.set_xlabel('x', color='#aaa');
    ax2.set_ylabel('h', color='#aaa')
    ax2.tick_params(colors='#aaa')
    for sp in ax2.spines.values(): sp.set_edgecolor('#444')
    ax2.grid(color='#2a2d3e', lw=0.6)

    plt.suptitle(f'Завдання 9 — RK4 адаптивний (eps={EPS:.0e})',
                 color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('task9_rk4_adaptive.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Графік збережено: task9_rk4_adaptive.png")


# ───────────────────────────────────────────────────────────── 
# ДОДАТКОВО: Порівняльний графік Адамс vs RK4 
# ───────────────────────────────────────────────────────────── 

def extra_comparison():
    print("\n" + "=" * 60)
    print("ДОДАТКОВО: Порівняння Адамс 2-го порядку vs RK4")
    print("=" * 60)
    h = 0.1
    N = int(round((XN - X0) / h))
    x_fine = np.linspace(X0, XN, 400)

    x_rk, y_rk = runge_kutta4(f, X0, Y0, h, N)
    x_ad, y_ad = adams2(f, X0, Y0, XN, h)
    err_rk = np.abs(y_rk - exact(x_rk))
    err_ad = np.abs(y_ad - exact(x_ad))

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0f1117')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # 1) Розв'язки 
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#0f1117')
    ax1.plot(x_fine, exact(x_fine), color='#ffd54f', lw=2.5, label='Точний')
    ax1.plot(x_rk, y_rk, 'o-', color='#4fc3f7', lw=1.8, markersize=4, label='RK4')
    ax1.plot(x_ad, y_ad, 's--', color='#f48fb1', lw=1.8, markersize=4, label='Адамс 2')
    ax1.set_title(f'Порівняння методів (h={h})', color='white', fontsize=13)
    ax1.set_xlabel('x', color='#aaa');
    ax1.set_ylabel('y', color='#aaa')
    ax1.tick_params(colors='#aaa')
    for sp in ax1.spines.values(): sp.set_edgecolor('#444')
    ax1.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=11)
    ax1.grid(color='#2a2d3e', lw=0.6)

    # 2) Похибки 
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#0f1117')
    ax2.semilogy(x_rk, err_rk + 1e-20, color='#4fc3f7', lw=2, label='RK4')
    ax2.semilogy(x_ad, err_ad + 1e-20, color='#f48fb1', lw=2, linestyle='--', label='Адамс 2')
    ax2.set_title('Локальна похибка', color='white', fontsize=11)
    ax2.set_xlabel('x', color='#aaa');
    ax2.set_ylabel('|помилка|', color='#aaa')
    ax2.tick_params(colors='#aaa')
    for sp in ax2.spines.values(): sp.set_edgecolor('#444')
    ax2.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=9)
    ax2.grid(color='#2a2d3e', lw=0.6, which='both')
    print(f"  RK4   max_err = {err_rk.max():.3e}")
    print(f"  Адамс max_err = {err_ad.max():.3e}")

    # 3) Порядок збіжності обох 
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#0f1117')
    h_vals = [0.2, 0.1, 0.05, 0.025, 0.0125]
    errs_rk, errs_ad = [], []
    for hv in h_vals:
        Nv = int(round((XN - X0) / hv))
        xr, yr = runge_kutta4(f, X0, Y0, hv, Nv)
        xa, ya = adams2(f, X0, Y0, XN, hv)
        errs_rk.append(np.abs(yr - exact(xr)).max())
        errs_ad.append(np.abs(ya - exact(xa)).max())
    ax3.loglog(h_vals, errs_rk, 'o-', color='#4fc3f7', lw=2, label='RK4')
    ax3.loglog(h_vals, errs_ad, 's-', color='#f48fb1', lw=2, label='Адамс 2')
    h_ref = np.array(h_vals)
    ax3.loglog(h_ref, errs_rk[0] * (h_ref / h_vals[0]) ** 4, ':',
               color='#4fc3f7', lw=1.2, label='O(h⁴)')
    ax3.loglog(h_ref, errs_ad[0] * (h_ref / h_vals[0]) ** 2, ':',
               color='#f48fb1', lw=1.2, label='O(h²)')
    ax3.set_title('Порядок збіжності', color='white', fontsize=11)
    ax3.set_xlabel('h', color='#aaa');
    ax3.set_ylabel('max|помилка|', color='#aaa')
    ax3.tick_params(colors='#aaa')
    for sp in ax3.spines.values(): sp.set_edgecolor('#444')
    ax3.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=8)
    ax3.grid(color='#2a2d3e', lw=0.6, which='both')

    plt.suptitle('Додатково: RK4 vs Адамс 2-го порядку — повне порівняння',
                 color='white', fontsize=14, y=1.01)
    plt.savefig('extra_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Графік збережено: extra_comparison.png")


# ───────────────────────────────────────────────────────────── 
# ГОЛОВНА ФУНКЦІЯ 
# ───────────────────────────────────────────────────────────── 

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Лабораторна робота №10 — Методи Рунге-Кутта та Адамса ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Рівняння : y' = y − 2x/y")
    print(f"  Відрізок : [{X0}, {XN}],  y({X0}) = {Y0}")
    print(f"  Точний розв'язок: y(x) = sqrt(1 + 2x)")
    print(f"  Задана точність : eps = {EPS:.0e}")

    # ── Ч.1 ── 
    task1_analytical()
    task2_adams_solve(h=0.1)
    task3_local_error_exact(h=0.1)
    task4_adams_runge_error(h=0.1)
    task5_adams_adaptive()

    # ── Ч.2 ── 
    task6_rk4_solve(h=0.1)
    task7_rk4_error_exact()
    task8_rk4_runge_error(h=0.1)
    task9_rk4_adaptive()

    # ── Додатково ── 
    extra_comparison()

    print("\n" + "=" * 60)
    print("Всі завдання виконано!")
    print("Збережені файли:")
    for fn in [
        'task1_analytical.png',
        'task3_adams_error_exact.png',
        'task4_adams_runge_error.png',
        'task5_adams_adaptive.png',
        'task6_rk4_solve.png',
        'task7_rk4_error.png',
        'task8_rk4_runge_error.png',
        'task9_rk4_adaptive.png',
        'extra_comparison.png',
    ]:
        print(f"  • {fn}")
    print("=" * 60)