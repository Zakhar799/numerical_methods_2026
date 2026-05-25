import numpy as np
import matplotlib.pyplot as plt
import cmath


# ==========================================
# ЧАСТИНА 1: Трансцендентне рівняння
# ==========================================

def F_trans(x):
    return np.exp(x) - 3 * x ** 2


def dF_trans(x):
    return np.exp(x) - 6 * x


def d2F_trans(x):
    return np.exp(x) - 6


# ПУНКТ 1: Табуляція з кроком h та запис у файл
def tabulate_and_find_roots(a=-1.0, b=4.0, h=0.1):
    x_vals = np.arange(a, b + h, h)
    y_vals = F_trans(x_vals)

    # Запис у текстовий файл
    with open("tabulation.txt", "w", encoding="utf-8") as f:
        f.write("x\t\tF(x)\n")
        for x, y in zip(x_vals, y_vals):
            f.write(f"{x:.4f}\t{y:.4f}\n")

    roots_approx = []
    trends = []

    # Пошук точок зміни знаку
    for i in range(len(y_vals) - 1):
        if y_vals[i] * y_vals[i + 1] <= 0:
            x_root = (x_vals[i] + x_vals[i + 1]) / 2
            roots_approx.append(x_root)
            if y_vals[i] < y_vals[i + 1]:
                trends.append("Зростає")
            else:
                trends.append("Спадає")

    return roots_approx, trends


# ПУНКТ 2-4: Ітераційні методи з подвійною умовою зупинки
# Метод простої ітерації (релаксації)
def simple_iteration(x0, tau, eps=1e-5, max_iter=1000):
    x = x0
    for iters in range(1, max_iter + 1):
        x_next = x + tau * F_trans(x)
        # Подвійна умова згідно з пунктом 3
        if abs(F_trans(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, iters
        x = x_next
    return None, max_iter


# Метод Ньютона
def newton_method(x0, eps=1e-5, max_iter=1000):
    x = x0
    for iters in range(1, max_iter + 1):
        df = dF_trans(x)
        if df == 0: break
        x_next = x - F_trans(x) / df
        if abs(F_trans(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, iters
        x = x_next
    return None, max_iter


# Метод Чебишева (3 члени ряду Тейлора)
def chebyshev_method(x0, eps=1e-5, max_iter=1000):
    x = x0
    for iters in range(1, max_iter + 1):
        fx = F_trans(x)
        dfx = dF_trans(x)
        d2fx = d2F_trans(x)
        if dfx == 0: break
        # Формула зі сторінки 5 методички
        x_next = x - fx / dfx - 0.5 * (fx ** 2 * d2fx) / (dfx ** 3)
        if abs(F_trans(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, iters
        x = x_next
    return None, max_iter


# Метод хорд (багатокроковий)
def secant_method(x0, x1, eps=1e-5, max_iter=1000):
    for iters in range(1, max_iter + 1):
        fx0 = F_trans(x0)
        fx1 = F_trans(x1)
        if (fx1 - fx0) == 0: break
        x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(F_trans(x_next)) < eps and abs(x_next - x1) < eps:
            return x_next, iters
        x0, x1 = x1, x_next
    return None, max_iter


# Метод парабол (Мюллера)
def parabola_method(x0, x1, x2, eps=1e-5, max_iter=1000):
    for iters in range(1, max_iter + 1):
        f0, f1, f2 = F_trans(x0), F_trans(x1), F_trans(x2)

        # Розділені різниці (стор. 5 методички)
        f01 = (f1 - f0) / (x1 - x0)
        f12 = (f2 - f1) / (x2 - x1)
        f012 = (f12 - f01) / (x2 - x0)

        w = f12 + f012 * (x2 - x1)
        discr = cmath.sqrt(w ** 2 - 4 * f2 * f012)

        denom1 = w + discr
        denom2 = w - discr
        denom = denom1 if abs(denom1) > abs(denom2) else denom2

        if denom == 0: break
        delta = -2 * f2 / denom
        x_next = x2 + delta.real

        if abs(F_trans(x_next)) < eps and abs(x_next - x2) < eps:
            return x_next, iters
        x0, x1, x2 = x1, x2, x_next
    return None, max_iter


# Метод зворотної інтерполяції Лагранжа (по 3 точках)
def inverse_interpolation(x0, x1, x2, eps=1e-5, max_iter=1000):
    for iters in range(1, max_iter + 1):
        y0, y1, y2 = F_trans(x0), F_trans(x1), F_trans(x2)

        if (y0 - y1) == 0 or (y0 - y2) == 0 or (y1 - y2) == 0: break

        # Формула Лагранжа для y=0 (стор. 6 методички)
        term1 = (y1 * y2) / ((y0 - y1) * (y0 - y2)) * x0
        term2 = (y0 * y2) / ((y1 - y0) * (y1 - y2)) * x1
        term3 = (y0 * y1) / ((y2 - y0) * (y2 - y1)) * x2
        x_next = term1 + term2 + term3

        if abs(F_trans(x_next)) < eps and abs(x_next - x2) < eps:
            return x_next, iters
        x0, x1, x2 = x1, x2, x_next
    return None, max_iter


# ==========================================
# ЧАСТИНА 2: Алгебраїчне рівняння
# ==========================================

# ПУНКТ 5-6: Запис коефіцієнтів многочлена у файл
# Многочлен: x^3 - 2x^2 + x - 2 = 0 (Корені: 2 та ±1j)
def write_poly_coefs():
    coefs = [1.0, -2.0, 1.0, -2.0]  # a_3, a_2, a_1, a_0
    with open("poly_coefs.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, coefs)))


# ПУНКТ 7: Зчитування коефіцієнтів
def read_poly_coefs():
    with open("poly_coefs.txt", "r", encoding="utf-8") as f:
        return list(map(float, f.read().split()))


# ПУНКТ 7 (продовження): Обчислення значення многочлена за схемою Горнера
def horner_eval(coefs, x):
    res = coefs[0]
    for i in range(1, len(coefs)):
        res = res * x + coefs[i]
    return res


# ПУНКТ 8: Повна схема Горнера для P(x) та P'(x) згідно з методичкою
def horner_scheme_full(coefs, x_n):
    m = len(coefs) - 1
    b = np.zeros(m + 1)
    c = np.zeros(m)

    # Системи рівнянь зі сторінки 10 методички
    b[0] = coefs[0]
    for i in range(1, m + 1):
        b[i] = coefs[i] + x_n * b[i - 1]

    c[0] = b[0]
    for i in range(1, m):
        c[i] = b[i] + x_n * c[i - 1]

    return b[m], c[m - 1]  # Повертає P(x_n) та P'(x_n)


def newton_horner(coefs, x0, eps=1e-5, max_iter=100):
    x = x0
    for iters in range(1, max_iter + 1):
        px, dpx = horner_scheme_full(coefs, x)
        if dpx == 0: break
        x_next = x - px / dpx
        if abs(px) < eps and abs(x_next - x) < eps:
            return x_next, iters
        x = x_next
    return None, max_iter


# ПУНКТ 9: Класичний метод Ліна для знаходження комплексних коренів
def lin_method(coefs, p0, q0, eps=1e-5, max_iter=500):
    p, q = p0, q0
    # Наш многочлен: a3*x^3 + a2*x^2 + a1*x + a0
    a3, a2, a1, a0 = coefs[0], coefs[1], coefs[2], coefs[3]

    for iters in range(1, max_iter + 1):
        # Ділення многочлена третього степеня на x^2 + px + q
        b3 = a3
        b2 = a2 - p * b3

        # Залишок (Лінійна частина)
        R = a1 - p * b2 - q * b3
        S = a0 - q * b2

        if abs(R) < eps and abs(S) < eps:
            break

        # Уточнення коефіцієнтів p та q (ітераційна процедура Ліна)
        if b2 == 0: break
        p_new = (a1 - q * b3) / b2
        q_new = a0 / b2

        if abs(p_new - p) < eps and abs(q_new - q) < eps:
            p, q = p_new, q_new
            break
        p, q = p_new, q_new

    # Знаходимо комплексні корені з квадратного рівняння x^2 + px + q = 0
    D = p ** 2 - 4 * q
    root1 = (-p + cmath.sqrt(D)) / 2
    root2 = (-p - cmath.sqrt(D)) / 2
    return root1, root2, iters


# ==========================================
# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
# ==========================================
if __name__ == "__main__":
    print("=============================================")
    print("   РЕЗУЛЬТАТИ ВИКОНАННЯ ЛАБОРАТОРНОЇ РОБОТИ   ")
    print("=============================================\n")

    # 1. Табуляція та пошук початкових наближень
    roots_approx, trends = tabulate_and_find_roots()

    # Вибираємо два корені з різною поведінкою (Пункт 1)
    x_dec = None  # Спадає
    x_inc = None  # Зростає

    for r, t in zip(roots_approx, trends):
        if t == "Спадає" and x_dec is None: x_dec = r
        if t == "Зростає" and x_inc is None: x_inc = r

    print(f"Обрано корінь 1 (функція спадає): початкове x0 = {x_dec:.2f}")
    print(f"Обрано корінь 2 (функція зростає): початкове x0 = {x_inc:.2f}\n")

    # Динамічний підбір кроку релаксації tau згідно з умовою (стор. 3)
    # -2 < tau * F'(x) < 0  => tau має бути протилежного знаку до похідної
    tau_dec = 0.1 if dF_trans(x_dec) < 0 else -0.1
    tau_inc = 0.05 if dF_trans(x_inc) < 0 else -0.05

    # Таблиця порівняння методів
    print(
        f"{'Метод уточнення':<25} | {'Корінь 1 (Спадає)':<18} | {'Ітер. 1':<7} | {'Корінь 2 (Зростає)':<18} | {'Ітер. 2'}")
    print("-" * 95)

    methods = [
        ("Проста ітерація", lambda x: simple_iteration(x, tau_dec if x == x_dec else tau_inc)),
        ("Метод Ньютона", lambda x: newton_method(x)),
        ("Метод Чебишева", lambda x: chebyshev_method(x)),
        ("Метод хорд", lambda x: secant_method(x - 0.1, x + 0.1)),
        ("Метод парабол", lambda x: parabola_method(x - 0.2, x - 0.1, x)),
        ("Зворотна інтерполяція", lambda x: inverse_interpolation(x - 0.2, x - 0.1, x))
    ]

    final_roots_trans = []
    for name, func in methods:
        r1, it1 = func(x_dec)
        r2, it2 = func(x_inc)

        # Збережемо точні значення для графіків з методу Ньютона
        if name == "Метод Ньютона":
            final_roots_trans.extend([r1, r2])

        r1_s = f"{r1:.6f}" if r1 is not None else "Розбіжний"
        r2_s = f"{r2:.6f}" if r2 is not None else "Розбіжний"
        print(f"{name:<25} | {r1_s:<18} | {it1:<7} | {r2_s:<18} | {it2}")

    # 2. Алгебраїчна частина
    write_poly_coefs()
    poly_coefs = read_poly_coefs()

    print("\n" + "=" * 50)
    print("Алгебраїчне рівняння: x^3 - 2x^2 + x - 2 = 0")
    print("=" * 50)

    # Дійсний корінь
    real_root, iters_real = newton_horner(poly_coefs, 3.0)
    print(f"Дійсний корінь (Ньютон-Горнер): x = {real_root:.6f} (Ітерацій: {iters_real})")

    # Комплексні корені
    c_root1, c_root2, iters_c = lin_method(poly_coefs, 0.5, 0.5)
    print(f"Комплексні корені (Метод Ліна):  x1 = {c_root1.real:.4f} + {c_root1.imag:.4f}i")
    print(f"                                 x2 = {c_root2.real:.4f} + {c_root2.imag:.4f}i (Ітерацій: {iters_c})")

    # ==========================================
    # ВІЗУАЛІЗАЦІЯ (ГРАФІКИ)
    # ==========================================
    x_arr = np.linspace(-1.0, 4.0, 500)
    y_trans = F_trans(x_arr)
    y_poly = [horner_eval(poly_coefs, xv) for xv in x_arr]

    plt.figure(figsize=(12, 5))

    # Графік 1: Трансцендентна функція
    plt.subplot(1, 2, 1)
    plt.plot(x_arr, y_trans, 'b-', label=r'$F(x) = e^x - 3x^2$', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Нанесення точок знайдених коренів
    for r in final_roots_trans:
        if r is not None:
            plt.plot(r, F_trans(r), 'ro', markersize=8)
            plt.text(r, F_trans(r) + 2, f'x={r:.3f}', fontsize=10, ha='center', color='red')

    plt.title("Трансцендентне рівняння та його корені", fontsize=12)
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.legend()

    # Графік 2: Алгебраїчний многочлен
    plt.subplot(1, 2, 2)
    plt.plot(x_arr, y_poly, 'g-', label=r'$P(x) = x^3 - 2x^2 + x - 2$', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Нанесення дійсного кореня
    if real_root is not None:
        plt.plot(real_root, horner_eval(poly_coefs, real_root), 'go', markersize=8)
        plt.text(real_root, 5, f'x={real_root:.1f}', fontsize=10, ha='center', color='green')

    plt.title("Алгебраїчне рівняння (дійсний корінь)", fontsize=12)
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.ylim(-15, 25)
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()