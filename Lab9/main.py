import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# 1. РЕАЛІЗАЦІЯ МЕТОДУ ХУКА-ДЖИВЗА
# =====================================================================

def hooke_jeeves(func, X0, delta, eps1, eps2, q=2.0, p=2.0, max_iter=1000):
    """
    Оптимізація функції методом Хука-Дживса.
    Повертає: точку мінімуму, кількість кроків та історію траєкторії.
    """
    X0 = np.array(X0, dtype=float)
    delta = np.array(delta, dtype=float)
    n = len(X0)

    X_base = np.copy(X0)  # X^(0)
    trajectory = [np.copy(X_base)]

    steps_count = 0

    for iteration in range(max_iter):
        steps_count += 1

        # --- Досліджуючий пошук ---
        X_curr = np.copy(X_base)  # X^(1) = X^(0)

        for i in range(n):
            # Крок у позитивному напрямку
            X_try = np.copy(X_curr)
            X_try[i] += delta[i]

            if func(X_try) < func(X_curr):
                X_curr = X_try
            else:
                # Крок у негативному напрямку
                X_try = np.copy(X_curr)
                X_try[i] -= delta[i]
                if func(X_try) < func(X_curr):
                    X_curr = X_try
                else:
                    # Зменшення кроку, якщо покращення немає
                    delta[i] /= q
                    # Якщо крок став занадто малим, повертаємо координату назад
                    if delta[i] < eps1:
                        X_curr[i] = X_base[i]

        # Перевірка умов завершення
        if np.array_equal(X_curr, X_base):
            # Якщо X^(1) == X^(0), ми в мінімумі
            return X_base, steps_count, np.array(trajectory)

        # Перевірка критеріїв закінчення за точністю
        norm_delta = np.linalg.norm(delta)
        f_diff = abs(func(X_curr) - func(X_base))

        if norm_delta < eps1 and f_diff < eps2:
            trajectory.append(np.copy(X_curr))
            return X_curr, steps_count, np.array(trajectory)

        # --- Пошук по зразку ---
        # Напрям задається як: X_pattern = X_curr + p * (X_curr - X_base)
        X_pattern = X_curr + p * (X_curr - X_base)

        # Досліджуючий пошук з точки X_pattern (без зменшення кроку за алгоритмом)
        X_curr_pattern = np.copy(X_pattern)
        for i in range(n):
            X_try = np.copy(X_curr_pattern)
            X_try[i] += delta[i]
            if func(X_try) < func(X_curr_pattern):
                X_curr_pattern = X_try
            else:
                X_try = np.copy(X_curr_pattern)
                X_try[i] -= delta[i]
                if func(X_try) < func(X_curr_pattern):
                    X_curr_pattern = X_try

        # Оцінка результату пошуку по зразку
        if func(X_curr_pattern) < func(X_curr):
            X_base = np.copy(X_curr)
            X_curr = np.copy(X_curr_pattern)
        else:
            X_base = np.copy(X_curr)

        trajectory.append(np.copy(X_base))

    return X_base, steps_count, np.array(trajectory)


# =====================================================================
# 2. ВИЗНАЧЕННЯ ФУНКЦІЙ ТА СИСТЕМИ РІВНЯНЬ
# =====================================================================

# Тестова функція 1: Функція Розенброка
def rosenbrock(X):
    return 100 * (X[0] ** 2 - X[1]) ** 2 + (X[0] - 1) ** 2


# Система нелінійних рівнянь (Пункт 1 ходу роботи)
# f1(x1, x2) = x1^2 + x2^2 - 4 = 0  (Коло радіуса 2)
# f2(x1, x2) = x2 - exp(x1) + 1 = 0 (Експонента)
def f1(x1, x2):
    return x1 ** 2 + x2 ** 2 - 4


def f2(x1, x2):
    return x2 - np.exp(x1) + 1


# Цільова функція системи рівнянь (Побудована за п.4: сума квадратів)
def system_objective(X):
    return f1(X[0], X[1]) ** 2 + f2(X[0], X[1]) ** 2


# =====================================================================
# MAIN ЕТАП ВИКОНАННЯ
# =====================================================================
if __name__ == "__main__":
    print("--- Лабораторна робота №9 ---")

    # -----------------------------------------------------------------
    # ГРАФІК 1: Візуалізація системи рівнянь для вибору початкової точки
    # -----------------------------------------------------------------
    x1_vals = np.linspace(-2.5, 2.5, 400)
    x2_vals = np.linspace(-2.5, 2.5, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    plt.figure(figsize=(8, 6))
    # ВИПРАВЛЕНО: Видалено некоректний аргумент labels
    plt.contour(X1, X2, f1(X1, X2), levels=[0], colors='blue')
    plt.contour(X1, X2, f2(X1, X2), levels=[0], colors='red')

    # Додаємо легенду через пусті лінії
    plt.plot([], [], color='blue', label='$x_1^2 + x_2^2 - 4 = 0$')
    plt.plot([], [], color='red', label='$x_2 - e^{x_1} + 1 = 0$')

    plt.title("Графік системи нелінійних рівнянь (знаходження точок перетину)")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Встановлюємо початкове наближення
    X0_sys = [1.0, 1.5]
    delta_sys = [0.1, 0.1]
    eps1, eps2 = 1e-5, 1e-5

    # -----------------------------------------------------------------
    # ПУНКТ 3: Тестування програми на функції Розенброка
    # -----------------------------------------------------------------
    print("\n[Тестування програми] Мінімізація функції Розенброка:")
    X0_rosen = [-1.2, 0.0]
    delta_rosen = [0.1, 0.1]

    best_rosen, steps_rosen, traj_rosen = hooke_jeeves(
        rosenbrock, X0_rosen, delta_rosen, eps1, eps2
    )
    print(f"Початкова точка: {X0_rosen}")
    print(f"Знайдений мінімум: {best_rosen}")
    print(f"Значення функції в мінімумі: {rosenbrock(best_rosen):.8f}")
    print(f"Кількість кроків: {steps_rosen}")

    # -----------------------------------------------------------------
    # ПУНКТ 4, 5: Розв'язання системи нелінійних рівнянь
    # -----------------------------------------------------------------
    print("\n[Розв'язання системи] Мінімізація цільової функції системи:")
    best_sys, steps_sys, traj_sys = hooke_jeeves(
        system_objective, X0_sys, delta_sys, eps1, eps2
    )
    print(f"Початкова точка: {X0_sys}")
    print(f"Знайдений розв'язок системи: {best_sys}")
    print(f"Значення цільової функції (має бути ~0): {system_objective(best_sys):.8e}")
    print(f"Кількість кроків на траєкторії спуску: {steps_sys}")

    # Запис траєкторії спуску у файл
    filename = "trajectory.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Крок\tX1\t\tX2\t\tЗначення функції\n")
        for idx, pt in enumerate(traj_sys):
            f.write(f"{idx}\t{pt[0]:.6f}\t{pt[1]:.6f}\t{system_objective(pt):.6e}\n")
    print(f"\nКоординати точок траєкторії успішно збережено у файл '{filename}'.")

    # -----------------------------------------------------------------
    # ГРАФІК 2: Візуалізація траєкторії спуску
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 6))

    Z = system_objective([X1, X2])
    contours = plt.contour(X1, X2, Z, levels=30, cmap='viridis')

    # ВИПРАВЛЕНО: Додано префікс r перед рядком із LaTeX розміткою
    plt.colorbar(contours, label=r'Значення цільової функції $\Phi(X)$')

    plt.plot(traj_sys[:, 0], traj_sys[:, 1], 'ro-', linewidth=1.5, markersize=4, label='Траєкторія спуску')
    plt.plot(X0_sys[0], X0_sys[1], 'go', markersize=8, label='Старт')
    plt.plot(best_sys[0], best_sys[1], 'b*', markersize=12, label='Знайдений мінімум')

    plt.title("Траєкторія оптимізації методом Хука-Дживса")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid(True)
    plt.show()