import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# МЕТОД ХУКА-ДЖИВСА 
# ───────────────────────────────────────────────────────────── 

def hooke_jeeves(f, x0, step=0.5, alpha=2.0, eps1=1e-6, eps2=1e-6, max_iter=10000):
    """ 
    Метод Хука-Дживса для знаходження мінімуму функції f. 

    Параметри: 
        f       - цільова функція 
        x0      - початкове наближення (список або масив) 
        step    - початковий крок (однаковий для всіх змінних) 
        alpha   - коефіцієнт збільшення кроку при пошуку по зразку 
        eps1    - критерій зупинки по значенню функції 
        eps2    - критерій зупинки по аргументу (кроку) 
        max_iter- максимальна кількість ітерацій 

    Повертає: 
        x_opt   - знайдена точка мінімуму 
        f_opt   - значення функції в точці мінімуму 
        trajectory - список точок траєкторії спуску 
        steps   - кількість кроків 
    """
    n = len(x0)
    x_base = np.array(x0, dtype=float)  # базисна точка
    h = np.full(n, step, dtype=float)  # кроки по кожній змінній
    trajectory = [x_base.copy()]
    iterations = 0

    def exploratory_search(x_start, h_arr):
        """Досліджуючий пошук навколо точки x_start."""
        x = x_start.copy()
        for i in range(n):
            x_new = x.copy()
            x_new[i] += h_arr[i]
            if f(x_new) < f(x):
                x = x_new
            else:
                x_new[i] = x[i] - h_arr[i]
                if f(x_new) < f(x):
                    x = x_new
        return x

    while iterations < max_iter:
        # ── Досліджуючий пошук з базисної точки ── 
        x_new = exploratory_search(x_base, h)

        if np.all(x_new == x_base):
            # Досліджуючий пошук не покращив — зменшуємо крок 
            h /= 2.0
            if np.all(h < eps2):
                break
            continue

            # ── Перевірка умов зупинки ──
        if (abs(f(x_new) - f(x_base)) < eps1 and
                np.linalg.norm(x_new - x_base) < eps2):
            x_base = x_new
            trajectory.append(x_base.copy())
            break

            # ── Пошук по зразку ──
        x_pattern = x_new + alpha * (x_new - x_base)
        x_base_old = x_base.copy()
        x_base = x_new.copy()
        trajectory.append(x_base.copy())

        # Досліджуючий пошук з точки зразку 
        x_trial = exploratory_search(x_pattern, h)

        if f(x_trial) < f(x_base):
            x_base = x_trial.copy()
            trajectory.append(x_base.copy())

        iterations += 1

    return x_base, f(x_base), trajectory, iterations


# ───────────────────────────────────────────────────────────── 
# ЦІЛЬОВІ ФУНКЦІЇ 
# ───────────────────────────────────────────────────────────── 

def rosenbrock(x):
    """Функція Розенброка: мінімум у (1,1), f=0"""
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def power_func(x):
    """Степенева функція"""
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def root_func(x):
    """Коренева функція"""
    return (x[0] - 2) ** 4 + (x[0] - 2 * x[1]) ** 2


def wood(x):
    """Функція Вуда: 4 змінні, мінімум у (1,1,1,1), f=0"""
    return (100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 +
            90 * (x[3] - x[2] ** 2) ** 2 + (1 - x[2]) ** 2 +
            10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) +
            19.8 * (x[1] - 1) * (x[3] - 1))


def powell(x):
    """Функція Пауелла: 4 змінні, мінімум у (0,0,0,0), f=0"""
    return ((x[0] + 10 * x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2 +
            (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4)


def miele(x):
    """Функція Мієлє"""
    return ((np.exp(x[0]) - x[1]) ** 4 + 100 * (x[1] - x[2]) ** 6 +
            np.tan(x[2] - x[3]) ** 4 + x[0] ** 8)


# ───────────────────────────────────────────────────────────── 
# СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ (ЗАВДАННЯ 1 і 4) 
# ───────────────────────────────────────────────────────────── 
# Система:  x^2 + y^2 = 4 
#           x^2 - y   = 0 
# Розв'язки: (√2, 2) та (-√2, 2) 

def eq1(x, y): return x ** 2 + y ** 2 - 4


def eq2(x, y): return x ** 2 - y


def target_system(xy):
    """Цільова функція для системи: F(x,y) = f1^2 + f2^2"""
    x, y = xy
    return eq1(x, y) ** 2 + eq2(x, y) ** 2


# ───────────────────────────────────────────────────────────── 
# ЗАВДАННЯ 1: Графіки рівнянь системи 
# ───────────────────────────────────────────────────────────── 

def task1_plot_system():
    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 1: Побудова графіків системи нелінійних рівнянь")
    print("=" * 60)
    print("Система рівнянь:")
    print("  f1: x² + y² = 4  (коло радіусом 2)")
    print("  f2: x² - y  = 0  (парабола y = x²)")

    x = np.linspace(-2.5, 2.5, 500)

    # Рівняння 1: коло x² + y² = 4 
    theta = np.linspace(0, 2 * np.pi, 500)
    circle_x = 2 * np.cos(theta)
    circle_y = 2 * np.sin(theta)

    # Рівняння 2: парабола y = x² 
    parabola_y = x ** 2

    # Аналітичні розв'язки 
    # x² + x⁴ = 4 => x² ≈ 1.5616... => x ≈ ±1.2496 
    x_sol = np.sqrt((-1 + np.sqrt(17)) / 2)
    y_sol = x_sol ** 2
    solutions = [(x_sol, y_sol), (-x_sol, y_sol)]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor('#0f1117')
    fig.patch.set_facecolor('#0f1117')

    ax.plot(circle_x, circle_y, color='#4fc3f7', lw=2.5, label=r'$x^2 + y^2 = 4$')
    ax.plot(x, parabola_y, color='#f48fb1', lw=2.5, label=r'$y = x^2$')

    for sx, sy in solutions:
        ax.scatter(sx, sy, color='#ffd54f', s=120, zorder=5)
        ax.annotate(f'({sx:.4f}, {sy:.4f})',
                    xy=(sx, sy), xytext=(sx + 0.15, sy + 0.15),
                    color='#ffd54f', fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='#ffd54f', lw=1.2))
        print(f"  Аналітичний розв'язок: x = {sx:.6f}, y = {sy:.6f}")

    ax.axhline(0, color='#555', lw=0.8)
    ax.axvline(0, color='#555', lw=0.8)
    ax.set_xlim(-3, 3);
    ax.set_ylim(-2.5, 4.5)
    ax.set_title('Завдання 1 — Графіки рівнянь системи', color='white', fontsize=14, pad=12)
    ax.set_xlabel('x', color='#aaa');
    ax.set_ylabel('y', color='#aaa')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values(): spine.set_edgecolor('#444')
    ax.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=11)
    ax.grid(color='#2a2d3e', lw=0.6)
    plt.tight_layout()
    plt.savefig('task1_system_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Графік збережено: task1_system_plot.png")


# ───────────────────────────────────────────────────────────── 
# ЗАВДАННЯ 3: Тестування на цільових функціях 
# ───────────────────────────────────────────────────────────── 

def task3_test_functions():
    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 3: Тестування методу на цільових функціях")
    print("=" * 60)

    tests = [
        ("Розенброка", rosenbrock, [-1.2, 1.0], 2),
        ("Степенева", power_func, [0.0, 0.0], 2),
        ("Коренева", root_func, [0.0, 3.0], 2),
        ("Вуда", wood, [-3, -1, -3, -1], 4),
        ("Пауелла", powell, [3, -1, 0, 1], 4),
        ("Мієлє", miele, [0, 1, 1, 1], 4),
    ]

    results = []
    for name, func, x0, dim in tests:
        x_opt, f_opt, traj, steps = hooke_jeeves(
            func, x0, step=0.5, alpha=2.0, eps1=1e-14, eps2=1e-14, max_iter=50000)
        print(f"\n  [{name}]")
        print(f"    Початкова точка : {x0}")
        print(f"    Знайдена точка  : {np.round(x_opt, 6).tolist()}")
        print(f"    f(x_opt)        = {f_opt:.2e}")
        print(f"    Кількість кроків: {steps}")
        results.append((name, func, x0, x_opt, f_opt, traj, dim))

        # ── Графіки для 2D функцій ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle('Завдання 3 — Траєкторії спуску (2D функції)',
                 color='white', fontsize=14, y=1.01)

    funcs_2d = [(n, f, x0, xo, fo, tr) for n, f, x0, xo, fo, tr, d in results if d == 2]

    for ax, (name, func, x0, x_opt, f_opt, traj, *_) in zip(axes, funcs_2d):
        ax.set_facecolor('#0f1117')
        traj = np.array(traj)

        # Рівні функції 
        margin = max(abs(traj[:, 0].max() - traj[:, 0].min()), 1.0) * 1.4
        cx = np.linspace(x_opt[0] - margin, x_opt[0] + margin, 200)
        cy = np.linspace(x_opt[1] - margin, x_opt[1] + margin, 200)
        CX, CY = np.meshgrid(cx, cy)
        CZ = np.array([[func([cx[j], cy[i]]) for j in range(len(cx))] for i in range(len(cy))])

        ax.contourf(CX, CY, CZ, levels=30, cmap='plasma', alpha=0.6)
        ax.contour(CX, CY, CZ, levels=15, colors='white', alpha=0.2, linewidths=0.5)

        ax.plot(traj[:, 0], traj[:, 1], 'o-', color='#4fc3f7',
                markersize=3, lw=1.4, label='Траєкторія')
        ax.scatter(*x0, color='#ffd54f', s=100, zorder=6, label='Старт')
        ax.scatter(*x_opt, color='#69f0ae', s=100, zorder=6, label='Мінімум')

        ax.set_title(f'{name}\nf = {f_opt:.2e}', color='white', fontsize=10)
        ax.set_xlabel('x₁', color='#aaa');
        ax.set_ylabel('x₂', color='#aaa')
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values(): spine.set_edgecolor('#444')
        ax.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=8)
        ax.grid(color='#2a2d3e', lw=0.5)

    plt.tight_layout()
    plt.savefig('task3_test_functions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nГрафік збережено: task3_test_functions.png")
    return results


# ───────────────────────────────────────────────────────────── 
# ЗАВДАННЯ 4 і 5: Розв'язок системи + траєкторія + файл 
# ───────────────────────────────────────────────────────────── 

def task4_5_solve_system():
    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 4–5: Розв'язок системи нелінійних рівнянь")
    print("=" * 60)
    print("Цільова функція: F(x,y) = (x²+y²-4)² + (x²-y)²")
    print()

    start_points = [
        ([1.0, 1.5], "Початкове наближення №1 (→ правий розв'язок)"),
        ([-1.0, 1.5], "Початкове наближення №2 (→ лівий розв'язок)"),
    ]

    all_trajectories = []
    all_solutions = []

    for x0, label in start_points:
        x_opt, f_opt, traj, steps = hooke_jeeves(
            target_system, x0, step=0.3, alpha=2.0,
            eps1=1e-10, eps2=1e-10, max_iter=100000)
        print(f"  {label}")
        print(f"    x0 = {x0}")
        print(f"    Розв'язок: x = {x_opt[0]:.8f}, y = {x_opt[1]:.8f}")
        print(f"    F(x,y)   = {f_opt:.2e}")
        print(f"    Кроків   : {steps}")
        print(f"    Перевірка: f1 = {eq1(*x_opt):.2e}, f2 = {eq2(*x_opt):.2e}")
        print()
        all_trajectories.append((traj, label, x0))
        all_solutions.append(x_opt)

        # ── Збереження траєкторії у файл ── 
        fname = f"trajectory_{'right' if x0[0] > 0 else 'left'}.txt"
        with open(fname, 'w', encoding='utf-8') as fp:
            fp.write(f"Траєкторія спуску — {label}\n")
            fp.write(f"Початкова точка: {x0}\n")
            fp.write(f"Знайдений розв'язок: x = {x_opt[0]:.8f}, y = {x_opt[1]:.8f}\n")
            fp.write(f"F(розв'язок) = {f_opt:.2e}\n")
            fp.write(f"Кількість кроків: {steps}\n\n")
            fp.write(f"{'Крок':>6}  {'x':>14}  {'y':>14}  {'F(x,y)':>16}\n")
            fp.write("-" * 58 + "\n")
            for i, pt in enumerate(traj):
                fp.write(f"{i:>6}  {pt[0]:>14.8f}  {pt[1]:>14.8f}  {target_system(pt):>16.6e}\n")
        print(f"  Траєкторію збережено у файл: {fname}")

        # ── Побудова 3 графіків для задачі ──
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0f1117')
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    colors_traj = ['#4fc3f7', '#f48fb1']
    colors_start = ['#ffd54f', '#ffb300']
    colors_end = ['#69f0ae', '#00e676']

    # ── Графік 1: Рівні цільової функції + траєкторії ── 
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#0f1117')

    xg = np.linspace(-2.5, 2.5, 300)
    yg = np.linspace(-0.5, 4.5, 300)
    XG, YG = np.meshgrid(xg, yg)
    ZG = eq1(XG, YG) ** 2 + eq2(XG, YG) ** 2

    cf = ax1.contourf(XG, YG, ZG, levels=40, cmap='inferno', alpha=0.7)
    ax1.contour(XG, YG, ZG, levels=20, colors='white', alpha=0.25, linewidths=0.6)
    plt.colorbar(cf, ax=ax1, label='F(x,y)', shrink=0.8)

    # Рівняння системи 
    theta = np.linspace(0, 2 * np.pi, 500)
    ax1.plot(2 * np.cos(theta), 2 * np.sin(theta), color='#4fc3f7',
             lw=2, linestyle='--', label='x²+y²=4', zorder=4)
    x_line = np.linspace(-2.2, 2.2, 300)
    ax1.plot(x_line, x_line ** 2, color='#f48fb1',
             lw=2, linestyle='--', label='y=x²', zorder=4)

    for (traj, label, x0), c_t, c_s, c_e in zip(
            all_trajectories, colors_traj, colors_start, colors_end):
        traj_arr = np.array(traj)
        ax1.plot(traj_arr[:, 0], traj_arr[:, 1], 'o-',
                 color=c_t, markersize=4, lw=1.5, alpha=0.9,
                 label=f'Траєкторія {x0}')
        ax1.scatter(*x0, color=c_s, s=120, zorder=8, marker='D')
        ax1.scatter(*traj_arr[-1], color=c_e, s=120, zorder=8, marker='*')

    ax1.set_xlim(-2.5, 2.5);
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_title('Рівні цільової функції F(x,y) та траєкторії спуску',
                  color='white', fontsize=13)
    ax1.set_xlabel('x', color='#aaa');
    ax1.set_ylabel('y', color='#aaa')
    ax1.tick_params(colors='#aaa')
    for spine in ax1.spines.values(): spine.set_edgecolor('#444')
    ax1.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white',
               fontsize=9, loc='upper right')
    ax1.grid(color='#2a2d3e', lw=0.5)

    # ── Графік 2: F(x,y) vs крок (перша траєкторія) ── 
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#0f1117')
    for (traj, label, x0), c_t in zip(all_trajectories, colors_traj):
        f_vals = [target_system(pt) for pt in traj]
        ax2.semilogy(f_vals, color=c_t, lw=1.8, label=f'x0={x0}')
    ax2.set_title('Збіжність F(x,y) по кроках', color='white', fontsize=11)
    ax2.set_xlabel('Крок', color='#aaa');
    ax2.set_ylabel('F(x,y)', color='#aaa')
    ax2.tick_params(colors='#aaa')
    for spine in ax2.spines.values(): spine.set_edgecolor('#444')
    ax2.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=9)
    ax2.grid(color='#2a2d3e', lw=0.5, which='both')

    # ── Графік 3: 3D поверхня F(x,y) ── 
    ax3 = fig.add_subplot(gs[1, 1], projection='3d')
    ax3.set_facecolor('#0f1117')
    xg3 = np.linspace(-2.5, 2.5, 80)
    yg3 = np.linspace(-0.5, 4.5, 80)
    XG3, YG3 = np.meshgrid(xg3, yg3)
    ZG3 = eq1(XG3, YG3) ** 2 + eq2(XG3, YG3) ** 2
    surf = ax3.plot_surface(XG3, YG3, ZG3, cmap='plasma',
                            alpha=0.85, linewidth=0, antialiased=True)
    for sol in all_solutions:
        ax3.scatter(sol[0], sol[1], target_system(sol),
                    color='#69f0ae', s=80, zorder=10)
    ax3.set_title('3D поверхня F(x,y)', color='white', fontsize=11, pad=8)
    ax3.set_xlabel('x', color='#aaa', labelpad=5)
    ax3.set_ylabel('y', color='#aaa', labelpad=5)
    ax3.set_zlabel('F', color='#aaa', labelpad=5)
    ax3.tick_params(colors='#aaa')
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    fig.colorbar(surf, ax=ax3, shrink=0.5, pad=0.1)

    fig.suptitle('Завдання 4–5: Метод Хука-Дживса — Розв\'язок системи нелінійних рівнянь',
                 color='white', fontsize=14, y=1.01)

    plt.savefig('task4_5_system_solution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Графік збережено: task4_5_system_solution.png")


# ───────────────────────────────────────────────────────────── 
# ДОДАТКОВИЙ БЛОК: порівняння функцій при різних параметрах 
# ───────────────────────────────────────────────────────────── 

def extra_comparison():
    print("\n" + "=" * 60)
    print("ДОДАТКОВО: Аналіз впливу параметрів методу на збіжність")
    print("=" * 60)

    alphas = [1.5, 2.0, 3.0]
    steps_init = [0.1, 0.5, 1.0]
    x0 = [-1.2, 1.0]  # Функція Розенброка 

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f1117')
    colors = ['#4fc3f7', '#f48fb1', '#69f0ae']

    # ── Вплив alpha ── 
    ax = axes[0]
    ax.set_facecolor('#0f1117')
    for a, c in zip(alphas, colors):
        _, _, traj, iters = hooke_jeeves(
            rosenbrock, x0, step=0.5, alpha=a, eps1=1e-8, eps2=1e-8, max_iter=50000)
        f_vals = [rosenbrock(pt) for pt in traj]
        ax.semilogy(f_vals, color=c, lw=1.8, label=f'α={a} ({iters} ітер.)')
        print(f"  alpha={a}: {iters} кроків")
    ax.set_title('Вплив α (крок збільшення) — Функція Розенброка',
                 color='white', fontsize=11)
    ax.set_xlabel('Крок', color='#aaa');
    ax.set_ylabel('f(x)', color='#aaa')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values(): spine.set_edgecolor('#444')
    ax.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=9)
    ax.grid(color='#2a2d3e', lw=0.5, which='both')

    # ── Вплив початкового кроку ── 
    ax = axes[1]
    ax.set_facecolor('#0f1117')
    for s, c in zip(steps_init, colors):
        _, _, traj, iters = hooke_jeeves(
            rosenbrock, x0, step=s, alpha=2.0, eps1=1e-8, eps2=1e-8, max_iter=50000)
        f_vals = [rosenbrock(pt) for pt in traj]
        ax.semilogy(f_vals, color=c, lw=1.8, label=f'h₀={s} ({iters} ітер.)')
        print(f"  step={s}: {iters} кроків")
    ax.set_title('Вплив початкового кроку h₀ — Функція Розенброка',
                 color='white', fontsize=11)
    ax.set_xlabel('Крок', color='#aaa');
    ax.set_ylabel('f(x)', color='#aaa')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values(): spine.set_edgecolor('#444')
    ax.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=9)
    ax.grid(color='#2a2d3e', lw=0.5, which='both')

    plt.suptitle('Додатково: Вплив параметрів методу Хука-Дживса',
                 color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('extra_parameter_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Графік збережено: extra_parameter_analysis.png")


# ───────────────────────────────────────────────────────────── 
# ГОЛОВНА ФУНКЦІЯ 
# ───────────────────────────────────────────────────────────── 

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Лабораторна робота №9 — Метод Хука-Дживса            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Завдання 1: графіки системи 
    task1_plot_system()

    # Завдання 3: тестування на цільових функціях 
    task3_test_functions()

    # Завдання 4–5: розв'язок системи + траєкторія + файли 
    task4_5_solve_system()

    # Додатково: аналіз параметрів 
    extra_comparison()

    print("\n" + "=" * 60)
    print("Всі завдання виконано успішно!")
    print("Файли збережено: task1_system_plot.png,")
    print("                 task3_test_functions.png,")
    print("                 task4_5_system_solution.png,")
    print("                 extra_parameter_analysis.png,")
    print("                 trajectory_right.txt,")
    print("                 trajectory_left.txt")
    print("=" * 60)

