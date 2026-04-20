import math
import numpy as np
import matplotlib.pyplot as plt


def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)


t0 = 1.0
exact_val = -5 * math.exp(-0.1 * t0) + 5 * math.cos(t0)

h_powers = np.arange(3, -21, -1, dtype=float)
h_values = 10.0 ** h_powers
errors_h = []
valid_h = []
valid_err = []

for hv in h_values:
    if t0 + hv == t0:
        errors_h.append(None)
        continue
    y_p = (M(t0 + hv) - M(t0 - hv)) / (2 * hv)
    err = abs(y_p - exact_val)

    errors_h.append(err)
    valid_h.append(hv)
    valid_err.append(err)

min_err = min(valid_err)
best_h_idx = valid_err.index(min_err)
h_opt = valid_h[best_h_idx]

h_fixed = 10 ** -3

t_plot = np.linspace(0, 20, 200)
M_plot = [M(t) for t in t_plot]

y_h = (M(t0 + h_fixed) - M(t0 - h_fixed)) / (2 * h_fixed)
y_2h = (M(t0 + 2 * h_fixed) - M(t0 - 2 * h_fixed)) / (4 * h_fixed)
y_4h = (M(t0 + 4 * h_fixed) - M(t0 - 4 * h_fixed)) / (8 * h_fixed)

y_RR = y_h + (y_h - y_2h) / 3
err_RR = abs(y_RR - exact_val)

num = (y_2h ** 2) - (y_4h * y_h)
den = 2 * y_2h - (y_4h + y_h)
y_E = num / den
err_E = abs(y_E - exact_val)
err_base = abs(y_h - exact_val)

p_val = (1 / math.log(2)) * math.log(abs((y_4h - y_2h) / (y_2h - y_h)))

fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(t_plot, M_plot, color='tab:blue', linewidth=2)
ax1.set_title("1. Модель вологості ґрунту M(t)", fontsize=12)
ax1.set_xlabel("t")
ax1.set_ylabel("M(t)")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2, 2, 2)
ax2.loglog(valid_h, valid_err, marker='o', markersize=4, color='tab:red')
ax2.axvline(h_fixed, color='black', linestyle='--', label=f'Заданий h={h_fixed}')
ax2.axvline(h_opt, color='green', linestyle='-', label=f'Оптимальний h={h_opt:.1e}')
ax2.set_title("2. Вплив кроку h на похибку (Log-Log)", fontsize=12)
ax2.set_xlabel("Крок h")
ax2.set_ylabel("Абсолютна похибка")
ax2.legend()
ax2.grid(True, which="both", alpha=0.2)

ax3 = fig.add_subplot(2, 1, 2)
methods = ['Базовий крок h', 'Рунге-Ромберг', 'Ейткен']
errors = [err_base, err_RR, err_E]
colors = ['tab:gray', 'tab:orange', 'tab:green']

bars = ax3.bar(methods, errors, color=colors)
ax3.set_yscale('log')
ax3.set_title("3. Порівняння похибок методів уточнення (Log scale)", fontsize=12)
ax3.set_ylabel("Похибка (менше = краще)")

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.2e}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

print("--- РЕЗУЛЬТАТИ ОБЧИСЛЕНЬ ---")
print(f"Точне значення: {exact_val:.6f}")
print(f"Автоматично знайдений оптимальний крок h: {h_opt:.1e} (похибка: {min_err:.2e})")
print(f"Похибка (Базова для h=1e-3): {err_base:.2e}")
print(f"Похибка (Рунге-Ромберг): {err_RR:.2e}")
print(f"Похибка (Ейткен): {err_E:.2e}")
print(f"Оцінка порядку точності p (Ейткен): {p_val:.4f}\n")

print("--- АНАЛІЗ ПРОЦЕСУ ПОЛИВУ (Висновок) ---")
print("Згідно з рівнянням M(t) = 50*e^(-0.1*t) + 5*sin(t), швидкість висихання ґрунту")
print("визначається величиною -dM/dt = 5*e^(-0.1*t) - 5*cos(t).")
print("Швидкість висихання є найбільшою в ті моменти, коли похідна dM/dt набуває")
print("найменших (найбільш від'ємних) значень. Це відбувається періодично, коли cos(t) ≈ -1,")
print("тобто при t ≈ π, 3π, 5π... (приблизно в моменти t = 3.14, 9.42, 15.71).")
print("ВИСНОВОК: Автоматичний полив найдоцільніше вмикати саме в ці періоди максимальної")
print("швидкості втрати вологи, або безпосередньо перед ними, щоб не допустити пересихання.")