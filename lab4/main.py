import math
import numpy as np
import matplotlib.pyplot as plt

def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

t0 = 1.0
exact_val = -5 * math.exp(-0.1 * t0) + 5 * math.cos(t0)
h_fixed = 10 ** -3

t_plot = np.linspace(0, 20, 200)
M_plot = [M(t) for t in t_plot]

h_powers = np.arange(3, -21, -1, dtype=float)
h_values = 10.0 ** h_powers
errors_h = []
for hv in h_values:
    if t0 + hv == t0:
        errors_h.append(None)
        continue
    y_p = (M(t0 + hv) - M(t0 - hv)) / (2 * hv)
    errors_h.append(abs(y_p - exact_val))

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

fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(t_plot, M_plot, color='tab:blue', linewidth=2)
ax1.set_title("1. Модель вологості ґрунту M(t)", fontsize=12)
ax1.set_xlabel("t")
ax1.set_ylabel("M(t)")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2, 2, 2)
valid_h = [hv for hv in h_values if t0 + hv != t0]
valid_err = [er for er in errors_h if er is not None]
ax2.loglog(valid_h, valid_err, marker='o', markersize=4, color='tab:red')
ax2.axvline(h_fixed, color='black', linestyle='--', label=f'Наш h={h_fixed}')
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
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2e}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

print(f"Точне значення: {exact_val:.6f}")
print(f"Похибка (Базова): {err_base:.2e}")
print(f"Похибка (Рунге-Ромберг): {err_RR:.2e}")
print(f"Похибка (Ейткен): {err_E:.2e}")