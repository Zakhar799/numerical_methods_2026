import math
import matplotlib.pyplot as plt

def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

t_values = []
M_values = []
t_current = 0.0

while t_current <= 20:
    t_values.append(t_current)
    M_values.append(M(t_current))
    t_current += 0.1

plt.figure(figsize=(8, 5))
plt.plot(t_values, M_values)
plt.title("Soil Moisture Model M(t)")
plt.xlabel("t")
plt.ylabel("M(t)")
plt.grid(True)
plt.show()

print("-" * 40)

t0 = 1.0

exact_val = -5 * math.exp(-0.1 * t0) + 5 * math.cos(t0)
print(f"1. Точне значення похідної y'(x0): {exact_val:.3f}")

best_h = 1.0
min_error = float('inf')

for power in range(3, -21, -1):
    h_test = 10 ** power

    if t0 + h_test == t0:
        continue

    y_prime_h_test = (M(t0 + h_test) - M(t0 - h_test)) / (2 * h_test)
    error = abs(y_prime_h_test - exact_val)

    if error < min_error:
        min_error = error
        best_h = h_test

print(f"2. Найкраща точність досягається при кроці h0: {best_h:.1e}")
print(f"   Досягнута точність R0: {min_error:.3e}")

h = 10 ** -3
print(f"\n3. Приймаємо фіксований крок h = {h}")

y_h = (M(t0 + h) - M(t0 - h)) / (2 * h)
y_2h = (M(t0 + 2 * h) - M(t0 - 2 * h)) / (4 * h)  # 4h, бо 2 * (2h)
print(f"4. Значення похідної з кроком h:  {y_h:.3f}")
print(f"   Значення похідної з кроком 2h: {y_2h:.3f}")

R1 = abs(y_h - exact_val)
print(f"5. Похибка при кроці h (R1): {R1:.3e}")

y_RR = y_h + (y_h - y_2h) / 3
R2 = abs(y_RR - exact_val)

print(f"\n6. Уточнене значення Рунге-Ромберга: {y_RR:.3f}")
print(f"   Похибка (R2): {R2:.3e}")
if R2 < R1:
    print("   -> Характер зміни похибки: похибка зменшилась.")

y_4h = (M(t0 + 4 * h) - M(t0 - 4 * h)) / (8 * h)

numerator = (y_2h ** 2) - (y_4h * y_h)
denominator = 2 * y_2h - (y_4h + y_h)
y_E = numerator / denominator

p = math.log(abs((y_4h - y_2h) / (y_2h - y_h))) / math.log(2)
R3 = abs(y_E - exact_val)

print(f"\n7. Уточнене значення за методом Ейткена: {y_E:.3f}")
print(f"   Порядок точності формули (p): {p:.3f}")
print(f"   Похибка (R3): {R3:.3e}")
if R3 < R1:
    print("   -> Характер зміни похибки: похибка значно зменшилась.")
S
print("\n--- Висновок щодо поливу ---")
if y_E < 0:
    print(f"Швидкість зміни вологості від'ємна ({y_E:.3f}).")
    print("Грунт висихає. Якщо абсолютне значення швидкості велике, потрібен інтенсивний режим поливу.")
else:
    print(f"Швидкість зміни вологості додатня ({y_E:.3f}).")
    print("Вологість зростає або стабільна. Полив поки що не потрібен.")