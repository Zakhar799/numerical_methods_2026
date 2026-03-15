import numpy as np
import matplotlib.pyplot as plt
import csv


# -------------------------------------------------
# 1. Зчитування даних з CSV
# -------------------------------------------------

def read_csv(filename):
    months = []
    temps = []

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            months.append(int(row['Month']))
            temps.append(float(row['Temp']))

    return np.array(months), np.array(temps)


# -------------------------------------------------
# 2. Функції МНК
# -------------------------------------------------

def form_matrix(x, m):
    """Формування матриці нормальних рівнянь"""
    n = m + 1
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i, j] = np.sum(x ** (i + j))

    return A


def form_vector(x, y, m):
    """Формування вектора правої частини"""
    n = m + 1
    b = np.zeros(n)

    for i in range(n):
        b[i] = np.sum(y * (x ** i))

    return b


# -------------------------------------------------
# 3. Метод Гауса
# -------------------------------------------------

def gauss_solve(A, b):
    """Розв’язок СЛАР методом Гауса з вибором головного елемента"""
    n = len(b)

    A = A.astype(float)
    b = b.astype(float)

    # Прямий хід
    for k in range(n):

        max_row = np.argmax(np.abs(A[k:, k])) + k

        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]

            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотній хід
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


# -------------------------------------------------
# 4. Обчислення полінома
# -------------------------------------------------

def polynomial(x, coef):
    y = np.zeros_like(x, dtype=float)

    for i, c in enumerate(coef):
        y += c * (x ** i)

    return y


# -------------------------------------------------
# 5. Обчислення дисперсії
# -------------------------------------------------

def variance(y_true, y_approx):
    return np.mean((y_true - y_approx) ** 2)


# -------------------------------------------------
# 6. Основна програма
# -------------------------------------------------

# Зчитування даних
months, temps = read_csv("temperature.csv")

max_degree = 10
variances = []

print("Ступінь | Дисперсія")
print("---------------------")

# Обчислення дисперсій
for m in range(1, max_degree + 1):

    A = form_matrix(months, m)
    b = form_vector(months, temps, m)

    coef = gauss_solve(A, b)

    y_approx = polynomial(months, coef)

    var = variance(temps, y_approx)

    variances.append(var)

    print(f"{m:6d} | {var:.4f}")


# -------------------------------------------------
# 7. Вибір оптимального степеня
# -------------------------------------------------

optimal_m = np.argmin(variances) + 1

print("\nОптимальний степінь полінома:", optimal_m)


# -------------------------------------------------
# 8. Побудова фінальної апроксимації
# -------------------------------------------------

A = form_matrix(months, optimal_m)
b = form_vector(months, temps, optimal_m)

coef = gauss_solve(A, b)

y_approx = polynomial(months, coef)


# -------------------------------------------------
# 9. Прогноз на 3 місяці
# -------------------------------------------------

future_months = np.array([25, 26, 27])

future_temp = polynomial(future_months, coef)

print("\nПрогноз температур:")
for m, t in zip(future_months, future_temp):
    print(f"Місяць {m}: {t:.2f}")


# -------------------------------------------------
# 10. Похибка
# -------------------------------------------------

error = temps - y_approx


# -------------------------------------------------
# 11. Графіки
# -------------------------------------------------

plt.figure(figsize=(12, 10))

# Графік апроксимації
plt.subplot(3, 1, 1)

plt.scatter(months, temps, label="Фактичні дані")
plt.plot(months, y_approx, label=f"Апроксимація (m={optimal_m})")

plt.title("Температура та апроксимація")
plt.legend()
plt.grid(True)


# Графік похибки
plt.subplot(3, 1, 2)

plt.bar(months, np.abs(error))

plt.title("Похибка апроксимації")
plt.grid(True)


# Графік залежності дисперсії
plt.subplot(3, 1, 3)

degrees = np.arange(1, max_degree + 1)

plt.plot(degrees, variances, marker='o')

plt.title("Залежність дисперсії від степеня полінома")
plt.xlabel("Степінь полінома")
plt.ylabel("Дисперсія")

plt.grid(True)

plt.tight_layout()
plt.show()
