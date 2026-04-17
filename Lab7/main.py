import numpy as np
import matplotlib.pyplot as plt

N = 100
np.random.seed(42)

A = np.random.rand(N, N) * 10
for i in range(N):
    A[i, i] += np.sum(np.abs(A[i])) + 10

X_exact = np.full(N, 2.5)

def matrix_vector_mult(M, v):
    n = len(v)
    res = np.zeros(n)
    for i in range(len(M)):
        res[i] = sum(M[i][j] * v[j] for j in range(n))
    return res


B = matrix_vector_mult(A, X_exact)

np.savetxt('matrix_A_lab8.txt', A, fmt='%.6f', delimiter='\t')
np.savetxt('vector_B_lab8.txt', B, fmt='%.6f', delimiter='\t')
print("Дані збережено у файли matrix_A_lab8.txt та vector_B_lab8.txt")

A_loaded = np.loadtxt('matrix_A_lab8.txt', delimiter='\t')
B_loaded = np.loadtxt('vector_B_lab8.txt', delimiter='\t')
A = A_loaded
B = B_loaded


def vector_norm(v1, v2):
    return max(abs(a - b) for a, b in zip(v1, v2))

def simple_iteration(A, B, X_true, epsilon=1e-7, max_iter=2000):
    n = len(B)
    C = np.zeros((n, n))
    d = np.zeros(n)
    errors = []

    for i in range(n):
        d[i] = B[i] / A[i][i]
        for j in range(n):
            if i != j:
                C[i][j] = -A[i][j] / A[i][i]

    X = np.ones(n)
    errors.append(vector_norm(X, X_true))

    for k in range(max_iter):
        X_new = np.zeros(n)
        for i in range(n):
            X_new[i] = sum(C[i][j] * X[j] for j in range(n)) + d[i]

        current_error_true = vector_norm(X_new, X_true)
        errors.append(current_error_true)

        if vector_norm(X_new, X) < epsilon:
            return X_new, k + 1, errors
        X = X_new

    return X, max_iter, errors


def jacobi(A, B, X_true, epsilon=1e-7, max_iter=2000):
    n = len(B)
    X = np.ones(n)
    errors = []
    errors.append(vector_norm(X, X_true))

    for k in range(max_iter):
        X_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i][j] * X[j] for j in range(n) if i != j)
            X_new[i] = (B[i] - s) / A[i][i]

        current_error_true = vector_norm(X_new, X_true)
        errors.append(current_error_true)

        if vector_norm(X_new, X) < epsilon:
            return X_new, k + 1, errors
        X = X_new

    return X, max_iter, errors


def seidel(A, B, X_true, epsilon=1e-7, max_iter=2000):
    n = len(B)
    X = np.ones(n)
    errors = []
    errors.append(vector_norm(X, X_true))

    for k in range(max_iter):
        X_new = np.copy(X)
        for i in range(n):
            s1 = sum(A[i][j] * X_new[j] for j in range(i))
            s2 = sum(A[i][j] * X[j] for j in range(i + 1, n))
            X_new[i] = (B[i] - s1 - s2) / A[i][i]

        current_error_true = vector_norm(X_new, X_true)
        errors.append(current_error_true)

        if vector_norm(X_new, X) < epsilon:
            return X_new, k + 1, errors
        X = X_new

    return X, max_iter, errors

epsilon = 1e-7
print(f"\n=== Розв'язання системи {N}x{N} ===")
print(f"Точність (epsilon) = {epsilon}\n")

X_si, iter_si, errors_si = simple_iteration(A, B, X_exact, epsilon=epsilon)
err_si = vector_norm(X_si, X_exact)
print(f"1. Метод простих ітерацій:")
print(f"   Ітерацій: {iter_si}")
print(f"   Кінцева похибка: {err_si:.4e}\n")

X_jacobi, iter_jacobi, errors_jacobi = jacobi(A, B, X_exact, epsilon=epsilon)
err_jacobi = vector_norm(X_jacobi, X_exact)
print(f"2. Метод Якобі:")
print(f"   Ітерацій: {iter_jacobi}")
print(f"   Кінцева похибка: {err_jacobi:.4e}\n")

X_seidel, iter_seidel, errors_seidel = seidel(A, B, X_exact, epsilon=epsilon)
err_seidel = vector_norm(X_seidel, X_exact)
print(f"3. Метод Зейделя:")
print(f"   Ітерацій: {iter_seidel}")
print(f"   Кінцева похибка: {err_seidel:.4e}\n")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(errors_si, label='Метод простих ітерацій', color='blue', linestyle='-')
plt.plot(errors_jacobi, label='Метод Якобі', color='green', linestyle='--')
plt.plot(errors_seidel, label='Метод Зейделя', color='red', linestyle='-', linewidth=2)

plt.yscale('log')
plt.title(f"Швидкість збіжності (N={N}, eps={epsilon})", fontsize=14)
plt.xlabel("Номер ітерації", fontsize=12)
plt.ylabel("Похибка $||X^{(k)} - X_{exact}||_\infty$ (log)", fontsize=12)
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()