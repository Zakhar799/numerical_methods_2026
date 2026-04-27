import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def read_matrix_txt(filename):
    return np.loadtxt(filename, delimiter='\t')

def read_vector_txt(filename):
    return np.loadtxt(filename, delimiter='\t').reshape(-1, 1)

def write_data_txt(filename, data):
    np.savetxt(filename, data, fmt='%.6f', delimiter='\t')

def multiply_matrix_vector(A, X):
    return A @ X

def vector_norm(V):
    return np.max(np.abs(V))

def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        U[i][i] = 1.0
        for j in range(i, n):
            sum1 = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = A[j][i] - sum1
        for j in range(i + 1, n):
            sum2 = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = (A[i][j] - sum2) / L[i][i]

    return L, U

def solve_lu(L, U, B):
    n = len(B)
    Z = np.zeros_like(B)
    X = np.zeros_like(B)

    for i in range(n):
        s = sum(L[i][j] * Z[j] for j in range(i))
        Z[i] = (B[i] - s) / L[i][i]

    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * X[j] for j in range(i + 1, n))
        X[i] = Z[i] - s

    return X

N = 100
A = np.random.rand(N, N) * 10
for i in range(N):
    A[i, i] += np.sum(np.abs(A[i])) + 10

X_exact = np.full((N, 1), 2.5)
B = multiply_matrix_vector(A, X_exact)

write_data_txt('matrix_A_lab7.txt', A)
write_data_txt('vector_B_lab7.txt', B)
print("Дані збережено у текстові файли: matrix_A_lab7.txt та vector_B_lab7.txt")

# Зчитування
A_loaded = read_matrix_txt('matrix_A_lab7.txt')
B_loaded = read_vector_txt('vector_B_lab7.txt')

L, U = lu_decomposition(A_loaded)
write_data_txt('matrix_L_lab7.txt', L)
write_data_txt('matrix_U_lab7.txt', U)
print("LU розклад збережено у файли 'matrix_L_lab7.txt' та 'matrix_U_lab7.txt'")

X_calc = solve_lu(L, U, B_loaded)

residual_initial = multiply_matrix_vector(A_loaded, X_calc) - B_loaded
initial_error = vector_norm(residual_initial)
print(f"Початкова похибка (max|AX - B|): {initial_error:.4e}")

epsilon = 1e-14
max_iterations = 20
X_refined = np.copy(X_calc)

residual_norms = [initial_error]

print("\n--- Ітераційне уточнення ---")
for i in range(max_iterations):
    # R = B - AX
    R = B_loaded - multiply_matrix_vector(A_loaded, X_refined)
    current_error = vector_norm(R)

    if current_error <= epsilon:
        print(f"Умова зупинки досягнута! (похибка <= {epsilon}) на ітерації {i}")
        break

    delta_X = solve_lu(L, U, R)
    X_refined += delta_X

    new_error = vector_norm(B_loaded - multiply_matrix_vector(A_loaded, X_refined))
    residual_norms.append(new_error)

    print(f"Ітерація {i + 1}: норма нев'язки = {new_error:.4e}")

final_error = vector_norm(multiply_matrix_vector(A_loaded, X_refined) - B_loaded)
print(f"\nКінцева похибка після уточнення: {final_error:.4e}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
iterations = range(len(residual_norms))
plt.plot(iterations, residual_norms, marker='o', linestyle='-', color='b', linewidth=2)
plt.yscale('log')
plt.title("Збіжність ітераційного уточнення", fontsize=14)
plt.xlabel("Номер ітерації", fontsize=12)
plt.ylabel("Норма нев'язки (логарифмічна шкала)", fontsize=12)
plt.xticks(iterations)
plt.grid(True, which="both", linestyle="--", alpha=0.7)

plt.subplot(1, 2, 2)
error_initial_components = np.abs(X_exact - X_calc).flatten()
error_final_components = np.abs(X_exact - X_refined).flatten()

plt.plot(error_initial_components, label='До уточнення (X_calc)', alpha=0.8, color='red')
plt.plot(error_final_components, label='Після уточнення (X_refined)', alpha=0.8, color='green')
plt.yscale('log')
plt.title("Абсолютна похибка кожної компоненти вектора $X$", fontsize=14)
plt.xlabel("Індекс компоненти вектора", fontsize=12)
plt.ylabel("$|X_{exact} - X_{calc}|$ (логарифмічна шкала)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, which="both", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()