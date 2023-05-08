#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <fstream>

using namespace std;

vector<double> solve_system(vector<vector<double> >& A, vector<double>& b, double eps = 1e-6, int max_iters = 10000, int rank = 0, int num_procs = 1) {
    int n = A.size();
    vector<double> x(n, 0), prev_x(n, 0);
    for (int i = 0; i < n; i++) {
        x[i] = b[i];
    }
    int iter = 0;
    while (iter < max_iters) {
        prev_x = x;  // Обновляем предыдущее значение x

        // Распространяем текуще значение x между процессами
        MPI_Bcast(&x[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Вычисление частичных сумм для каждого процесса
        vector<double> local_sums(n, 0);
        for (int i = 0; i < n; i++) {
            if (i % num_procs == rank) {  // Каждый процесс обрабатывает свои строки
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        local_sums[i] += A[i][j] * prev_x[j];
                    }
                }
            }
        }

        // Собираем частичные суммы со всех процессов
        vector<double> global_sums(n, 0);
        MPI_Allreduce(&local_sums[0], &global_sums[0], n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Вычисление нового значения x для каждого процесса
        for (int i = 0; i < n; i++) {
            if (i % num_procs == rank) {  // Каждый процесс обрабатывает свои строки
                if (A[i][i] != 0) {
                    x[i] = (b[i] - global_sums[i]) / A[i][i];
                } else {
                    x[i] = (b[i] - global_sums[i]) / 1e-9;
                }
            }
        }

        // Проверка на сходимость
        bool has_converged = true;
        for (int i = 0; i < n; i++) {
            if (abs(x[i] - prev_x[i]) > eps) {
                has_converged = false;
                break;
            }
        }

        // Все процессы должны согласовать флаг сходимости
        int global_convergence = has_converged ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_convergence, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (global_convergence) {
            break;
        }

        iter++;
    }
    return x;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n;
    if (rank == 0) {
        cout << "Enter the size of the matrix: ";
        cin >> n;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Рассылка размера матрицы от процесса 0 к остальным процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<vector<double> > A(n, vector<double>(n));
    vector<double> b(n);

    // Генерация случайных значений матрицы A и вектора b только в процессе 0
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                A[i][j] = rand() % 10 + 1;  // генерируем случайное число от 1 до 10
                sum += abs(A[i][j]);
            }
            A[i][i] = sum + 1;  // увеличиваем элемент на главной диагонали, чтобы матрица была диагонально-доминантной
            b[i] = rand() % 10 + 1;  // генерируем случайное число от 1 до 10 для вектора b
        }
    }

    // Рассылка матрицы A и вектора b от процесса 0 к остальным процессам
    for (int i = 0; i < n; i++) {
        MPI_Bcast(&A[i][0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(&b[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вычисление частичных результатов для каждого процесса
    vector<double> x = solve_system(A, b, 1e-6, 1000, rank, num_procs);

    // Сбор результатов со всех процессов
    vector<vector<double> > all_x(num_procs, vector<double>(n));
    MPI_Allgather(&x[0], n, MPI_DOUBLE, &all_x[0][0], n, MPI_DOUBLE, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Вывод результатов в процессе 0
    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            cout << "Results from process " << i << ":" << endl;
            for (int j = 0; j < n; j++) {
                cout << "x[" << j << "] = " << all_x[i][j] << endl;
            }
        }
    }

    cout << num_procs << " " << duration.count() / 10;

    MPI_Finalize();
    return 0;
}
