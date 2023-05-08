//
// Created by Антон Чанчиков on 08.05.2023.
//
#include <iostream>
#include <vector>
using namespace std;

vector<double> solve_system(vector<vector<double>> A, vector<double> b, double eps=1e-6, int max_iters=1000) {
    int n = A.size();
    vector<double> x(n, 0), prev_x(n, 0);
    for (int i = 0; i < n; i++) {
        x[i] = b[i];
    }
    int iter = 0;
    while (iter < max_iters) {
        prev_x = x;  // Обновляем предыдущее значение x
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum += A[i][j] * prev_x[j];
                }
            }
            if (A[i][i] != 0) {
                x[i] = (b[i] - sum) / A[i][i];
            } else {
                x[i] = (b[i] - sum) / 1e-9;
            }
        }
        bool has_converged = true;
        for (int i = 0; i < n; i++) {
            if (abs(x[i] - prev_x[i]) > eps) {
                has_converged = false;
                break;
            }
        }
        if (has_converged) {
            break;
        }
        iter++;
    }
    return x;
}


int main() {
    int n;
    cin >> n;
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);
//    for (int i = 0; i < n; i++) {
//        for (int j = 0; j < n; j++) {
//            cin >> A[i][j];
//        }
//        cin >> b[i];
//    }
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 10 + 1;  // генерируем случайное число от 1 до 10
            sum += abs(A[i][j]);
        }
        A[i][i] = sum + 1;  // увеличиваем элемент на главной диагонали, чтобы матрица была диагонально-доминантной
        b[i] = rand() % 10 + 1;  // генерируем случайное число от 1 до 10 для вектора b
    }

    vector<double> x = solve_system(A, b);
    for (int i = 0; i < n; i++) {
        cout << "x[" << i << "] = " << x[i] << endl;
    }
    return 0;
}