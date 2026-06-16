#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// QxBin Edge in C - Personal Cubit Simulator on Classical Hardware
// Translated from Python version for performance, embedded/edge use, and learning
// Core: Binary Probability Matrix (2D grid of fractions) + fractional exponent superposition

typedef struct {
    int grid_size;
    double **state;
} QxBinEdge;

// Allocate and initialize a 2D double matrix
double** alloc_matrix(int size) {
    double **mat = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        mat[i] = (double*)malloc(size * sizeof(double));
    }
    return mat;
}

void free_matrix(double **mat, int size) {
    for (int i = 0; i < size; i++) free(mat[i]);
    free(mat);
}

// Normalize so sum of all elements == 1.0 (probability distribution)
void normalize(QxBinEdge *qx) {
    double sum = 0.0;
    for (int i = 0; i < qx->grid_size; i++) {
        for (int j = 0; j < qx->grid_size; j++) {
            sum += qx->state[i][j];
        }
    }
    if (sum > 1e-12) {
        for (int i = 0; i < qx->grid_size; i++) {
            for (int j = 0; j < qx->grid_size; j++) {
                qx->state[i][j] /= sum;
            }
        }
    } else {
        // Fallback uniform
        double val = 1.0 / (qx->grid_size * qx->grid_size);
        for (int i = 0; i < qx->grid_size; i++) {
            for (int j = 0; j < qx->grid_size; j++) {
                qx->state[i][j] = val;
            }
        }
    }
}

// Create new QxBinEdge instance
QxBinEdge* create_qxbin(int grid_size) {
    QxBinEdge *qx = (QxBinEdge*)malloc(sizeof(QxBinEdge));
    qx->grid_size = grid_size;
    qx->state = alloc_matrix(grid_size);

    // Random initial probability matrix (analog starting point)
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            qx->state[i][j] = (double)rand() / RAND_MAX;
        }
    }
    normalize(qx);
    return qx;
}

void free_qxbin(QxBinEdge *qx) {
    free_matrix(qx->state, qx->grid_size);
    free(qx);
}

// Apply QxBin superposition: fractional leaning coin via bias**n and (1-bias)**m
// Builds grid via "linspace" outer-product style, then blends with current state
void apply_superposition(QxBinEdge *qx, double bias, int n, int m) {
    double frac = pow(bias, n);
    double tail = pow(1.0 - bias, m);

    // Create coordinate vector: linspace(frac, tail, grid_size)
    double *vec = (double*)malloc(qx->grid_size * sizeof(double));
    if (qx->grid_size > 1) {
        double step = (tail - frac) / (qx->grid_size - 1);
        for (int i = 0; i < qx->grid_size; i++) {
            vec[i] = frac + i * step;
        }
    } else {
        vec[0] = frac;
    }

    // Allocate temp new matrix for outer(vec, vec)
    double **new_mat = alloc_matrix(qx->grid_size);
    for (int i = 0; i < qx->grid_size; i++) {
        for (int j = 0; j < qx->grid_size; j++) {
            new_mat[i][j] = vec[i] * vec[j];
        }
    }

    // Blend (superposition-like update): average with current
    for (int i = 0; i < qx->grid_size; i++) {
        for (int j = 0; j < qx->grid_size; j++) {
            qx->state[i][j] = (qx->state[i][j] + new_mat[i][j]) / 2.0;
        }
    }

    free(vec);
    free_matrix(new_mat, qx->grid_size);
    normalize(qx);
}

// Probabilistic measurement: collapse to one "classical" outcome (one-hot in the grid)
double** measure(QxBinEdge *qx) {
    // Flatten and weighted random choice
    int total = qx->grid_size * qx->grid_size;
    double *flat = (double*)malloc(total * sizeof(double));
    int idx = 0;
    for (int i = 0; i < qx->grid_size; i++) {
        for (int j = 0; j < qx->grid_size; j++) {
            flat[idx++] = qx->state[i][j];
        }
    }

    // Cumulative sum for weighted sampling
    double r = (double)rand() / RAND_MAX;
    double cum = 0.0;
    int chosen = 0;
    for (int k = 0; k < total; k++) {
        cum += flat[k];
        if (r <= cum) {
            chosen = k;
            break;
        }
    }

    // Create one-hot result matrix
    double **result = alloc_matrix(qx->grid_size);
    for (int i = 0; i < qx->grid_size; i++) {
        for (int j = 0; j < qx->grid_size; j++) {
            result[i][j] = 0.0;
        }
    }
    int row = chosen / qx->grid_size;
    int col = chosen % qx->grid_size;
    result[row][col] = 1.0;

    free(flat);
    return result;
}

// Pretty print the probability matrix
void print_matrix(QxBinEdge *qx, const char *label) {
    printf("\n%s (grid %dx%d):\n", label, qx->grid_size, qx->grid_size);
    for (int i = 0; i < qx->grid_size; i++) {
        for (int j = 0; j < qx->grid_size; j++) {
            printf("%6.3f ", qx->state[i][j]);
        }
        printf("\n");
    }
    printf("Sum = 1.000 (normalized probability)\n");
}

void print_collapsed(double **mat, int size, const char *label) {
    printf("\n%s:\n", label);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%6.3f ", mat[i][j]);
        }
        printf("\n");
    }
}

int main() {
    srand((unsigned int)time(NULL));

    printf("QxBin Edge in C - Personal Cubit Simulator\n");
    printf("Room-temperature Binary Probability Matrix on classical hardware\n\n");

    int grid = 5;
    QxBinEdge *qx = create_qxbin(grid);
    print_matrix(qx, "Initial random probability matrix");

    // Apply strong directional superposition (like Python demo: bias=0.8, n=3, m=1)
    apply_superposition(qx, 0.8, 3, 1);
    print_matrix(qx, "After apply_superposition(bias=0.8, n=3, m=1)");

    // Measure (collapse)
    double **collapsed = measure(qx);
    print_collapsed(collapsed, grid, "Measured (collapsed) classical outcome");

    // Cleanup
    free_matrix(collapsed, grid);
    free_qxbin(qx);

    printf("\nDone. QxBin-C Edge ready for edge deployment, embedded systems, or learning.\n");
    printf("Compile: gcc -o qxbin-c-edge qxbin_edge.c -lm\n");

    return 0;
}