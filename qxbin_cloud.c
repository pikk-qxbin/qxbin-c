#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// QxBin Cloud in C - Scalable Ensemble of Cubit Chains
// Multiple Binary Probability Matrices evolved in parallel (serial loop here; add OpenMP for real speed)
// Same fractional exponent logic as Edge version
// Ideal for Monte Carlo, optimization, batch probabilistic workloads on servers or edge clusters

typedef struct {
    int num_cubits;
    int grid_size;
    double ***states;   // [cubits][grid][grid]
} QxBinCloud;

// Allocate 3D array: num_cubits x grid x grid
double*** alloc_3d(int num_cubits, int grid) {
    double ***arr = (double***)malloc(num_cubits * sizeof(double**));
    for (int c = 0; c < num_cubits; c++) {
        arr[c] = (double**)malloc(grid * sizeof(double*));
        for (int i = 0; i < grid; i++) {
            arr[c][i] = (double*)malloc(grid * sizeof(double));
        }
    }
    return arr;
}

void free_3d(double ***arr, int num_cubits, int grid) {
    for (int c = 0; c < num_cubits; c++) {
        for (int i = 0; i < grid; i++) free(arr[c][i]);
        free(arr[c]);
    }
    free(arr);
}

// Normalize a single matrix
void normalize_matrix(double **mat, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++) sum += mat[i][j];
    if (sum > 1e-12) {
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) mat[i][j] /= sum;
    } else {
        double val = 1.0 / (size * size);
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) mat[i][j] = val;
    }
}

// Create cloud simulator
QxBinCloud* create_qxbin_cloud(int num_cubits, int grid_size) {
    QxBinCloud *cloud = (QxBinCloud*)malloc(sizeof(QxBinCloud));
    cloud->num_cubits = num_cubits;
    cloud->grid_size = grid_size;
    cloud->states = alloc_3d(num_cubits, grid_size);

    for (int c = 0; c < num_cubits; c++) {
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                cloud->states[c][i][j] = (double)rand() / RAND_MAX;
            }
        }
        normalize_matrix(cloud->states[c], grid_size);
    }
    return cloud;
}

void free_qxbin_cloud(QxBinCloud *cloud) {
    free_3d(cloud->states, cloud->num_cubits, cloud->grid_size);
    free(cloud);
}

// Evolve one cubit chain with given bias, n, m
void evolve_one_cubit(double **mat, int size, double bias, int n, int m) {
    double frac = pow(bias, n);
    double tail = pow(1.0 - bias, m);

    double *vec = (double*)malloc(size * sizeof(double));
    if (size > 1) {
        double step = (tail - frac) / (size - 1);
        for (int i = 0; i < size; i++) vec[i] = frac + i * step;
    } else {
        vec[0] = frac;
    }

    double **new_mat = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) new_mat[i] = (double*)malloc(size * sizeof(double));

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            new_mat[i][j] = vec[i] * vec[j];

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            mat[i][j] = (mat[i][j] + new_mat[i][j]) / 2.0;

    free(vec);
    for (int i = 0; i < size; i++) free(new_mat[i]);
    free(new_mat);

    normalize_matrix(mat, size);
}

// Evolve all cubit chains (serial; easy to parallelize with OpenMP)
void evolve_chains(QxBinCloud *cloud) {
    for (int c = 0; c < cloud->num_cubits; c++) {
        double bias = 0.5 + ((double)rand() / RAND_MAX) * 0.35;  // 0.5 - 0.85
        int nn = 1 + rand() % 4;   // 1-4
        int mm = 1 + rand() % 4;
        evolve_one_cubit(cloud->states[c], cloud->grid_size, bias, nn, mm);
    }
}

// Compute aggregate mean matrix across all cubits
double** get_aggregate(QxBinCloud *cloud) {
    int g = cloud->grid_size;
    double **agg = (double**)malloc(g * sizeof(double*));
    for (int i = 0; i < g; i++) agg[i] = (double*)malloc(g * sizeof(double));

    for (int i = 0; i < g; i++) {
        for (int j = 0; j < g; j++) {
            double sum = 0.0;
            for (int c = 0; c < cloud->num_cubits; c++) {
                sum += cloud->states[c][i][j];
            }
            agg[i][j] = sum / cloud->num_cubits;
        }
    }
    return agg;
}

void free_aggregate(double **agg, int g) {
    for (int i = 0; i < g; i++) free(agg[i]);
    free(agg);
}

// Simple optimization loop: evolve until aggregate mean close to target
void optimize_to_target(QxBinCloud *cloud, double target_mean, int max_steps) {
    for (int step = 0; step < max_steps; step++) {
        evolve_chains(cloud);

        double **agg = get_aggregate(cloud);
        double current = 0.0;
        int g = cloud->grid_size;
        for (int i = 0; i < g; i++)
            for (int j = 0; j < g; j++) current += agg[i][j];
        current /= (g * g);
        free_aggregate(agg, g);

        if (fabs(current - target_mean) < 0.003) {
            printf("Converged after %d steps | mean prob ≈ %.4f\n", step + 1, current);
            return;
        }
    }
    printf("Reached max steps without exact convergence.\n");
}

// Print a matrix (for aggregate or single)
void print_matrix(double **mat, int size, const char *label) {
    printf("\n%s:\n", label);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) printf("%6.3f ", mat[i][j]);
        printf("\n");
    }
}

// Export aggregate to CSV (easy to plot in Python, Excel, gnuplot)
void export_aggregate_csv(QxBinCloud *cloud, const char *filename) {
    double **agg = get_aggregate(cloud);
    int g = cloud->grid_size;
    FILE *f = fopen(filename, "w");
    if (!f) {
        printf("Error opening %s for writing.\n", filename);
        free_aggregate(agg, g);
        return;
    }
    for (int i = 0; i < g; i++) {
        for (int j = 0; j < g; j++) {
            fprintf(f, "%.6f", agg[i][j]);
            if (j < g-1) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("Exported aggregate to %s (ready for visualization)\n", filename);
    free_aggregate(agg, g);
}

int main() {
    srand((unsigned int)time(NULL));

    printf("QxBin Cloud in C — Scalable Cubit Ensemble Simulator\n");
    printf("Room-temperature quantum-inspired batch processing\n\n");

    int num_cubits = 24;
    int grid = 6;
    QxBinCloud *cloud = create_qxbin_cloud(num_cubits, grid);
    printf("Created ensemble of %d cubit chains (grid %dx%d)\n", num_cubits, grid, grid);

    printf("Evolving chains and optimizing toward target mean 0.68...\n");
    optimize_to_target(cloud, 0.68, 80);

    double **agg = get_aggregate(cloud);
    print_matrix(agg, grid, "Aggregate probability landscape (mean across ensemble)");
    free_aggregate(agg, grid);

    export_aggregate_csv(cloud, "qxbin_aggregate.csv");

    free_qxbin_cloud(cloud);

    printf("\n✅ Done. Compile: gcc -o qxbin-c-cloud qxbin_cloud.c -lm\n");
    printf("Add -fopenmp for parallel evolve_chains if desired.\n");
    printf("CSV ready for plotting the probability surface.\n");
    printf("Perfect for Pikk edge clusters or server-side Monte Carlo workloads.\n");

    return 0;
}