#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>

#define Nx 32 // Número de puntos en la dirección x
#define Ny 32 // Número de puntos en la dirección y
#define Nt 200 // Número de pasos de tiempo
#define t_end 5.0
#define Lx 1.0
#define Ly 1.0
#define D 0.1 // Coeficiente de difusión
#define ux 1.0
#define uy 1.0

#define MAX_ITER 1000
#define TOLERANCE 1e-4


typedef float FLOAT;

// CONDICIONES INICIALES

void initial_condition(FLOAT phi[Ny * Nx]) {
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            int index = i * Nx + j;
            phi[index] = 0.0;  // Valor inicial por defecto
        }
    }
}

void apply_circle_temperature(FLOAT phi[Ny * Nx], FLOAT circle_radius, FLOAT circle_temperature) {
    int circle_center_x = Nx / 2;
    int circle_center_y = Ny / 2;
    for (int i = circle_center_x - circle_radius; i <= circle_center_x + circle_radius; ++i) {
        for (int j = circle_center_y - circle_radius; j <= circle_center_y + circle_radius; ++j) {
            if ((i - circle_center_x) * (i - circle_center_x) + (j - circle_center_y) * (j - circle_center_y) <= circle_radius * circle_radius) {
                phi[i + j * Nx] = circle_temperature;
            }
        }
    }
}


// CONDICIONES DE BORDE

// Condiciones de Dirichlet (valores fijos en los bordes)
void apply_dirichlet_boundary_conditions(FLOAT phi[Ny * Nx]) {
    for (int i = 0; i < Ny; ++i) {
        int left_index = i * Nx;
        int right_index = i * Nx + (Nx - 1);
        phi[left_index] = 0.0;   // Borde izquierdo
        phi[right_index] = 0.0;    // Borde derecho
    }
    for (int j = 0; j < Nx; ++j) {
        int bottom_index = j;
        int top_index = (Ny - 1) * Nx + j;
        phi[bottom_index] = 0.0;   // Borde inferior
        phi[top_index] = 0.0;      // Borde superior
    }
}

// Condiciones Periódicas
void apply_periodic_boundary_conditions(FLOAT phi[Ny * Nx]) {

    // Bordes izquierdo y derecho (condición periódica)
    for (int i = 0; i < Ny; ++i) {
        int left_index = i * Nx;
        int right_index = i * Nx + (Nx - 1);
        // Conectar el borde izquierdo con el derecho
        phi[left_index] = phi[right_index - 1];   // Borde izquierdo
        phi[right_index] = phi[left_index + 1];   // Borde derecho
    }
    // Bordes inferior y superior (condición periódica)
    for (int j = 0; j < Nx; ++j) {
        int bottom_index = j;
        int top_index = (Ny - 1) * Nx + j;
        // Conectar el borde inferior con el superior
        phi[bottom_index] = phi[top_index - Nx];   // Borde inferior
        phi[top_index] = phi[bottom_index + Nx];   // Borde superior
    }
}

// Condiciones de Neumann
void apply_neumann_boundary_conditions(FLOAT phi[Ny * Nx]) {
    // Bordes izquierdo y derecho (condición de Neumann)
    for (int i = 0; i < Ny; ++i) {
        int left_index = i * Nx;
        int right_index = i * Nx + (Nx - 1);
        // Derivada nula en el borde izquierdo
        phi[left_index] = phi[left_index + 1];
        // Derivada nula en el borde derecho
        phi[right_index] = phi[right_index - 1];
    }
    // Bordes inferior y superior (condición de Neumann)
    for (int j = 0; j < Nx; ++j) {
        int bottom_index = j;
        int top_index = (Ny - 1) * Nx + j;
        // Derivada nula en el borde inferior
        phi[bottom_index] = phi[bottom_index + Nx];
        // Derivada nula en el borde superior
        phi[top_index] = phi[top_index - Nx];
    }
}

// Generación de Matrices A y B
void generate_coefficient_matrix(int N, int M, FLOAT alpha, FLOAT beta, FLOAT epsilon, FLOAT gamma, FLOAT eta, FLOAT *A, int is_A) {
	//is_A=1 --> A	
	//is_A=0 --> B
    int num_unknowns = N * M;
    for (int i = 0; i < num_unknowns; i++) {
        int row = i / M;
        int col = i % M;
        FLOAT diagonal = is_A ? (1 + alpha) : (1 - alpha);
        *(A + i * num_unknowns + i) = diagonal;
        if (row > 0) {
            *(A + i * num_unknowns + (row - 1) * M + col) = is_A ? (-epsilon) : epsilon;
        }
        if (row < N - 1) {
            *(A + i * num_unknowns + (row + 1) * M + col) = is_A ? beta : (-beta);
        }
        if (col > 0) {
            *(A + i * num_unknowns + row * M + col - 1) = is_A ? (-eta) : eta;
        }
        if (col < M - 1) {
            *(A + i * num_unknowns + row * M + col + 1) = is_A ? gamma : (-gamma);
        }
    }
}

// Producto Matices para arreglos 1D
void product(int N, FLOAT *A, FLOAT *x, FLOAT *result) {
    for (int i = 0; i < N; i++) {
        result[i] = 0.0;
        for (int j = 0; j < N; j++) {
            result[i] += A[i * N + j] * x[j];
        }
    }
}

// Método de Gauss-Seidel para resolver sistemas lineales
void gauss_seidel(int N, FLOAT *A, FLOAT *b, FLOAT *x) {
    FLOAT *new_x = (FLOAT*)malloc(N * sizeof(FLOAT));
    if (new_x == NULL) {
        printf("Error: No se pudo asignar memoria para el vector auxiliar\n");
        exit(1);
    }
    FLOAT residual = TOLERANCE + 1;
    int iter = 0;
    while (residual > TOLERANCE && iter < MAX_ITER) {
        for (int i = 0; i < N; i++) {
            FLOAT sum = 0.0;
            for (int j = 0; j < N; j++) {
                if (j != i) {
                    sum += A[i * N + j] * x[j];
                }
            }
            new_x[i] = (b[i] - sum) / A[i * N + i];
        }
        residual = 0.0;
        for (int i = 0; i < N; i++) {
            residual += fabs(new_x[i] - x[i]);
            x[i] = new_x[i];
        }
        iter++;
    }
    free(new_x);
    if (iter >= MAX_ITER) {
        //printf("Advertencia: Se alcanzó el número máximo de iteraciones sin convergencia.\n");
    } else {
        //printf("Convergencia alcanzada en %d iteraciones.\n", iter);
    }
}

// Copia de Matrices (Esto podría hacerlo directamente en el main)
void copy_new_to_old(FLOAT *phi, FLOAT *phi_new, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            phi[i * N + j] = phi_new[i * N + j];
        }
    }
}

// Save de la matrices que contiene el campo escalar phi
void save_phi(FLOAT *phi, int M, int N, int step) {
    char foldername[] = "data";
    struct stat st = {0};
    if (stat(foldername, &st) == -1) {
        mkdir(foldername, 0700);
    }
    char filename[50];
    sprintf(filename, "%s/temperature_%d.txt", foldername, step);
    FILE *file = fopen(filename, "w");
    if (file != NULL) {
        for (int i = 1; i < M - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                fprintf(file, "%f ", phi[i * N + j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
    } else {
        printf("Error: No se pudo abrir el archivo para escribir.\n");
    }
}

/*
// Funcion para imprimir matrices
void print_coefficient_matrix(int N_x, int N_y, FLOAT *A) {
    int num_unknowns = (N_x) * (N_y);
    for (int i = 0; i < num_unknowns; i++) {
        for (int j = 0; j < num_unknowns; j++) {
            printf("%10.4f ", *(A + i * num_unknowns + j));
        }
        printf("\n");
    }
}
*/

int main() {
	
    FLOAT dx = Lx / (Nx - 1);
    FLOAT dy = Ly / (Ny - 1);
    FLOAT dt = t_end / Nt;
    
    // Componentes de las matrices A y B
    FLOAT alpha = dt * D * (1 / (dx * dx) + 1 / (dy * dy));
    FLOAT beta = (uy * dt) / (4 * dx) - (D * dt) / (2 * (dx * dx));
    FLOAT gamma = (ux * dt) / (4 * dy) - (D * dt) / (2 * (dy * dy));
    FLOAT epsilon = (uy * dt) / (4 * dx) + (D * dt) / (2 * (dx * dx));
    FLOAT eta = (ux * dt) / (4 * dy) + (D * dt) / (2 * (dy * dy));
    
    // Alojamiento dinámico de arrays 1D
    FLOAT *phi = (FLOAT *)malloc(sizeof(FLOAT) * (Nx * Ny));
    FLOAT *ti = (FLOAT *)malloc(sizeof(FLOAT) * (Nx * Ny));
    FLOAT *phi_new = (FLOAT *)malloc(sizeof(FLOAT) * (Nx * Ny));
    int num_unknowns = (Nx) * (Ny);
    FLOAT *A = (FLOAT *)malloc(sizeof(FLOAT) * num_unknowns * num_unknowns);
    FLOAT *B = (FLOAT *)malloc(sizeof(FLOAT) * num_unknowns * num_unknowns);
    
    initial_condition(phi);
    initial_condition(phi_new);

    generate_coefficient_matrix(Ny, Nx, alpha, beta, epsilon, gamma, eta, A, 1);
    generate_coefficient_matrix(Ny, Nx, alpha, beta, epsilon, gamma, eta, B, 0);

    int save_interval = 10;
    
    
	#ifdef CIRCLE
    apply_circle_temperature (phi,1.,100);
    #endif
    
    
    for (int step = 0; step < Nt; ++step) {

        #ifdef DIRICHLET
        	apply_dirichlet_boundary_conditions;
		#endif
		
		#ifdef PERIODIC
        	apply_periodic_boundary_conditions;
		#endif
		
		#ifdef NEUMANN
        	apply_neumann_boundary_conditions;
		#endif
		
        product(num_unknowns, B, phi, ti);
        
        gauss_seidel(num_unknowns, A, ti, phi_new);
        
        copy_new_to_old(phi, phi_new, Ny, Nx);
        
        if (step % save_interval == 0) {
        	printf("%d \n",step);
            save_phi(phi_new, Nx, Ny, step);
        }
    }

    //save_phi(phi_new, Ny, Nx, Nt);

    free(A);
    free(B);
    free(phi);
    free(phi_new);
    free(ti);

    return 0;
}

