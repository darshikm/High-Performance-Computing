#include <iostream>
#include <cstdlib>
#include "mpi.h"
#include <assert.h>
#include <complex>

#define PI 3.14159

using namespace std;
typedef complex<float> sig;

// Initialize a unit Step function..
void init_unitStep(sig *in, int N) {
    int i;
    for(i = 0; i < N; i++) {
        in[i] = complex<float> (1, 0);
    }
}

// Initialize random set of values..
void init_random(sig *in, int N) {
    int i;
    for(i = 0; i < N; i++) {
        in[i] = complex<float> ((double) rand () / RAND_MAX, (double) rand () / RAND_MAX);
    }
}

// Display FFT results..
void disp(sig *in, int N) {
    int i;
    for(i = 0; i < N; i++) {
        std::cout<<"X ["<<i<<"] := "<<real(in[i])<<" + j "<<imag(in[i])<<endl;
    }
}


int main(int argc, char *argv[]) {

    // User inputs np := no. of processes.. and N sized input samples
    int N, N_copy, j = 0;
    if(argc == 2) {
        N = atoi(argv[1]);
    }
    else {
        cout<<"argc := "<<argc<<endl;
        for(j = 0; j < argc; j++) {
            std::cout<<"argv at j:= "<<j<<", is := "<<argv[0]<<endl;
        }
        std::cout<<"Invalid execution of the program!\n";
        return -1;
    }
    int np, rank, root = 0, i; 

    // divide the input size N into 'chunk's of elements for each process
    int chunk, remainder;
    N_copy = N;

    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    chunk = N/np;
    remainder = N%np;
    if(remainder != 0) { // uneven distribution
        chunk++;
        N = np*chunk;
    }
    sig *x = (sig *)malloc(N*sizeof(sig)); // Input sig defined
    sig *DFT_X = (sig *)malloc(N*sizeof(sig)); // Output sig defined

    // specify result buffers..for individual processes
    float *result_dft_real = (float *) malloc((sizeof(float))*N);
    float *result_dft_imag = (float *) malloc((sizeof(float))*N); 

    // specify receive buffers by all processes except root.
    float *recv_real = (float *) malloc((sizeof(float))*(chunk));
    float *recv_imag = (float *) malloc((sizeof(float))*(chunk));

    float *send_real, *send_imag; // used for scattering
    float *DFT_real, *DFT_imag; // final buffer
        
    if(rank == root) {
        send_real = (float *) malloc((sizeof(float))*N);
        send_imag = (float *) malloc((sizeof(float))*N);

        DFT_real = (float *) malloc((sizeof(float))*N);
        DFT_imag = (float *) malloc((sizeof(float))*N);

        // initialize sig of N samples..
        init_random(x, N);
        for(i = 0; i < N; i++) {
            send_real[i] = real(x[i]);
            send_imag[i] = imag(x[i]);
        }
        // for debugging..
        //printf("The input sig is:\n");
        //disp(x, N_copy); 
    }

    // Measure time
    double time = 0.0, final_time;
    time -= MPI_Wtime();

    MPI_Scatter(send_real, chunk, MPI_FLOAT, recv_real, chunk, MPI_FLOAT, root, MPI_COMM_WORLD); 
    MPI_Scatter(send_imag, chunk, MPI_FLOAT, recv_imag, chunk, MPI_FLOAT, root, MPI_COMM_WORLD); 
    MPI_Barrier(MPI_COMM_WORLD);

    // do the necessary DFT computation..
    double angle;
    float sumr, sumi;
    
    for(i = 0; i < N; i++) {
        sumr = 0;
        sumi = 0;
       
        for(j = 0; j < chunk; j++) {
            angle = (-2*PI*(chunk*rank + j)*i)/N_copy; // have to give right offset to the angle

            sumr += recv_real[j] * cos(angle) - recv_imag[j] * sin(angle);
            sumi += recv_imag[j] * cos(angle) + recv_real[j] * sin(angle);
        }
        // each process will calculate its respective DFT result..in send buf
        result_dft_real[i] = sumr; 
        result_dft_imag[i] = sumi;
    }

    MPI_Reduce(result_dft_real, DFT_real, N, MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce(result_dft_imag, DFT_imag, N, MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);

    // record time diff..
    time += MPI_Wtime();
    MPI_Reduce(&time, &final_time, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

    if(rank == root) {
        for(i = 0; i < N; i++)	DFT_X[i] = complex<float> (DFT_real[i], DFT_imag[i]);
        // debug..
        //printf("The DFT operation on sig x := \n");
        //disp(DFT_X, N_copy);
        std::cout<<"Elapsed time := "<<(final_time/np)<<" sec\n"; 
    }
    MPI_Finalize();

    return 0;
}
