#include <iostream>
#include <cstdlib>
#include "mpi.h"

#include <math.h>
#include <cassert>
#include <complex>

#define PI 3.14159

using namespace std;
typedef complex<float> sig;

//np := no. of processes
// first we begin by assuming that there are N/2 no. of processes available to me
// we are not dealing with np < N/2 at the moment, which involves a mapping input array to processes at various stages of the butterfly diagram.

// this initialize array..
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

// This does the bit reversal for input sequence
//Example:
// 000 -> 000
// 100 -> 001 ; index at 4 is moved to index at 1..
//Reshuffle Src array into bit-reverse patterned indices..
void bit_reverse(sig *Dest, sig *Src, int N) {
    int i, j, k, bits;
    bits = (int) ceil(log2(N));

    //#pragma omp parallel for schedule(static)
    for(i = 0; i < N; i++) Dest[i] = Src[i]; // copy first

    //#pragma omp parallel for schedule(static)
    for(i = 0; i < N; i++) {
        j = 0;
        for(k = 0; k < bits; k++) {
           if (i & ((int)pow(2, k))) { j += ((int)pow(2, bits-k-1)); }
        }  
        // swap if the reversed index is greater than the original
        if(j > i) { 
            Dest[i] = Src[j];
            Dest[j] = Src[i];
        }
    }
}


// The N-point DFT butterfly formula.. for decimation in Time..
// for now I don't think we need info about rank and stage, size will take care of it.
void dit(float *x_real, float *x_imag, int size, int stage, int N) { 
    int N_half = size/2, i, k = 0;
    int index = pow(2, (stage + 1));
    index = N/index;

    float twiddle_real;
    float twiddle_imag;
    float *bfly_real = (float *)malloc(sizeof(float)*size);
    float *bfly_imag = (float *)malloc(sizeof(float)*size);

    // Do 2 Point DFT for N half iterations
    for (i = 0; i < size; i++) {
        if(i == N_half) k = 0;
        twiddle_real = cos((-2*PI*k*index)/N); // will have to recheck the twiddle factor and this logic
        twiddle_imag = sin((-2*PI*k*index)/N);

        if(i < N_half) {
            bfly_real[i] = x_real[k] + x_real[N_half + k] * twiddle_real - x_imag[N_half + k] * twiddle_imag;
            bfly_imag[i] = x_imag[k] + x_real[N_half + k] * twiddle_imag + x_imag[N_half + k] * twiddle_real;
        }
        else {
            bfly_real[i] = x_real[k] - x_real[N_half + k] * twiddle_real + x_imag[N_half + k] * twiddle_imag;
            bfly_imag[i] = x_imag[k] - x_real[N_half + k] * twiddle_imag - x_imag[N_half + k] * twiddle_real;
        }
        k++;	
    } 
    /** In-place results */
    for (i = 0; i < size; i++) {
        x_real[i] = bfly_real[i];
        x_imag[i] = bfly_imag[i];
    }
    free(bfly_real); 
    free(bfly_imag); 
}  

// sequential DFT..when np == 1
void perform_seq_dft(sig *DFT_X, sig *in, int N) {
    int i, j;
    double angle;
    float sumr, sumi;

    for(i = 0; i < N; i++) {
        sumr = 0.0;
        sumi = 0.0;
        for(j = 0; j < N; j++) {
            angle = -(2*PI*i*j)/N;
            sumr += real(in[j]) * cos(angle) - imag(in[j]) * sin(angle);
            sumi += imag(in[j]) * cos(angle) + real(in[j]) * sin(angle) ;
        }
        DFT_X[i] = complex<float> (sumr, sumi);
    }
}

int main(int argc, char *argv[]) {
// User inputs np := no. of processes.. and N sized input sample via command line argument

    int N, j = 0;
    if(argc == 2) {
        N = atoi(argv[1]);
        //printf("Size x[N] := %d\n",N);  //debug
    }
    else {
        std::cout<<"argc := "<<argc<<endl;
        for(j = 0; j < argc; j++) {
            std::cout<<"argv at j:= "<<j<<", is := "<<argv[0]<<endl;
        }
        std::cout<<"Invalid execution of the program!\n";
        return -1;
    }

    //Input, Output sigs
    sig *x, *DFT_X;
    double time = 0.0, final_time;

    // x cointainer in real and imag parts..
    float *x_real = (float *) malloc((sizeof(float))*N); // the float counter part of sig x
    float *x_imag = (float *) malloc((sizeof(float))*N);

    float *send_x_real, *send_x_imag;
    int *stage_tag, stage_mark, s_size = 0;
    int np, rank, root = 0, i, stage; 
       
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    stage_tag = (int *)malloc(sizeof(int) * np);
    stage_mark = 2 * np;
    stage = (int)ceil(log2(N)); // total stages of 2-point DFTs

    MPI_Status status;
        
    if(rank == root) {
        x = (sig *)malloc(N*sizeof(sig)); // Input sig defined
        DFT_X = (sig *)malloc(N*sizeof(sig)); // Output sig defined

        send_x_real = (float *) malloc((sizeof(float))*N*np); // the float counter part of sig x
        send_x_imag = (float *) malloc((sizeof(float))*N*np);

        init_random(x, N); // root will initialize array
        //printf("The input sig is:\n");
        //disp(x, N);

        if(np == 1) { 
            time -= MPI_Wtime();
            perform_seq_dft(DFT_X, x, N);
            //printf("The output DFT is:\n");
            //disp(DFT_X, N);
            time += MPI_Wtime();
            std::cout<<"The elasped time for single processor is: "<<time<<endl;
             
            MPI_Finalize();
            return 0;
        }
        else {
            // root bit reverses the input..
            bit_reverse(DFT_X, x, N);

            // assign to mpi send_buffer
            for(i = 0; i < N; i++) {
                send_x_real[i] = real(DFT_X[i]);
                send_x_imag[i] = imag(DFT_X[i]);    
            }
            // work of root ends here
            //printf(" No. of stages := %d\n", stage);
        }
    }

    // start time...
    time -= MPI_Wtime();
     
    // Main process of FFT..DIT..
    for(i = 0; i < stage; i++) {
        s_size = pow(2, i + 1);
        // set stage tags for valid processes..
        stage_mark = stage_mark/2;
        for(j = 0; j < np; j ++) {
            if(j < stage_mark) stage_tag[j] = 1;
            else stage_tag[j] = 0;
        }
        // Master will send data to all the valid processes by making use of the tag..
        if(rank == root) {
            // root will send data to all other processes
            for(j = 1; j < np; j++) {
                if(stage_tag[j] == 1) {
                    MPI_Send(send_x_real + s_size*j, s_size, MPI_FLOAT, j, stage_tag[j], MPI_COMM_WORLD);
                    MPI_Send(send_x_imag + s_size*j, s_size, MPI_FLOAT, j, stage_tag[j], MPI_COMM_WORLD);
                }
            } 
            // copy contents from send buf to x_real and x_imag..at each stage..
            for(j = 0; j < s_size; j++) {
                x_real[j] = send_x_real[j];
                x_imag[j] = send_x_imag[j];
            }
            dit(x_real, x_imag, s_size, i, N);
        }
        else if(stage_tag[rank] == 1) { //If I am a valid process
            // I will wait till I receive data from the root
            MPI_Recv(x_real, s_size, MPI_FLOAT, root, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(x_imag, s_size, MPI_FLOAT, root, 1, MPI_COMM_WORLD, &status);
            
            // All valid processes can now compute s_size point dft
            dit(x_real, x_imag, s_size, i, N); // (real part, imag part, array size, rank, nth stage, N)   
        }
        // Now I will Gather all the s_size DFT data back in the root process of the master.. Gather at each stage is most expensive
        MPI_Gather(x_real, s_size, MPI_FLOAT, send_x_real, s_size, MPI_FLOAT, root, MPI_COMM_WORLD);
        MPI_Gather(x_imag, s_size, MPI_FLOAT, send_x_imag, s_size, MPI_FLOAT, root, MPI_COMM_WORLD);
    }

    // check elapsed time of all processes
    time += MPI_Wtime();

    // reduce it in an final_time buffer for average..
    MPI_Reduce(&time, &final_time, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD); 

    // now we will generate the sig DFT_X
    if(rank == root) {
        for(i = 0; i < N; i++) DFT_X[i] = complex<float> (send_x_real[i], send_x_imag[i]);	// maintain back the sig structure in freq.

        // debug..
        //printf("The output DFT is:\n");
        //disp(DFT_X, N);
        std::cout<<"The total elasped time is: "<<(final_time/np)<<endl;
    }
    MPI_Finalize();
    return 0;
}
