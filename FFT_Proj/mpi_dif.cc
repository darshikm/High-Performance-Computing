#include <iostream>
#include <cstdlib>
#include "mpi.h"
#include <cmath>
#include <cassert>
#include <complex>

#define PI 3.14159

using namespace std;
typedef complex<float> sig;

/*Assumptions for executing this program..
  Let the Problem size be N.. (for doing radix-2 FFT N has to be a power of 2)
  At all times, there are N/2 number of compute nodes (or processes) available to run the program.

  At stage 0, only the first process (root process) executes.. and computes a N-point DFT.
  Next stage, two processes get activated depending on their stage_tag.. the two processes then perform N/2-point DFT independently;

  Similarly, at the last stage.. the stage_tag of all the processes will get activated..
  These processes will then compute 2-point DFT..
  
  The final result will be accumulated by the root process..
*/

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


// The N-point DFT butterfly formula.. for Decimation in Frequency
void dif(float *x_real, float *x_imag, int size, int N) {
    int N_half = size/2, i, k = 0;
    int index = N/size;
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
            bfly_real[i] = x_real[k] + x_real[N_half + k] ;
            bfly_imag[i] = x_imag[k] + x_imag[N_half + k] ;
        }
        else {
            bfly_real[i] = (x_real[k] - x_real[N_half + k])*twiddle_real - (x_imag[k] - x_imag[N_half + k])*twiddle_imag;
            bfly_imag[i] = (x_imag[k] - x_imag[N_half + k])*twiddle_real + (x_real[k] - x_real[N_half + k])*twiddle_imag;
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

//sequential DFT.. when np == 1..
void perform_seq_dft(sig *DFT_X, sig *in, int N)
{
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

   // User inputs np --> no. of processes.. and N --> sized input sample problem
   int N, j = 0, k = 0;
 
    if(argc == 2) {
        N = atoi(argv[1]);
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

    // x cointainer in real and imag parts..
    float *x_real = (float *) malloc((sizeof(float))*N); 
    float *x_imag = (float *) malloc((sizeof(float))*N);

    float *send_x_real, *send_x_imag;
    double time = 0.0, final_time;
    int *stage_tag, *root, s_size = N, opt_np;
    int np, rank, i, stage; 
    const int rank_root = 0;

    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // this is done to deal with np > (N/2)..
    if(np > N/2) opt_np = N/2;
    else opt_np = np;

    stage_tag = (int *)malloc(sizeof(int) * np);
    root = (int *)malloc(sizeof(int) * np);
    stage = (int)ceil(log2(N)); // total stages of 2-point DFTs

    MPI_Status status;

    if(rank == rank_root) {
        x = (sig *)malloc(N*sizeof(sig)); // Input sig defined
        DFT_X = (sig *)malloc(N*sizeof(sig)); // Output sig defined

        send_x_real = (float *) malloc((sizeof(float))*N*np); // the float counter part of sig x
        send_x_imag = (float *) malloc((sizeof(float))*N*np); // try setting size to N..
     
        init_random(x, N); // root will initialize array
     
        if(np == 1) { 
            time -= MPI_Wtime();
            perform_seq_dft(DFT_X, x, N);
            std::cout<<"The output DFT is:\n";
            disp(DFT_X, N);
            time += MPI_Wtime();
            std::cout<<"Elapsed time in single processor := "<<time<<endl;
            MPI_Finalize();
            return 0;
        }
        else if(np != N/2) {
            std::cout<<"Hi, it currently works only for np == N/2\n";
            std::cout<<"The mapping on any arbitrary number of processes to N samples is difficult.. at this point of time.";
            MPI_Finalize();
            return -1;
        }
        else {
            for(i = 0; i < N; i++) {
                x_real[i] = real(x[i]);
                x_imag[i] = imag(x[i]);    
            }
      
            // compute DFT at stage 0..
            dif(x_real, x_imag, s_size, N);
      
            // and send the result to the (np/2) th process..
            MPI_Send(x_real + s_size/2, s_size/2, MPI_FLOAT, np/2, 1, MPI_COMM_WORLD);
            MPI_Send(x_imag + s_size/2, s_size/2, MPI_FLOAT, np/2, 1, MPI_COMM_WORLD);
        }
    }

    //Set stage tag for all processes to be 0 expect root..
    stage_tag[0] = 1; root[0] = -2;
    for(j = 1; j < np; j++) { stage_tag[j] = 0; root[j] = -1; }

    // start time..
    time -= MPI_Wtime();     

    //Main process of FFT..Decimation in Frequency..
    for(i = 1; i < stage; i++)  {
    // preset conditions:
    	s_size = s_size/2; // NB: s_size = N, at stage 0
    	for(j = 0; j < opt_np; j++) {
            k = pow(2, i);
            k = opt_np/k;
            if(k*j < opt_np) {
                stage_tag[k*j] = 1;
                if(root[k*j] == -1) { root[k*j] = k*j - s_size/2; }
                else if(root[k*j] >= 0) { root[k*j] = -2; }  
            } 
    	}

        if(stage_tag[rank] == 1 && root[rank] >= 0) { // first receive.. individual portion of work
            //The receive will block the process until the data is available to it when its stage is set..
            MPI_Recv(x_real, s_size, MPI_FLOAT, root[rank], 1, MPI_COMM_WORLD, &status);
            MPI_Recv(x_imag, s_size, MPI_FLOAT, root[rank], 1, MPI_COMM_WORLD, &status);
        } 
        if(stage_tag[rank] == 1) { // Then compute.. 
            dif(x_real, x_imag, s_size, N); // I will compute DFT on s_size elements.
            if(i < stage - 1) {    // send only upto second last stage..
                MPI_Send(x_real + s_size/2, s_size/2, MPI_FLOAT, (rank + (s_size/4)), stage_tag[rank], MPI_COMM_WORLD);
                MPI_Send(x_imag + s_size/2, s_size/2, MPI_FLOAT, (rank + (s_size/4)), stage_tag[rank], MPI_COMM_WORLD);
            }
        }
    }

    // After All stages are done.. Gather the results from neach process..
    MPI_Gather(x_real, s_size, MPI_FLOAT, send_x_real, s_size, MPI_FLOAT, rank_root, MPI_COMM_WORLD);
    MPI_Gather(x_imag, s_size, MPI_FLOAT, send_x_imag, s_size, MPI_FLOAT, rank_root, MPI_COMM_WORLD); 

    time += MPI_Wtime(); // stop time and take an average time per process..
    MPI_Reduce(&time, &final_time, 1, MPI_DOUBLE, MPI_SUM, rank_root, MPI_COMM_WORLD);

    // now we will generate the sig DFT_X
    if(rank == rank_root) {
        sig *temp = (sig *)malloc(sizeof(sig)*N);
        for(i = 0; i < N; i++) temp[i] = complex<float> (send_x_real[i], send_x_imag[i]);

        // bit reverses the output..
        bit_reverse(DFT_X, temp, N);
        free(temp);   
        //check output..
        //printf("The output DFT is:\n");
        //disp(DFT_X, N);
        std::cout<<"Elapsed time := "<<(final_time/np)<<" sec\n"; 
    }
    MPI_Finalize();
    
    return 0;
}
