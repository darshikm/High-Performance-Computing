#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "timer.cc"

#define PI 3.14159
#define DEBUG 0

//Complex number structure
struct complex { 
    double real;    //Real part of the number
    double imag;    //imag part of the number
} typedef complex;

// Initialize a unit Step function..
void init_unitStep(complex *in, int N) {
    int i;
    for(i = 0; i < N; i++) {
        in[i].real = 1;
        in[i].imag = 0;
    }
}

// Initialize random set of values..
void init_random(complex *in, int N) {
    int i;
    for(i = 0; i < N; i++) {
        in[i].real = (double) rand () / RAND_MAX;
        in[i].imag = (double) rand () / RAND_MAX;
    }
}

//Reshuffle Src array into bit-reverse patterned indices..
void bit_reverse(complex *Dest, complex *Src, int N) {
    int i, j, k, bits;
    bits = (int) ceil(log2(N));

    for(i = 0; i < N; i++) Dest[i] = Src[i]; // copy first

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

// Compare Arrays..
int cmpArr (complex *a, complex *b, int N) {
    int cnt = 0, i, k = 0;
    
    for(i = 0; i < N; i++) {
        k = 0;
        if( (abs(a[i].real) - abs(b[i].real)) > 1e-2 ) { printf("Due to real, error at index := %d\n", i); k++; }

        if( (abs(a[i].imag) - abs(b[i].imag)) > 1e-2 ) { printf("Due to imag, error at index := %d\n", i); k++; } 
        if(k != 0) cnt++;
    }
    return cnt;
}

//Decimation in Frequency..
void dif_fft(complex *X, complex *data, int N) {
    int N_half = N/2, i, k;
    complex twiddle, bfly[2];
    /** Do 2 Point DFT */
    for (i = 0; i < N_half; i++) {
    	twiddle.real = cos((-2*PI*i)/N);
     	twiddle.imag = sin((-2*PI*i)/N);
    	
    	bfly[0].real = (data[i].real + data[N_half + i].real);
    	bfly[0].imag = (data[i].imag + data[N_half + i].imag);
    	
    	bfly[1].real = (data[i].real - data[N_half + i].real) * twiddle.real - ((data[i].imag - data[N_half + i].imag) * twiddle.imag); 
    	bfly[1].imag = (data[i].imag - data[N_half + i].imag) * twiddle.real + ((data[i].real - data[N_half + i].real) * twiddle.imag);

    	/** In-place results */
    	for (k = 0; k < 2; k++) {
    	    data[i + N_half*k].real = bfly[k].real;
    	    data[i + N_half*k].imag = bfly[k].imag;
    	}
    }
    
    memcpy(X, data, N);
    /** Don't recurse if we're down to one butterfly */
    if (N_half > 1) {
        for (k = 0; k < 2; k++)  dif_fft(X, data + (N_half*k), N_half);
    }
} 

//Decimation in time..
void dit_fft(complex *X, complex *in, int N) {
    complex *X_G, *X_H;
    complex *x_even, *x_odd;
    int i = 0;

    //Memory allocation
    X_G = (complex*)malloc(sizeof(complex)* N/2);
    X_H = (complex*)malloc(sizeof(complex)* N/2);

    x_even = (complex*)malloc(sizeof(complex)*(N/2));
    x_odd = (complex*)malloc(sizeof(complex)*(N/2));

    // limiting condition for recursion..
    if(N == 2) {
        X[0].real = in[0].real + in[1].real;
        X[0].imag = in[0].imag + in[1].imag;
        X[1].real = in[0].real - in[1].real;
        X[1].imag = in[0].imag - in[1].imag;
        return;
    }

    //Split the input for recursion (Divide Step)
    for(i = 0; i < N; i++) {
        if(i%2==0) { 	// for even indices
            x_even[i/2].real = in[i].real;
            x_even[i/2].imag = in[i].imag;
        }
        else {  		// for odd indices
            x_odd[(i-1)/2].real = in[i].real;
            x_odd[(i-1)/2].imag = in[i].imag;
        }
    }
    
    //Recurse.. 
    dit_fft(X_G, x_even, N/2);
    dit_fft(X_H, x_odd, N/2);
     
    //Combine Step..
    for(i = 0; i < N; i++) {
        X[i].real = X_G[i%(N/2)].real + X_H[i%(N/2)].real * cos((2*PI*i)/N) + X_H[i%(N/2)].imag * sin((2*PI*i)/N);
        X[i].imag = X_G[i%(N/2)].imag + X_H[i%(N/2)].imag * cos((2*PI*i)/N) - X_H[i%(N/2)].real * sin((2*PI*i)/N);
    }

    //deallocate the memory
    free(X_G);
    free(X_H);
    free(x_even);
    free(x_odd);
}

// Display FFT results..
void disp(complex *in, int N) {
    int i;
    for(i = 0; i < N; i++) {
        printf("X [%d] := %.2f + j %.2f \n", i, in[i].real, in[i].imag);
    }
}

int main(int argc, char *argv[]) {
    int N = -1, i = 0;

    //Assuming the N input is a power of 2.. read value of N from command line.
    if(argc == 2) {
        N = atoi (argv[1]);
        assert (N > 1);
    }
    else {
        fprintf (stderr, "usage: %s <N>\n", argv[0]);
        exit (EXIT_FAILURE);
    }

    //verify if N is a power of 2..
    int checkN = N;
    while(checkN >= 2) {
        if(checkN%2 == 0) {
            checkN = checkN/2;
            i++;
        }
        else {
            printf("The given N is not a power of 2. Program Terminating!\n");
            return -1;
        }
    }
    printf("N := %d is the %d - th power of 2\n", N, i);

    complex *x;
    complex *DIT_X, *DIF_X; //Different methods.. and their output vectors

    //allocating memory for the input
    x = (complex *)malloc(N*sizeof(complex));
    DIT_X = (complex *)malloc(N*sizeof(complex));
    DIF_X = (complex *)malloc(N*sizeof(complex));

    init_random(x, N);

    /*create timers */
    struct stopwatch_t* timer = NULL;
    stopwatch_init ();
    timer = stopwatch_create ();
    assert (timer);
    
    printf("\nStaring DIT...\n");
    if(DEBUG) disp(x, N);

    stopwatch_start(timer);
    //Perform..decimation in time
    dit_fft(DIT_X, x, N);
    long double t_dit = stopwatch_stop(timer);
    printf("Time taken to evaluate DIT_FFT := %Lg\n", t_dit);
    if(DEBUG) {
        printf("\n Decimation in Time FFT: \n");
        disp(DIT_X, N);
    }
    
    printf("\nStaring DIF...\n");
    if(DEBUG) disp(x, N);
    stopwatch_start(timer);
    //Perform.. decimation in frequency
    dif_fft(DIF_X, x, N);
    long double t_dif = stopwatch_stop(timer);
    bit_reverse(DIF_X, x, N);
    printf("Time taken to evaluate DIF_FFT := %Lg\n", t_dif);
    if(DEBUG) {
        printf("\n Decimation in Frequency FFT: \n");
        disp(DIF_X, N);
    }
    
    //Compare compute times..
    printf("\n==================\nCompare Decimation in Freq with Decimation in Time \n\n");
    int error_count = cmpArr(DIT_X, DIF_X, N);
    printf("The error count is: %d\n", error_count);
    printf("The ratio (t_dit)/(t_dif) := %Lg\n", t_dit/t_dif);

    return 0;
}
