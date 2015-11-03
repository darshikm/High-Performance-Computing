/**
 *  Parallelize the quick sort algorithm to sort large arrays.. on high performance cluster using OpenMP
 *	cluster link: hpc.oit.uci.edu
 *	A few enhancements need to be made in this program..

 *	The program can potentially scale differently on different computer systems and compiling with -fopenmp flag in gcc,
 	and performing thread-binding by setting the GOMP_CPU_AFFINITY variable in Linux is recommended 
 	(depending on the architecture of the shared memory).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "sort.hh"

/* The algorithm routine to perform prefix scan in parallel is recommended.. */
void prefix_scan(int *A, int N, int *A_new) {
	int i, sum = 0;
	for(i=0;i<N;i++) {
		A_new[i] = sum + A[i];
  		sum = A_new[i];
  	}
}

void partition(int N, keytype* A, int k, int *lower, int *upper) {
	if(N <= 1) {
		*lower = 0;
		*upper = N;
   		return;
   	}
  	else {
  		int i = 0, count = 0;
  		keytype pivot = A[k];
  
  		/* Create new sub Arrays */
  		keytype *A_orig = newCopy(N,A); // Copy of Original A
  		int *less_t = (int *)malloc (N * sizeof (int)); // Array of 1s for values less than pivot
  		int *greater_t = (int *)malloc (N * sizeof (int)); // Array of 1s for values greater than pivot
 
  		/* populate less_t[] and greater_t[] with values < or >= pivot values with 1s and 0s */
  		// spawn num_threads 
  		#pragma omp parallel for schedule(static)
  		for(i=0;i<N;i++) {
  			if(A_orig[i] < pivot) {
  				less_t[i] = 1;
  				greater_t[i] = 0;
  			}
  			else if(A_orig[i] > pivot) {
  				less_t[i] = 0;
   				greater_t[i] = 1;
   			} 
   			else { 
   				count++; 
   				less_t[i] = 0; 
   				greater_t[i] = 0;
   			}
  		} //end parallel for

  		int *less_t_prefix = (int *)malloc (N * sizeof (int)); // prefix sum of lt[N]
  		int *greater_t_prefix = (int *)malloc (N * sizeof (int)); // prefix sum for gt[N]


  		prefix_scan(less_t,N,less_t_prefix);
  		prefix_scan(greater_t,N,greater_t_prefix);

  		*lower = less_t_prefix[N-1]; //this is the index corresponding to size of less_than pivot partiton 
  		*upper = less_t_prefix[N-1] + count; // this is the index corresponding to size of greater_than pivot partiton
  
  		#pragma omp parallel for
  		for(i = less_t_prefix[N-1]; i < (less_t_prefix[N-1] + count); i++) { A[i] = pivot; } // store for repeating no. of pivot values

  		#pragma omp parallel for schedule(static)
  		for(i = 0; i < N; i++) {
  			if(less_t[i] == 1) {
  				A[less_t_prefix[i] - 1] = A_orig[i];
  			}
  			else if(greater_t[i] == 1) {
  				A[(less_t_prefix[N-1] + count) + greater_t_prefix[i] - 1] = A_orig[i];
  			}
  		}
  		
  		/* free up all the local memory spaces*/
  		free(A_orig);
   		free(less_t);
   		free(greater_t);
   		free(less_t_prefix);
   		free(greater_t_prefix);
   	}
}

void quickSort(int N, keytype* A) {
	const int G = 100; /* base case size, a tuning parameter */
	if(N < G) {
		sequentialSort(N, A);
	}
	else {
		/* choose any random index for the pivot value*/
		int k = rand()%N;
		int lower = 0, upper = 0;

		partition(N, A, k, &lower, &upper);
		#pragma omp task 
		{
			if(lower >= 1) {
				quickSort(lower, A); //recursively solve for the lower parition
			}
			if(N - upper >= 1) {
				quickSort(N - upper, A + upper); //recursively solve for the upper parition
			}
		}
	}
}

void parallelSort(int N, keytype* A) {
	#pragma omp parallel
	{
		/** 
		 * Let just one thread call the quickSort function.. 
		 * this thread will then spawn threads within the function call 
		 */
		#pragma omp single
		quickSort(N, A);
	}
}
