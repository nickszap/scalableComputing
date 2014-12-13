/*------------------------------------------------------

         Template taken from vecadd

--------------------------------------------------------*/

#include <stdio.h>
#include <sys/time.h>
#include "kernel.cu"
//#include "kernel_opt.cu"

// Declare error and timing utilities =========================================

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)


typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

// Main function ==============================================================
int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize GPU/CUDA ----------------------------------------------------

    printf("\nInitializing GPU/CUDA..."); fflush(stdout);
    startTime(&timer);
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // Input parameters and host variables -------------------------

    printf("Setting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int nSimu, nSteps;
    nSimu = 100000;
    nSteps = 1000;
    
    unsigned int nTotal = nAssets*nSimu;
    float* prices_h = (float*) malloc( sizeof(float)*nTotal );
    
    float initialPrices_h[nAssets];
    int i;
    for (i=0; i<nAssets; i++){
    	initialPrices_h[i] = (i+1)*2.0;
    }
    	

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("  Case = %u assets for %u steps with %u simulations \n", nAssets, nSteps, nSimu);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    float* prices_d;
    cuda_ret = cudaMalloc((void**) &prices_d, sizeof(float)*nTotal);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    
    float* initialPrices_d;
    cuda_ret = cudaMalloc((void**) &initialPrices_d, sizeof(float)*nAssets);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(initialPrices_d, initialPrices_h, sizeof(float)*nAssets, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    
    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    launch_kernel(prices_d, initialPrices_d, nSimu,nSteps);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(prices_h, prices_d, sizeof(float)*nTotal, cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n\n", elapsedTime(timer));
    
    // print some output ----------------------------------------
#if 0
    int iAsset, iSimu, iPrice;
    for (iAsset=0; iAsset<nAssets; iAsset++){
    	printf("\nAsset %d price0 %g\n",iAsset, initialPrices_h[iAsset]);
    	for (iSimu=0; iSimu<nSimu; iSimu++){
    		iPrice = indexFlat(iSimu, iAsset,nSimu);
    		printf(" %g ",prices_h[iPrice]);
    	}
    }
    printf("\n");
#endif

    // Free memory ------------------------------------------------------------

    free(prices_h);

    cudaFree(prices_d);

    return 0;

}

// Define timing utilities ====================================================

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}
