/*------------------------------------------------------

         Template taken from vecadd

--------------------------------------------------------*/

#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <iostream>

const unsigned int nAssets=2; //unclear how to set at runtime and allocate memory
int indexFlat(int iSimu, int iAsset, int nSimu){
  return iAsset*nSimu+iSimu; //neighboring threads should be neighbors in memory
}

//price evolution -------------------------------------
float priceModel_meanReversion(float val, float meanVal, float revRate, float stdev, std::normal_distribution<float> distribution, std::default_random_engine generator){
	// dx = k*dx_mean + sigma*dz
	float vol = distribution(generator); vol*=stdev; //volatility
	return val+revRate*(meanVal-val)+vol;
}

void simPrice_mr(float* prices, float* initialPrices, const int nSimu, const int nSteps) {
		
		float meanVal[nAssets] = {7., 25.};
		float revRate[nAssets] = {.01, .5};
		float stdev[nAssets] = {.5, .01};
    
    // seed the random number generator
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0,1.0); //val=distribution(generator)

    for (int iSimu=0; iSimu<nSimu; iSimu++){
      int iStep, iAsset;
    	float prices_t[nAssets];
    
    	//initial prices
    	for (iAsset=0; iAsset<nAssets; iAsset++){
    		prices[indexFlat(iSimu, iAsset,nSimu)] = initialPrices[iAsset];
    	}
    	
    	//time evolution
    	for (iStep=0; iStep<nSteps; iStep++){
    		for (iAsset=0; iAsset<nAssets; iAsset++){
    			//prices_t[iAsset] = priceModel(prices_t[iAsset], iSimu);
    			prices[indexFlat(iSimu, iAsset,nSimu)] = priceModel_meanReversion(prices[indexFlat(iSimu, iAsset,nSimu)], meanVal[iAsset], revRate[iAsset], stdev[iAsset], distribution, generator);
    		}
    	}
    	
    }//iSimu
}

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

    // Input parameters and host variables -------------------------

    printf("Setting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int nSimu, nSteps;
    nSimu = 200;
    nSteps = 100;
    
    unsigned int nTotal = nAssets*nSimu;
    float* prices_h = (float*) malloc( sizeof(float)*nTotal );
    
    float initialPrices_h[nAssets];
    int i;
    for (i=0; i<nAssets; i++){
    	initialPrices_h[i] = (i+1)*10.0;
    }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("  Case = %u assets for %u steps with %u simulations \n", nAssets, nSteps, nSimu);
    
    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    
    simPrice_mr(prices_h, initialPrices_h, nSimu, nSteps);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    
    // print some output ----------------------------------------
    int iAsset, iSimu, iPrice;
    for (iAsset=0; iAsset<nAssets; iAsset++){
    	printf("\nAsset %d price0 %g\n",iAsset, initialPrices_h[iAsset]);
    	for (iSimu=0; iSimu<nSimu; iSimu++){
    		iPrice = indexFlat(iSimu, iAsset,nSimu);
    		printf(" %g ",prices_h[iPrice]);
    	}
    }
    printf("\n");
    
    // Free memory ------------------------------------------------------------

    free(prices_h);

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
