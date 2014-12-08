/*------------------------------------------------------

         Template taken from vecadd

--------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
//#include <random>

const int nAssets=2; //unclear how to set at runtime and allocate memory

int indexFlat(int iSimu, int iAsset, int nSimu){
  return iAsset*nSimu+iSimu; //neighboring threads should be neighbors in memory
}

//price evolution -------------------------------------
float genUniform(){
  //uniform random number in [0,1]
  float val = (float) rand();
  val /= RAND_MAX;
  return val;
}

float priceModel_meanReversion(float val, float meanVal, float revRate, float stdev){
	// dx = k*dx_mean + sigma*dz
	float vol = genUniform(); vol = 2.0*vol-1.0; //(-1,1)
	vol*=stdev; //volatility
	return val+revRate*(meanVal-val)+vol;
}

void simPrice_mr(float* prices, float* initialPrices, const int nSimu, const int nSteps) {
		
		float meanVal[nAssets]; meanVal[0] = 7.; meanVal[1]= 25.;
		float revRate[nAssets]; revRate[0] = .01; revRate[1] = .5;
		float stdev[nAssets]; stdev[0]=.5; stdev[1]=.01;
    
    // seed the random number generator
    srand(0);
    
    int iSimu;
    for (iSimu=0; iSimu<nSimu; iSimu++){
      int iStep, iAsset;
    
    	//initial prices
    	for (iAsset=0; iAsset<nAssets; iAsset++){
    		prices[indexFlat(iSimu, iAsset,nSimu)] = initialPrices[iAsset];
    	}
    	
    	//time evolution
    	for (iStep=0; iStep<nSteps; iStep++){
    		for (iAsset=0; iAsset<nAssets; iAsset++){
    		  int iPrice = indexFlat(iSimu, iAsset,nSimu);
    			prices[iPrice] = priceModel_meanReversion(prices[iPrice], meanVal[iAsset], revRate[iAsset], stdev[iAsset]);
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
    nSimu = 10000;
    nSteps = 1000;
    
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
