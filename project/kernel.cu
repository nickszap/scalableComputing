#include <curand_kernel.h>

const unsigned int nAssets=2; //unclear how to set at runtime and allocate memory

/* 
From curand API info (http://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes):
State setup can be an expensive operation. One way to speed up the setup is to use different seeds for each thread and a constant sequence number of 0. This can be especially helpful if many generators need to be created. While faster to set up, this method provides less guarantees about the mathematical properties of the generated sequences. If there happens to be a bad interaction between the hash function that initializes the generator state from the seed and the periodicity of the generators, there might be threads with highly correlated outputs for some seed values. We do not know of any problem values; if they do exist they are likely to be rare.
*/

__host__ __device__ 
int indexFlat(int iSimu, int iAsset, int nSimu){
  return iAsset*nSimu+iSimu; //neighboring threads should be neighbors in memory
}

__device__ 
float priceModel_fake(float val, int i){
	return val+(val/100.)*sin(val + (float) i);
}

__device__ 
float priceModel_meanReversion(float val, float meanVal, float revRate, float stdev, curandState *stateRand){
	// dx = k*dx_mean + sigma*dz
	float vol = curand_normal(stateRand)*stdev; //volatility
	return val+revRate*(meanVal-val)+vol;
}

__device__ 
float priceModel_jump(float val, int nNews, float *lambdaNews, float *stdevNews, curandState *stateRand){
	// sum of jump perturbations (e.g., news events) modelled as poisson processes.
	//data-dependent thread divergence as threads compute different numbers of events.
	
	float dVal = 0.0;
	for (int iNews=0; iNews<nNews; iNews++){
		//int nNewsEvents = curand_poisson(stateRand, (double) lambdaNews[iNews]);
		int nNewsEvents = round( lambdaNews[iNews]/curand_uniform(stateRand) ) ; //curand_poisson is undefined????
		for (int iEvent=0; iEvent<nNewsEvents; iEvent++){
			//dVal += curand_normal(stateRand)*stdevNews[iNews];
			dVal += curand_normal(stateRand)*stdevNews[iNews]*val/100.;
		}
	}
	
	return val+dVal;
}

__global__ 
void simPrice_jump(float* prices, float* initialPrices, const int nSimu, const int nSteps) {
		
		const int nNews = 3;
		float lambdaNews[nNews] = {.1,.05, .005}; //mean # news events per timestep
		float stdevNews[nNews] = {1., 2., 5.}; //useful either as dollars or percent of current price
		
    int iSimu = blockDim.x * blockIdx.x + threadIdx.x;
    int iStep, iAsset;
    
    // seed the random number generator
    curandState stateRand;
    curand_init(iSimu, 0, 0, &stateRand); //float x = curand_normal(&state)

    if (iSimu<nSimu){
    	float prices_t[nAssets];
    
    	//initial prices - privatized
    	for (iAsset=0; iAsset<nAssets; iAsset++){
    		prices_t[iAsset] = initialPrices[iAsset];
    	}
    	
    	//time evolution
    	for (iStep=0; iStep<nSteps; iStep++){
    		for (iAsset=0; iAsset<nAssets; iAsset++){
    			prices_t[iAsset] = priceModel_jump(prices_t[iAsset], nNews, lambdaNews, stdevNews, &stateRand);
    		}
    	}
    	
    	//store result
    	for (iAsset=0; iAsset<nAssets; iAsset++){
    		prices[indexFlat(iSimu, iAsset,nSimu)] = prices_t[iAsset];
    	}
    	
    }//iSimu
}

__global__ 
void simPrice_mr(float* prices, float* initialPrices, const int nSimu, const int nSteps) {
		
		float meanVal[nAssets] = {7., 25.};
		float revRate[nAssets] = {.01, .5};
		float stdev[nAssets] = {.5, .01};
		
    int iSimu = blockDim.x * blockIdx.x + threadIdx.x;
    int iStep, iAsset;
    
    // seed the random number generator
    curandState stateRand;
    curand_init(iSimu, 0, 0, &stateRand); //float x = curand_normal(&state)

    if (iSimu<nSimu){
    	float prices_t[nAssets];
    
    	//initial prices - privatized
    	for (iAsset=0; iAsset<nAssets; iAsset++){
    		prices_t[iAsset] = initialPrices[iAsset];
    	}
    	
    	//time evolution
    	for (iStep=0; iStep<nSteps; iStep++){
    		for (iAsset=0; iAsset<nAssets; iAsset++){
    			//prices_t[iAsset] = priceModel(prices_t[iAsset], iSimu);
    			prices_t[iAsset] = priceModel_meanReversion(prices_t[iAsset], meanVal[iAsset], revRate[iAsset], stdev[iAsset], &stateRand);
    		}
    	}
    	
    	//store result
    	for (iAsset=0; iAsset<nAssets; iAsset++){
    		prices[indexFlat(iSimu, iAsset,nSimu)] = prices_t[iAsset];
    	}
    	
    }//iSimu
}

__host__
void launch_kernel(float* prices_d, float* initialPrices_d, int nSimu, int nSteps){
		//we need additional logic based on the chosen pricing model
		//e.g., cudarand API "robust device" poisson distribution requires function calls on host
		
		const unsigned int THREADS_PER_BLOCK = 64;
    const unsigned int numBlocks = (nSimu - 1)/THREADS_PER_BLOCK + 1;
    dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);
    
    if (0){
    	//mean reversion
    	printf("Running mean reversion  model\n");
    	simPrice_mr <<< gridDim, blockDim >>> (prices_d, initialPrices_d, nSimu,nSteps);
    }
    else if (1){
    	//jump processes
    	printf("Running jump price model...");
    	simPrice_jump <<< gridDim, blockDim >>> (prices_d, initialPrices_d, nSimu,nSteps);
    }
    else {
    	printf("\n Uhoh. No pricing model selected. Nothing to do here.\n");
   	}
}
