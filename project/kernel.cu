// cuda kernel ====================================================
const unsigned int nAssets=2; //if at runtime, difficult to handle memory for each thread

#include <curand_kernel.h>

__host__ __device__ int indexFlat(int iSimu, int iAsset, int nSimu){
  return iAsset*nSimu+iSimu; //neighboring threads should be neighbors in memory
}

__device__ float priceModel(float val, int i){
	return val+(val/100.)*sin(val + (float) i);
}

__global__ void simPrice(float* prices, float* initialPrices, const int nSimu, const int nSteps) {

    int iSimu = blockDim.x * blockIdx.x + threadIdx.x;
    int iStep, iAsset;
    
    if (iSimu<nSimu){
    	float prices_t[nAssets];
    
    	//initial prices - privatized
    	for (iAsset=0; iAsset<nAssets; iAsset++){
    		prices_t[iAsset] = initialPrices[iAsset];
    	}
    	
    	//time evolution
    	for (iStep=0; iStep<nSteps; iStep++){
    		for (iAsset=0; iAsset<nAssets; iAsset++){
    			prices_t[iAsset] = priceModel(prices_t[iAsset], iSimu);
    		}
    	}
    	
    	//store result
    	for (iAsset=0; iAsset<nAssets; iAsset++){
    		prices[indexFlat(iSimu, iAsset,nSimu)] = prices_t[iAsset];
    	}
    	
    }//iSimu

}
