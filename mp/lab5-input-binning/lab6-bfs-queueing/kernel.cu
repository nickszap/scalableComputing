/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE/WARP_SIZE)

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY 128

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_global_queuing_kernel(unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *currLevelNodes, unsigned int *nextLevelNodes,
  unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  
  // Loop over all nodes in the curent level. 
  //since can't assume we have more threads than numCurrLevelNodes,
  int nThreads=blockDim.x*gridDim.x; 
  int nSteps = (*numCurrLevelNodes-1)/nThreads+1;
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int iStep=0; iStep<nSteps; iStep++){
    if (idx<*numCurrLevelNodes){
    	unsigned int node = currLevelNodes[idx];
    	for(unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx){
    		unsigned int neighbor = nodeNeighbors[nbrIdx];
    		//See if neighbor has been visited yet. If not, add it to the queue.
    		//This needs to be atomic op to avoid race conditions between the read and modify.
    		int isVisited = atomicAdd(&nodeVisited[neighbor],1); //atomicAdd returns old value
        if(!isVisited) {
          // Already marked, add it to the queue
          int iNextLevel = atomicAdd(numNextLevelNodes,1);
          nextLevelNodes[iNextLevel] = neighbor;
        }
    	}
    }
    idx += nThreads;
  }

}

__global__ void gpu_block_queuing_kernel(unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *currLevelNodes, unsigned int *nextLevelNodes,
  unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  
  // Loop over all nodes in the curent level.
  __shared__ int nextLevelNodes_s[BQ_CAPACITY];
  __shared__ int numNextLevelNodes_s; //can't initialize shared vars
  if (threadIdx.x==0){ //dunno if it matters if all threads modify
  	numNextLevelNodes_s=0;
  }
  __syncthreads();
  
  int nThreads=blockDim.x*gridDim.x; 
  int nSteps = (*numCurrLevelNodes-1)/nThreads+1;
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int iStep=0; iStep<nSteps; iStep++){
    if (idx<*numCurrLevelNodes){
    	unsigned int node = currLevelNodes[idx];
    	for(unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx){
    		unsigned int neighbor = nodeNeighbors[nbrIdx];
    		//See if neighbor has been visited yet. If not, add it to the queue.
    		//This needs to be atomic op to avoid race conditions between the read and modify.
    		int isVisited = atomicAdd(&nodeVisited[neighbor],1); //atomicAdd returns old value
        if(!isVisited) {
          // Already marked it, try to add to the shared queue but
          //have to deal w/ overflow
          int iNextLevel = atomicAdd(&numNextLevelNodes_s,1); //candidate index into shared array
          if (iNextLevel>=BQ_CAPACITY){ //no room in block's shared space
            //add to global queue
            iNextLevel = atomicAdd(numNextLevelNodes,1);
            nextLevelNodes[iNextLevel] = neighbor;
          }
          else { //room in block's shared space
            //add to block's queue
            nextLevelNodes_s[iNextLevel] = neighbor;
          }
        }
    	}
    }
    idx += nThreads;
  }
  //now insert block's queue into global queue
  __syncthreads();
  //reserve block's space in global queue
  __shared__ int iStartGlobal;
  if (threadIdx.x==0){
    numNextLevelNodes_s = MIN(numNextLevelNodes_s, BQ_CAPACITY);
  	iStartGlobal = atomicAdd(numNextLevelNodes,numNextLevelNodes_s);
  }
  __syncthreads();
  //fill in global queue collaboratively
  nSteps = (numNextLevelNodes_s-1)/blockDim.x+1;
  for (unsigned int iStep=0; iStep<nSteps; iStep++){
  	idx = iStep*blockDim.x+threadIdx.x;
  	if (idx<numNextLevelNodes_s){
  		nextLevelNodes[idx+iStartGlobal]=nextLevelNodes_s[idx];
  	}
  }
}

__global__ void gpu_warp_queuing_kernel(unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *currLevelNodes, unsigned int *nextLevelNodes,
  unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE

}

/******************************************************************************
 Functions
*******************************************************************************/

void cpu_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  // Loop over all nodes in the curent level
  for(unsigned int idx = 0; idx < *numCurrLevelNodes; ++idx) {
    unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for(unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
      ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      if(!nodeVisited[neighbor]) {
        // Mark it and add it to the queue
        nodeVisited[neighbor] = 1;
        nextLevelNodes[*numNextLevelNodes] = neighbor;
        ++(*numNextLevelNodes);
      }
    }
  }

}

void gpu_global_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel <<< numBlocks , BLOCK_SIZE >>> (nodePtrs,
    nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
    numCurrLevelNodes, numNextLevelNodes);

}

void gpu_block_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel <<< numBlocks , BLOCK_SIZE >>> (nodePtrs,
    nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
    numCurrLevelNodes, numNextLevelNodes);

}

void gpu_warp_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queuing_kernel <<< numBlocks , BLOCK_SIZE >>> (nodePtrs,
    nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
    numCurrLevelNodes, numNextLevelNodes);

}

