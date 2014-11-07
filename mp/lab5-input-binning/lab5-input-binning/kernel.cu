/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// For simplicity, fix #bins=1024 so scan can use a single block and no padding
#define NUM_BINS 1024

/******************************************************************************
 GPU main computation kernels
*******************************************************************************/

__global__ void gpu_normal_kernel(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in) {

    // INSERT KERNEL CODE HERE
    /* instructions:
    Edit the kernel gpu normal kernel in the ﬁle kernel.cu to imple-
ment the same computation as cpu normal on the GPU. Note however that
cpu normal performs the computation using a scatter pattern. For the kernel
gpu normal kernel, you are required to use a gather pattern which is more efficient.
		*/
    int outIdx;
    float in_val2, dist2;
    float sum=0.0;
    
    outIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (outIdx<grid_size){ //don't exceed bounds of output array
      for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
          in_val2 = in_val[inIdx]*in_val[inIdx];
          dist2 = in_pos[inIdx]- (float) outIdx; dist2 = dist2*dist2;
          sum += in_val2/dist2;
      }
      out[outIdx] = sum;
    }

}

__global__ void gpu_cutoff_kernel(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in, float cutoff2) {

    // INSERT KERNEL CODE HERE
    /*
    Edit the kernel gpu cutoff kernel in the ﬁle kernel.cu to imple-
ment the computation on the GPU, this time only considering input values that
fall within a cutoff range of the output grid point.
    */
    
    int outIdx;
    float in_val2, dist2;
    float sum=0.0;
    
    outIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (outIdx<grid_size){ //don't exceed bounds of output array
      for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
          in_val2 = in_val[inIdx]*in_val[inIdx];
          dist2 = in_pos[inIdx]- (float) outIdx; dist2 = dist2*dist2;
          if (dist2<=cutoff2){
  	      	sum += in_val2/dist2;
  	      }
      }
      out[outIdx] = sum;
    }

}

__global__ void gpu_cutoff_binned_kernel(unsigned int* binPtrs,
    float* in_val_sorted, float* in_pos_sorted, float* out,
    unsigned int grid_size, float cutoff2) {

    // INSERT KERNEL CODE HERE
    /*
    Edit the kernel gpu cutoff binned kernel in the ﬁle kernel.cu to
implement the computation on the GPU. In this version, the input has been
sorted into bins for you. You must loop over the bins and for each bin check if
either of its bounds is within the cutoff range. If yes, you must loop over the
input elements in the bin, check if each element is within the cutoff range, and
if yes include it in your computation.

		From the cpu_preprocess(), it looks like binPtrs[i] is the left index of bin i.
		The provided index/count directly accesses the sorted arrays.
    */
    
    int outIdx, iBinL, iBinR;
    float in_val2, dist2, dist2L, dist2R;
    float sum=0.0;
    
    outIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (outIdx<grid_size){ //don't exceed bounds of output array
    	for (unsigned int iBin=0; iBin<NUM_BINS; iBin++){
    		iBinL = binPtrs[iBin]; iBinR = binPtrs[iBin+1];
    		dist2L = in_pos_sorted[iBinL]- (float) outIdx; dist2L = dist2L*dist2L;
    		dist2R = in_pos_sorted[iBinR]- (float) outIdx; dist2R = dist2R*dist2R;
    		if (dist2L > cutoff2 && dist2R>cutoff2){ //bin outside cutoff
    			continue;
    		}
      	for(unsigned int inIdx = iBinL; inIdx < iBinR; ++inIdx) {
      		//do < iBinR to avoid double counting borders
        	in_val2 = in_val_sorted[inIdx]*in_val_sorted[inIdx];
          dist2 = in_pos_sorted[inIdx]- (float) outIdx; dist2 = dist2*dist2;
          if (dist2<=cutoff2){
  	      	sum += in_val2/dist2;
  	      }
      	} //inIdx
      } //iBin
      out[outIdx] = sum;
    } //outIdx

}

/******************************************************************************
 Main computation functions
*******************************************************************************/

void cpu_normal(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in) {

    for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
        const float in_val2 = in_val[inIdx]*in_val[inIdx];
        for(unsigned int outIdx = 0; outIdx < grid_size; ++outIdx) {
            const float dist = in_pos[inIdx] - (float) outIdx;
            out[outIdx] += in_val2/(dist*dist);
        }
    }

}

void gpu_normal(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in) {

    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (grid_size - 1)/numThreadsPerBlock + 1;
    gpu_normal_kernel <<< numBlocks , numThreadsPerBlock >>>
        (in_val, in_pos, out, grid_size, num_in);

}

void gpu_cutoff(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in, float cutoff2) {

    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (grid_size - 1)/numThreadsPerBlock + 1;
    gpu_cutoff_kernel <<< numBlocks , numThreadsPerBlock >>>
        (in_val, in_pos, out, grid_size, num_in, cutoff2);

}

void gpu_cutoff_binned(unsigned int* binPtrs, float* in_val_sorted,
    float* in_pos_sorted, float* out, unsigned int grid_size, float cutoff2) {

    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (grid_size - 1)/numThreadsPerBlock + 1;
    gpu_cutoff_binned_kernel <<< numBlocks , numThreadsPerBlock >>>
        (binPtrs, in_val_sorted, in_pos_sorted, out, grid_size, cutoff2);

}


/******************************************************************************
 Preprocessing kernels
*******************************************************************************/

__global__ void histogram(float* in_pos, unsigned int* binCounts,
    unsigned int num_in, unsigned int grid_size) {

    // INSERT KERNEL CODE HERE










}

__global__ void scan(unsigned int* binCounts, unsigned int* binPtrs) {

    // INSERT KERNEL CODE HERE























}

__global__ void sort(float* in_val, float* in_pos, float* in_val_sorted,
    float* in_pos_sorted, unsigned int grid_size, unsigned int num_in,
    unsigned int* binCounts, unsigned int* binPtrs) {

    // INSERT KERNEL CODE HERE













}

/******************************************************************************
 Preprocessing functions
*******************************************************************************/

void cpu_preprocess(float* in_val, float* in_pos, float* in_val_sorted,
    float* in_pos_sorted, unsigned int grid_size, unsigned int num_in,
    unsigned int* binCounts, unsigned int* binPtrs) {

    // Histogram the input positions
    for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
        const unsigned int binIdx =
            (unsigned int) ((in_pos[inIdx]/grid_size)*NUM_BINS);
        ++binCounts[binIdx];
    }

    // Scan the histogram to get the bin pointers
    binPtrs[0] = 0;
    for(unsigned int binIdx = 0; binIdx < NUM_BINS; ++binIdx) {
        binPtrs[binIdx + 1] = binPtrs[binIdx] + binCounts[binIdx];
    }

    // Sort the inputs into the bins
    for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
        const unsigned int binIdx =
            (unsigned int) ((in_pos[inIdx]/grid_size)*NUM_BINS);
        const unsigned int newIdx = binPtrs[binIdx + 1] - binCounts[binIdx];
        --binCounts[binIdx];
        in_val_sorted[newIdx] = in_val[inIdx];
        in_pos_sorted[newIdx] = in_pos[inIdx];
    }

}

void gpu_preprocess(float* in_val, float* in_pos, float* in_val_sorted,
    float* in_pos_sorted, unsigned int grid_size, unsigned int num_in,
    unsigned int* binCounts, unsigned int* binPtrs) {

    const unsigned int numThreadsPerBlock = 512;

    // Histogram the input positions
    histogram <<< 30 , numThreadsPerBlock >>>
        (in_pos, binCounts, num_in, grid_size);

    // Scan the histogram to get the bin pointers
    if(NUM_BINS != 1024) FATAL("NUM_BINS must be 1024. Do not change.");
    scan <<< 1 , numThreadsPerBlock >>> (binCounts, binPtrs);

    // Sort the inputs into the bins
    sort <<< 30 , numThreadsPerBlock >>> (in_val, in_pos, in_val_sorted,
        in_pos_sorted, grid_size, num_in, binCounts, binPtrs);

}


