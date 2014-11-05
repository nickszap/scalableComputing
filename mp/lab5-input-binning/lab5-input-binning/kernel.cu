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












}

__global__ void gpu_cutoff_kernel(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in, float cutoff2) {

    // INSERT KERNEL CODE HERE












}

__global__ void gpu_cutoff_binned_kernel(unsigned int* binPtrs,
    float* in_val_sorted, float* in_pos_sorted, float* out,
    unsigned int grid_size, float cutoff2) {

    // INSERT KERNEL CODE HERE


















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


