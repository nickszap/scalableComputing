#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

__host__ __device__ float Clamp(float val, float start, float end){
  if (val>end) return end;
  if (val<start) return start;
  return val;
}

__host__ __device__ int Index3D(int width, int depth, int i, int j, int k){
	//i=y, j=x, k=z???
  return (i*width+j)*depth+k;
}

__host__ __device__ void stencil_cpu(float *_out, float *_in, int width, int height, int depth) {

#define out(i, j, k) _out[(( i )*width + (j)) * depth + (k)]
#define in(i, j, k) _in[(( i )*width + (j)) * depth + (k)]

  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      for (int k = 1; k < depth - 1; ++k) {
        float val = in(i, j, k + 1) + in(i, j, k - 1) + in(i, j + 1, k) +
                       in(i, j - 1, k) + in(i + 1, j, k) + in(i - 1, j, k) -
                       6 * in(i, j, k);
		out(i, j, k) = Clamp(val, 0., 255.);
      }
    }
  }
#undef out
#undef in
}

__host__ __device__ void stencil_index(float *_out, float *_in, int width, int height, int depth) {

#define out(i, j, k) _out[Index3D(width, depth, i, j, k)]
#define in(i, j, k) _in[Index3D(width, depth, i, j, k)]

  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      for (int k = 1; k < depth - 1; ++k) {
        float val = in(i, j, k + 1) + in(i, j, k - 1) + in(i, j + 1, k) +
                       in(i, j - 1, k) + in(i + 1, j, k) + in(i - 1, j, k) -
                       6 * in(i, j, k);
		out(i, j, k) = Clamp(val, 0., 255.);
      }
    }
  }
#undef out
#undef in
}

const unsigned int TILE_SIZE = 32;
const unsigned int BLOCK_SIZE = TILE_SIZE;

__global__ void stencil(float *Anext, float *Aorig, int width, int height,
                        int depth) {
  //@@ INSERT CODE HERE
	__shared__ float ds_A[TILE_SIZE][TILE_SIZE];
	
	//for indexing,
	int tx = threadIdx.x; int Nx = width;
	int ty = threadIdx.y; int Ny = height; int Nz = depth;
	int iGlobalx = blockIdx.x*blockDim.x+tx;
	int iGlobaly = blockIdx.y*blockDim.y+ty;
	
	//tricky to avoid out of bounds array accesses and have syncthreads() called by all w/in block
	float bottom = ((iGlobalx<Nx && iGlobaly<Ny)? Aorig[Index3D(Nx, Nz, iGlobaly, iGlobalx, 0)]:0);
	float current = ((iGlobalx<Nx && iGlobaly<Ny)? Aorig[Index3D(Nx, Nz, iGlobaly, iGlobalx, 1)]:0);
	ds_A[tx][ty] = current;
	__syncthreads(); //so ds_A is updated. all threads in block march up together
	
	for (int k=1; k < Nz-1; k++) {
		//either read from shared tile or global memory
		
		float top = ((iGlobalx<Nx && iGlobaly<Ny)? Aorig[Index3D(Nx, Nz, iGlobaly, iGlobalx, k+1)]:0);
		
		if (iGlobalx<Nx-1 && iGlobalx>0 && iGlobaly<Ny-1 && iGlobaly>0){
			//center column, left, right, S, N
			float val = bottom + top - 6.0*current +
				((tx > 0)? ds_A[tx-1][ty]: Aorig[Index3D(Nx, Nz, iGlobaly, iGlobalx-1, k)]) +
				((tx < TILE_SIZE-1)? ds_A[tx+1][ty]: Aorig[Index3D(Nx, Nz, iGlobaly, iGlobalx+1, k)]) +
				((ty > 0)? ds_A[tx][ty-1]: Aorig[Index3D(Nx, Nz, iGlobaly-1, iGlobalx, k)]) +
				((ty < TILE_SIZE-1)? ds_A[tx][ty+1]: Aorig[Index3D(Nx, Nz, iGlobaly+1, iGlobalx, k)]);
			Anext[Index3D(Nx,Nz,iGlobaly, iGlobalx,k)] = Clamp(val, 0.0, 255.0);
		}
		
		//update tile for next k up
		__syncthreads(); //Anext is finished before ds_A is changed.
		ds_A[tx][ty] = top;
		__syncthreads(); //ds_A is updated for next level...not needed on last loop though
		bottom = current;
		current = top;
	}
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData,
                           int width, int height, int depth) {
  //@@ INSERT CODE HERE
	
  dim3 gDim((width-1)/BLOCK_SIZE + 1, (height-1)/BLOCK_SIZE + 1, 1);
  dim3 bDim(BLOCK_SIZE, BLOCK_SIZE, 1);

  // Invoke CUDA kernel -----------------------------------------------------
  stencil <<< gDim, bDim >>> (deviceOutputData, deviceInputData, width, height, depth);
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int width;
  int height;
  int depth;
  char *inputFile;
  wbImage_t input;
  wbImage_t output;
  float *hostInputData;
  float *hostOutputData;
  float *deviceInputData;
  float *deviceOutputData;

  arg = wbArg_read(argc, argv);

  inputFile = wbArg_getInputFile(arg, 0);

  input = wbImport(inputFile);

  width = wbImage_getWidth(input);
  height = wbImage_getHeight(input);
  depth = wbImage_getChannels(input);

  output = wbImage_new(width, height, depth);

  hostInputData = wbImage_getData(input);
  hostOutputData = wbImage_getData(output);

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc(( void ** )&deviceInputData,
             width * height * depth * sizeof(float));
  cudaMalloc(( void ** )&deviceOutputData,
             width * height * depth * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputData, hostInputData,
             width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
  cudaDeviceSynchronize(); //is this needed?
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputData, deviceOutputData,
             width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

#if 0	
  int idx = Index3D(width, depth, 120, 120, 4);
  wbLog(TRACE, "Dimensions of input = ", width, "_",  height, "_", depth);
  wbLog(TRACE, "Index = ", idx);	
  wbLog(TRACE, "Input = ", hostInputData[idx]);
  wbLog(TRACE, "Output = ", hostOutputData[idx]);
  //stencil_cpu(hostOutputData, hostInputData, width, height, depth);
  //stencil_index(hostOutputData, hostInputData, width, height, depth); 
  //wbLog(TRACE, "Output = ", hostOutputData[idx]);
#endif
  wbSolution(arg, output);

  cudaFree(deviceInputData);
  cudaFree(deviceOutputData);

  wbImage_delete(output);
  wbImage_delete(input);

  return 0;
}
