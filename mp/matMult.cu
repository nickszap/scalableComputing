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

__host__ __device__ int index_A(int iRow, int iCol, int nRow, int nCol){
	//column major format
	return iCol*nRow+iRow;
}

__host__ __device__ int index_B(int iRow, int iCol, int nRow, int nCol){
	//row major format
	return iRow*nCol+iCol;
}

void matMult_cpu(float *A, float *B, float *C, int numARows,
				 int numAColumns, int numBRows, int numBColumns,
				 int numCRows, int numCColumns){
	
	int iRowA, iColB, k;
	int indA, indB, indC;
	float sum;
	for(iRowA=0;iRowA<numARows;iRowA++){
		for(iColB=0;iColB<numBColumns;iColB++){
			sum = 0.0;
			for(k=0;k<numBRows;k++){
				indA = index_A(iRowA,k,numARows,numAColumns);
				indB = index_B(k,iColB,numBRows,numBColumns);
				sum += A[indA]*B[indB];
			}
			indC = index_B(iRowA,iColB,numCRows,numCColumns);
			C[indC] = sum;
		}
	}
}

#define TILE_SIZE_A 32
#define TILE_SIZE_B 8
//must have size_a>=size_b

// Compute C = A * B
__global__ void matrixMultiply_kernel(float *A, float *B, float *C,
                                      int numARows, int numAColumns,
                                      int numBRows, int numBColumns,
                                      int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to perform register tiling for this MP
	//per (revised) instructions, need to use:
	//shared memory, thread coarsening, and register tiling optimization techniques
	__shared__ float B_s[TILE_SIZE_B];
	float C_l[TILE_SIZE_B] = {0.0};
	
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int iRow = by * TILE_SIZE_A + ty;
	int iCol = bx * TILE_SIZE_B + tx;
	
	for(int k=0;k<numBRows;k++){
		//load chunk of row of B, in shared fashion
		if (iRow<numARows && iCol<numBColumns){
			if (tx<TILE_SIZE_B){ 
				//this approach satisfies the instructions but results in control divergence.
				//the marginal knowledge gain from using k steps didn't justify the headache of
				//figuring out the indexing.
				int indB = index_B(k,iCol,numBRows,numBColumns);
				B_s[tx] = B[indB];
			}
		}
		__syncthreads();
		
		//perform kth step of matMult for C_l[size_a,size_b]
		if (iRow<numARows && iCol<numBColumns){
			int indA = index_A(iRow,k,numARows,numAColumns);
			float Aval = A[indA];
			for (int iB=0; iB<TILE_SIZE_B; iB++){
				C_l[iB] += Aval*B_s[iB];
			}
		}
		__syncthreads();
	}
	//update global C
	if (iRow<numARows && iCol<numBColumns){
		for (int iB=0; iB<TILE_SIZE_B; iB++){
			iCol = bx * TILE_SIZE_B + iB;
			if (iCol>=numBColumns){//avoid stepping outside C's bounds
				continue;
			}
			int indC = index_B(iRow,iCol,numCRows,numCColumns);
			C[indC] = C_l[iB];
		}
	}
}

static void matrixMultiply(float *A, float *B, float *C, int numARows,
                           int numAColumns, int numBRows, int numBColumns,
                           int numCRows, int numCColumns) {
  //@@ Insert code to launch matrix multiplication
	
  //for output C, block_size rows and tile_width cols
  dim3 gDim((numBColumns-1)/TILE_SIZE_B + 1, (numARows-1)/TILE_SIZE_A + 1, 1);
  dim3 bDim(TILE_SIZE_B, TILE_SIZE_A, 1);

  // Invoke CUDA kernel -----------------------------------------------------
  matrixMultiply_kernel <<< gDim, bDim >>> (A,B,C, numARows,numAColumns,numBRows,numBColumns,
							  numCRows,numCColumns);
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numAColumns, &numARows);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  numCRows = numARows;
  numCColumns = numBColumns;
  hostC = ( float * )malloc(sizeof(float) * numCRows * numCColumns);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void**) &deviceA, sizeof(float) * numARows * numAColumns);
  cudaMalloc((void**) &deviceB, sizeof(float) * numBRows * numBColumns);
  cudaMalloc((void**) &deviceC, sizeof(float) * numCRows * numCColumns);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns,
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns,
             cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiply(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows,
                 numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns,
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");
	
  //matMult_cpu(hostA, hostB, hostC, numARows, numAColumns, numBRows,numBColumns, numCRows, numCColumns);
	
  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
