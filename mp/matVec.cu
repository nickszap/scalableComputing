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

__global__ void spmvCSRKernel(float *out, int *matCols, int *matRows,
                              float *matData, float *vec, int dim) {
    //@@ insert spmv kernel for csr format
	
	unsigned int iRow = blockIdx.x * blockDim.x + threadIdx.x;
	if (iRow<dim){ //stay w/in matrix bounds
		int iFlat=matRows[iRow];
		float sum=0.0;
		for (int iCol=matRows[iRow]; iCol<matRows[iRow+1]; iCol++){ //# elts in row
			//every thread touches its own 
			sum += matData[iFlat]*vec[matCols[iFlat]];
			iFlat++;
		}
		out[iRow] = sum;
	} //iRow<dim	
}

const unsigned int BLOCK_SIZE = 32;

__global__ void spmvJDSKernel(float *out, int *matColStart, int *matCols,
                              int *matRowPerm, int *matRows, float *matData,
                              float *vec, int dim) {
    //@@ insert spmv kernel for jds format
	
	unsigned int iRow = blockIdx.x * blockDim.x + threadIdx.x;
	if (iRow<dim){ //stay w/in matrix bounds
		int rowInd = matRowPerm[iRow];
		float sum = 0.0;
		for (int iCol=0; iCol<matRows[iRow]; iCol++){ //# elts in row
			int iFlat = matColStart[iCol]+iRow;
			int colInd = matCols[iFlat];
			sum += matData[iFlat]*vec[colInd];
		}
		out[rowInd] = sum;
	}
	
}

static void spmvCSR(float *out, int *matCols, int *matRows, float *matData,
                    float *vec, int dim) {

    //@@ invoke spmv kernel for csr format
	dim3 gridDim((dim-1)/BLOCK_SIZE + 1, 1, 1);
	dim3 blockDim(BLOCK_SIZE, 1, 1);
	
	// Invoke CUDA kernel -----------------------------------------------------
	spmvCSRKernel <<< gridDim, blockDim >>> (out,matCols,matRows,matData,vec,dim);
	
}

static void spmvJDS(float *out, int *matColStart, int *matCols, int *matRowPerm,
                    int *matRows, float *matData, float *vec, int dim) {

    //@@ invoke spmv kernel for jds format
	
	dim3 gridDim((dim-1)/BLOCK_SIZE + 1, 1, 1);
	dim3 blockDim(BLOCK_SIZE, 1, 1);
	
	// Invoke CUDA kernel -----------------------------------------------------
	spmvJDSKernel <<< gridDim, blockDim >>> (out,matColStart,matCols,matRowPerm,matRows,matData,vec,dim);
}

void cpu_csr(float *out, int *matCols, int *matRows, float *matData,
             float *vec, int dim) {
	//testing on cpu first
	//from lectures:
	//data[nNonZeros], columnInds[nNonZeros], rowPointers[nRows+1]
	
	//from class forum (piazza):
	//The matrix is square and dim is the number of rows and columns in the matrix and number or elements in the vectors.
	//matRows is the same as jdsRowsNNZ
	
	int iRow, iCol;
	int iFlat=0;
	for (iRow=0; iRow<dim; iRow++){
		//
		for (iCol=matRows[iRow]; iCol<matRows[iRow+1]; iCol++){ //# elts in row
			out[iRow] += matData[iFlat]*vec[matCols[iFlat]];
			iFlat++;
		}
	}
	
}

void cpu_jds(float *out, int *matColStart, int *matCols, int *matRowPerm,
             int *matRows, float *matData, float *vec, int dim){
	//testing on cpu first
	//from class forum:
	/*
	matColStart~jdsColStartIdx
	MatCols~jdsColIdx 
	matRowPerm~jdsRowPerm
	matData~jdsData
	matRows~jdsRowsNNZ
	*/
	int iRow, iCol, rowInd, colInd;
	int iFlat=0;
	for (iRow=0; iRow<dim; iRow++){
		//
		rowInd = matRowPerm[iRow];
		for (iCol=0; iCol<matRows[iRow]; iCol++){ //# elts in row
			iFlat = matColStart[iCol]+iRow;
			colInd = matCols[iFlat];
			out[rowInd] += matData[iFlat]*vec[colInd];
		}
	}
}

int main(int argc, char **argv) {
  wbArg_t args;
  bool usingJDSQ;
  int *hostCSRCols;
  int *hostCSRRows;
  float *hostCSRData;
  int *hostJDSColStart;
  int *hostJDSCols;
  int *hostJDSRowPerm;
  int *hostJDSRows;
  float *hostJDSData;
  float *hostVector;
  float *hostOutput;
  int *deviceCSRCols;
  int *deviceCSRRows;
  float *deviceCSRData;
  int *deviceJDSColStart;
  int *deviceJDSCols;
  int *deviceJDSRowPerm;
  int *deviceJDSRows;
  float *deviceJDSData;
  float *deviceVector;
  float *deviceOutput;
  int dim, ncols, nrows, ndata;
  int maxRowNNZ;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  usingJDSQ = wbImport_flag(wbArg_getInputFile(args, 0)) == 1;
  hostCSRCols = (int *)wbImport(wbArg_getInputFile(args, 1), &ncols, "Integer");
  hostCSRRows = (int *)wbImport(wbArg_getInputFile(args, 2), &nrows, "Integer");
  hostCSRData = (float *)wbImport(wbArg_getInputFile(args, 3), &ndata, "Real");
  hostVector = (float *)wbImport(wbArg_getInputFile(args, 4), &dim, "Real");

  //hostOutput = (float *)malloc(sizeof(float) * dim);
  hostOutput = (float *)calloc(dim, sizeof(float)); //when just running cpu versions

  wbTime_stop(Generic, "Importing data and creating memory on host");

  if (usingJDSQ) {
    CSRToJDS(dim, hostCSRRows, hostCSRCols, hostCSRData, &hostJDSRowPerm,
             &hostJDSRows, &hostJDSColStart, &hostJDSCols, &hostJDSData);
    maxRowNNZ = hostJDSRows[0];
  }

  wbTime_start(GPU, "Allocating GPU memory.");
  if (usingJDSQ) {
    cudaMalloc((void **)&deviceJDSColStart, sizeof(int) * maxRowNNZ);
    cudaMalloc((void **)&deviceJDSCols, sizeof(int) * ndata);
    cudaMalloc((void **)&deviceJDSRowPerm, sizeof(int) * dim);
    cudaMalloc((void **)&deviceJDSRows, sizeof(int) * dim);
    cudaMalloc((void **)&deviceJDSData, sizeof(float) * ndata);
  } else {
    cudaMalloc((void **)&deviceCSRCols, sizeof(int) * ncols);
    cudaMalloc((void **)&deviceCSRRows, sizeof(int) * nrows);
    cudaMalloc((void **)&deviceCSRData, sizeof(float) * ndata);
  }
  cudaMalloc((void **)&deviceVector, sizeof(float) * dim);
  cudaMalloc((void **)&deviceOutput, sizeof(float) * dim);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  if (usingJDSQ) {
    cudaMemcpy(deviceJDSColStart, hostJDSColStart, sizeof(int) * maxRowNNZ,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSCols, hostJDSCols, sizeof(int) * ndata,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSRowPerm, hostJDSRowPerm, sizeof(int) * dim,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSRows, hostJDSRows, sizeof(int) * dim,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSData, hostJDSData, sizeof(float) * ndata,
               cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(deviceCSRCols, hostCSRCols, sizeof(int) * ncols,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCSRRows, hostCSRRows, sizeof(int) * nrows,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCSRData, hostCSRData, sizeof(float) * ndata,
               cudaMemcpyHostToDevice);
  }
  cudaMemcpy(deviceVector, hostVector, sizeof(float) * dim,
             cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  if (usingJDSQ) {
    spmvJDS(deviceOutput, deviceJDSColStart, deviceJDSCols, deviceJDSRowPerm,
            deviceJDSRows, deviceJDSData, deviceVector, dim);
  } else {
    spmvCSR(deviceOutput, deviceCSRCols, deviceCSRRows, deviceCSRData,
            deviceVector, dim);
  }
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * dim,
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceCSRCols);
  cudaFree(deviceCSRRows);
  cudaFree(deviceCSRData);
  cudaFree(deviceVector);
  cudaFree(deviceOutput);
  if (usingJDSQ) {
    cudaFree(deviceJDSColStart);
    cudaFree(deviceJDSCols);
    cudaFree(deviceJDSRowPerm);
    cudaFree(deviceJDSRows);
    cudaFree(deviceJDSData);
  }
  wbTime_stop(GPU, "Freeing GPU Memory");
  
  // run my cpu versions
#if 0
	if (usingJDSQ) {
		cpu_jds(hostOutput, hostJDSColStart, hostJDSCols, hostJDSRowPerm,
			hostJDSRows, hostJDSData, hostVector, dim);
	}
	else {
		cpu_csr(hostOutput, hostCSRCols, hostCSRRows, hostCSRData, hostVector, dim);
	}
#endif	
  //
  wbSolution(args, hostOutput, dim);

  free(hostCSRCols);
  free(hostCSRRows);
  free(hostCSRData);
  free(hostVector);
  free(hostOutput);
  if (usingJDSQ) {
    free(hostJDSColStart);
    free(hostJDSCols);
    free(hostJDSRowPerm);
    free(hostJDSRows);
    free(hostJDSData);
  }

  return 0;
}
