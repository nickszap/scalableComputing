
CC       = nvcc
CC_FLAGS = -O3 -I/usr/local/cuda/include
LD_FLAGS = -lcudart -L/usr/local/cuda/lib64
EXE	     = derivPrice

default: $(EXE)

$(EXE): kernel.cu main.cu kernel_opt.cu main_opt.cu
	$(CC) $(CC_FLAGS) $(LD_FLAGS) main_opt.cu -o $(EXE)

clean:
	rm -rf $(EXE)

