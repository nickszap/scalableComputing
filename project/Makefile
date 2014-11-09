
CC       = nvcc
CC_FLAGS = -O3 -I/usr/local/cuda/include
LD_FLAGS = -lcudart -L/usr/local/cuda/lib64
EXE	     = derivPrice

default: $(EXE)

$(EXE): main.cu
	$(CC) $(CC_FLAGS) $(LD_FLAGS) main.cu -o $(EXE)

clean:
	rm -rf $(EXE)

