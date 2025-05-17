NVCC = nvcc
FLAGS_CUDA = -Iinc

GCC = gcc
FLAGS = -Iinc -g -Wall -fopenmp -O3

# LIBDIRS = -L/usr/local/cuda-12.6/lib64
# INCDIRS = -I/usr/local/cuda-12.6/include
LIBS = -lm
LIBS_CUDA = -lcudart

OUT = spmv
OUT_CUDA = spmv_cuda

DEPS = inc/spmv.h inc/data.h 
DEPS_CUDA = inc/spmv_cpu.cuh inc/data_cuda.cuh inc/spmv_cuda.cuh inc/error.cuh inc/utils.cuh

OBJ = obj/main.o obj/spmv.o obj/data.o
OBJ_CUDA = obj/spmv_cpu.o obj/main_cuda.o obj/spmv_cuda.o obj/data_cuda.o obj/error.o obj/utils.o

all: bin/$(OUT) bin/$(OUT_CUDA)

obj/%.o : src/%.c $(DEPS)
	$(GCC) $(FLAGS) -c -o $@ $<

obj/%.o : src/%.cu $(DEPS_CUDA)
	$(NVCC) $(FLAGS_CUDA) -c -o $@ $<

bin/$(OUT):	$(OBJ)
	$(GCC) $(FLAGS) -o $@ $^ $(LIBS)

bin/$(OUT_CUDA): $(OBJ_CUDA)
	$(NVCC) $(FLAGS_CUDA) -o $@ $^ $(LIBS_CUDA)

.PHONY : clean

clean :
	rm -rf bin/$(OUT) obj/*