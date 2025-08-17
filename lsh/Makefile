NVCC = nvcc
FLAGS_CUDA = -Iinc

GCC = gcc
FLAGS = -Iinc -g -Wall -fopenmp -O3

LIBS = -lm
LIBS_CUDA = -lcudart

OUT = lsh
# OUT_CUDA = spmv_cuda

DEPS = inc/data.h inc/lsh.h inc/nn_search.h
# DEPS_CUDA = inc/spmv_cpu.cuh inc/data_cuda.cuh inc/spmv_cuda.cuh inc/error.cuh inc/utils.cuh

OBJ = obj/main.o obj/data.o obj/lsh.o obj/nn_search.o
# OBJ_CUDA = obj/spmv_cpu.o obj/main_cuda.o obj/spmv_cuda.o obj/data_cuda.o obj/error.o obj/utils.o

all: bin/$(OUT) # bin/$(OUT_CUDA)

obj/%.o : src/%.c $(DEPS)
	$(GCC) $(FLAGS) -c -o $@ $<

# obj/%.o : src/%.cu $(DEPS_CUDA)
# 	$(NVCC) $(FLAGS_CUDA) -c -o $@ $<

bin/$(OUT):	$(OBJ)
	$(GCC) $(FLAGS) -o $@ $^ $(LIBS)

# bin/$(OUT_CUDA): $(OBJ_CUDA)
# 	$(NVCC) $(FLAGS_CUDA) -o $@ $^ $(LIBS_CUDA)

.PHONY : clean

clean :
	rm -rf bin/$(OUT) obj/*