NVCC = nvcc
FLAGS_CUDA = -Iinc -Wno-deprecated-gpu-targets -O3 -arch=sm_50 --expt-relaxed-constexpr

GCC = gcc
FLAGS = -Iinc -g -Wall -fopenmp -O3

LIBS = -lm
LIBS_CUDA = -lcudart -lcurand -lm

OUT = lsh
OUT_CUDA = lsh_cuda

DEPS = inc/data.h inc/lsh.h inc/nn_search.h
DEPS_CUDA = inc/lsh_cuda.cuh inc/error.cuh inc/gpu_hash_table.cuh inc/data.h

OBJ = obj/main.o obj/data.o obj/lsh.o obj/nn_search.o
OBJ_CUDA = obj/main_cuda.o obj/data_cuda.o obj/lsh_cuda.o obj/error.o obj/gpu_hash_table.o

all: directories bin/$(OUT) bin/$(OUT_CUDA)

directories:
	@mkdir -p obj bin

obj/%.o : src/%.c $(DEPS)
	$(GCC) $(FLAGS) -c -o $@ $<

obj/%.o : src/%.cu $(DEPS_CUDA)
	$(NVCC) $(FLAGS_CUDA) -c -o $@ $<

obj/main_cuda.o : src/main.cu $(DEPS_CUDA)
	$(NVCC) $(FLAGS_CUDA) -c -o $@ $<

obj/data_cuda.o : src/data.c $(DEPS)
	$(NVCC) $(FLAGS_CUDA) -x c -c -o $@ $<

bin/$(OUT):	$(OBJ)
	$(GCC) $(FLAGS) -o $@ $^ $(LIBS)

bin/$(OUT_CUDA): $(OBJ_CUDA)
	$(NVCC) -o $@ $^ $(LIBS_CUDA)

.PHONY : clean directories

clean :
	rm -rf bin/$(OUT) bin/$(OUT_CUDA) obj/*