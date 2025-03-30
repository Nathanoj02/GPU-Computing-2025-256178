NVCC = nvcc
FLAGS = -Iinc

# LIBDIRS = -L/usr/local/cuda-12.6/lib64
# INCDIRS = -I/usr/local/cuda-12.6/include
LIBS = -lcudart

OUT = spmv

DEPS = inc/spmv.cuh inc/data.cuh inc/spmv_cuda.cuh inc/error.cuh inc/utils.cuh

OBJ = obj/main.o obj/spmv.o obj/data.o obj/spmv_cuda.o obj/error.o obj/utils.o

all: bin/$(OUT)

obj/%.o : src/%.cu $(DEPS)
	$(NVCC) $(FLAGS) -c -o $@ $<

bin/$(OUT):	$(OBJ)
	$(NVCC) $(FLAGS) -o $@ $^ $(LIBS)

.PHONY : clean

clean :
	rm -rf bin/$(OUT) obj/*