NVCC = nvcc
FLAGS = -Iinc

# LIBDIRS = -L/usr/local/cuda-12.6/lib64
# INCDIRS = -I/usr/local/cuda-12.6/include
LIBS = -lcudart

OUT = smdv

DEPS = inc/smdv.cuh inc/data.cuh inc/smdv_cuda.cuh inc/error.cuh inc/utils.cuh

OBJ = obj/main.o obj/smdv.o obj/data.o obj/smdv_cuda.o obj/error.o obj/utils.o

all: bin/$(OUT)

obj/%.o : src/%.cu $(DEPS)
	$(NVCC) $(FLAGS) -c -o $@ $<

bin/$(OUT):	$(OBJ)
	$(NVCC) $(FLAGS) -o $@ $^ $(LIBS)

.PHONY : clean

clean :
	rm -rf bin/$(OUT) obj/*