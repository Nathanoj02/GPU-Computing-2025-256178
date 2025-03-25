CC = gcc
CFLAGS = -Wall -Iinc

NVCC = nvcc
NV_FLAGS = -Iinc

LIBDIRS=-L/usr/local/cuda-12.6/lib64
INCDIRS=-I/usr/local/cuda-12.6/include

OUT = smdv

DEPS = inc/smdv.h inc/data.h
NV_DEPS = inc/smdv_cuda.cuh inc/error.cuh

OBJ = obj/main.o obj/smdv.o obj/data.o obj/smdv_cuda.o obj/error.o


all: bin/$(OUT)

obj/%.o : src/%.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

obj/%.o : src/%.cu $(NV_DEPS)
	$(NVCC) $(NV_FLAGS) -c -o $@ $<

bin/$(OUT):	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBDIRS) $(INCDIRS)

.PHONY : clean

clean :
	rm -rf bin/$(OUT) obj/*