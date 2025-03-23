CC = gcc
CFLAGS = -Wall -Iinc

OUT = smdv

DEPS = inc/smdv.h inc/data.h
OBJ = obj/main.o obj/smdv.o obj/data.o


all: bin/$(OUT)

obj/%.o : src/%.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<


bin/$(OUT):	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY : clean

clean :
	rm -rf bin/$(OUT) obj/*