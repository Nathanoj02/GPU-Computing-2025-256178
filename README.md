# GPU Computing assignments

## 1° assignment - SpMV

### How to execute
Go inside [spmv](./spmv/) folder and run the following commands
```
make
```
```
./bin/spmv <filename>
```

There are some data files are located in the [data](./spmv/data/) folder

If run with no parameters, uses test data to perform the multiplication

To run with the GPU algorithm, don't add other flags.
For the CPU algorithm use one of the following flags:
- `--cpu` for the naive CPU algorithm
- `--cpu1` for the sorted CPU algorithm
- `--cpu2` for the sorted and tiled CPU algorithm



To profile the program, run the following command
```
valgrind --tool=cachegrind ./bin/spmv <filename> -p
```
You can use the `--cpu[num]` flag to select the algorithm