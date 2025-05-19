# GPU Computing assignments

## 1° assignment - SpMV

### How to execute
Go inside [spmv](./spmv/) folder and run the following commands
```
make
```

For the CPU algorithms:
```
./bin/spmv <filename>
```
For the GPU algorithm:
```
./bin/spmv_cuda <filename>
```

There are some data files are located in the [data](./spmv/data/) folder

If run with no parameters, uses test data to perform the multiplication

For the CPU algorithm use one of the following flags:
- `--alg` (or nothing) for the naive CPU algorithm
- `--alg1` for the sorted CPU algorithm
- `--alg2` for the OpenMP CPU algorithm

To profile the program, run the following command
```
valgrind --tool=cachegrind ./bin/spmv -p <filename>
```
You can use the `--alg[num]` flag to select the algorithm (only base and 1)

And then use the following command to analyze each part of the program
```
cg_annotate --auto=yes <cachegrind.out>
```

To run on the DISI cluster:
```
module load CUDA/12.3.2
```
```
make
```

For CPU:
```
sbatch run.sh <filename>
```

For GPU:
```
sbatch run_gpu.sh <filename>
```