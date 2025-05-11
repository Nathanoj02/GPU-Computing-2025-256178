# GPU Computing assignments

## 1° assignment - SpMV

### How to execute
Go inside `spmv` folder and run the following commands
```
make
```
```
./bin/spmv <filename>
```

Some files are located in the `data` folder

If run with no parameters, uses test data to perform the multiplication

To profile the program, run the following command
```
valgrind --tool=cachegrind ./bin/spmv <filename> -p
```