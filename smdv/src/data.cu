#include "data.cuh"

#include <malloc.h>

#define MAX_LINE 256

void set_array_random(float *arr, size_t size, float max_value);

// TODO -> fix gestione errori con puntatore a Data e ritorno int
Data read_from_file(char *path)
{
    Data data;

    FILE *fp = fopen(path, "r");

    // Check if file is opened
    if (!fp)
    {
        perror("File opening failed!\n");
        return data;    // TODO fix here
    }

    char line[MAX_LINE];

    // Skip first comment lines
    while (fgets(line, MAX_LINE, fp)) 
    {
        if (line[0] != '%')
            break;
    }

    // Get #rows, #cols, #vals
    size_t rows, cols, vals;
    if (sscanf(line, "%ld %ld %ld", &rows, &cols, &vals) != 3)
    {
        printf("Error: Invalid matrix header format\n");
        fclose(fp);
        return data;    // TODO fix here
    }

    data.row_num = rows;
    data.col_num = cols;
    data.val_num = vals;

    int counter = 0;

    // Alloc memory
    data.row = (size_t *) malloc(sizeof(size_t) * data.val_num);
    data.col = (size_t *) malloc(sizeof(size_t) * data.val_num);
    data.val = (float *) malloc(sizeof(float) * data.val_num);
    
    size_t *row_count = (size_t *) calloc(data.row_num, sizeof(size_t));    // set to 0
    
    // Read matrix entries
    size_t row, col;
    float value;
    while (fscanf(fp, "%ld %ld %f", &row, &col, &value) == 3)
    {
        data.row[counter] = row - 1;    // In data, indexing starts at 1
        data.col[counter] = col - 1;
        data.val[counter] = value;
        
        row_count[row - 1] += 1;    // Preparatory for thread_start
        counter++;
    }
    
    fclose(fp);
    
    data.thread_start = (size_t *) malloc(sizeof(size_t) * data.row_num);
    data.arr = (float *) malloc(sizeof(float) * data.col_num);
    data.res = (float *) malloc(sizeof(float) * data.row_num);

    // Thread start calculation
    data.thread_start[0] = 0;
    for (int i = 1; i < data.row_num; i++)
    {
        data.thread_start[i] = data.thread_start[i - 1] + row_count[i - 1];
    }

    // Random array for multiplication
    set_array_random(data.arr, data.col_num, 1);

    return data;
}

/**
 * @brief Fills array with random values in [0, max_value]
 */
void set_array_random(float *arr, size_t size, float max_value)
{
    srand((unsigned) time(NULL));

    for (int i = 0; i < size; i++) {
        arr[i] = rand() / (float) RAND_MAX * max_value;
    }
}

/**
 * @details Result should be [8, 15, 0, 2, 18, 10]
 */
Data test_data()
{
    Data data;

    data.row_num = 6;
    data.col_num = 4;
    data.val_num = 7;

    data.row = (size_t *) malloc(sizeof(size_t) * data.val_num);
    data.col = (size_t *) malloc(sizeof(size_t) * data.val_num);
    data.val = (float *) malloc(sizeof(float) * data.val_num);
    data.arr = (float *) malloc(sizeof(float) * data.col_num);
    data.res = (float *) malloc(sizeof(float) * data.row_num);
    data.thread_start = (size_t *) malloc(sizeof(size_t) * data.row_num);

    size_t row[] = {0, 0, 1, 3, 4, 5, 5};
    size_t col[] = {1, 2, 3, 0, 1, 0, 2};
    float val[] = {1, 2, 5, 1, 3, 4, 2};
    float arr[] = {2, 6, 1, 3};
    size_t thread_start[] = {0, 2, 3, 3, 4, 5};

    for (int i = 0; i < data.val_num; i++) {
        data.row[i] = row[i];
        data.col[i] = col[i];
        data.val[i] = val[i];
    }
    for (int i = 0; i < data.col_num; i++) {
        data.arr[i] = arr[i];
    }
    for (int i = 0; i < data.row_num; i++) {
        data.thread_start[i] = thread_start[i];
    }

    return data;
}