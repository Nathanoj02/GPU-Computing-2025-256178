#include "data.cuh"

#include <malloc.h>

#define MAX_LINE 256

#define EPS 1.e-3

void set_array_random(float *arr, size_t size, float max_value);


int read_from_file_ordered(char *path, DataOrdered &data)
{
    FILE *fp = fopen(path, "r");

    // Check if file is opened
    if (!fp)
    {
        perror("File opening failed!\n");
        return -1;
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
        fclose(fp);
        printf("Error: Invalid matrix header format\n");
        return -2;
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

    return 0;
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
DataOrdered test_data_ordered()
{
    DataOrdered data;

    data.row_num = 6;
    data.col_num = 4;
    data.val_num = 7;

    data.row = (size_t *) malloc(sizeof(size_t) * data.val_num);
    data.col = (size_t *) malloc(sizeof(size_t) * data.val_num);
    data.val = (float *) malloc(sizeof(float) * data.val_num);
    data.arr = (float *) malloc(sizeof(float) * data.col_num);
    data.res = (float *) malloc(sizeof(float) * data.row_num);
    data.thread_start = (size_t *) malloc(sizeof(size_t) * data.row_num);

    size_t row[] = {0, 0, 3, 1, 4, 5, 5};
    size_t col[] = {1, 2, 0, 3, 1, 0, 2};
    float val[] = {1, 2, 1, 5, 3, 4, 2};
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


// Merges two subarrays of row[], while maintaining col[] and val[]
void merge(size_t *row, size_t *col, float *val, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Temporary arrays for row, col, and val
    size_t *leftRow = (size_t *)malloc(n1 * sizeof(size_t));
    size_t *leftCol = (size_t *)malloc(n1 * sizeof(size_t));
    float *leftVal = (float *)malloc(n1 * sizeof(float));

    size_t *rightRow = (size_t *)malloc(n2 * sizeof(size_t));
    size_t *rightCol = (size_t *)malloc(n2 * sizeof(size_t));
    float *rightVal = (float *)malloc(n2 * sizeof(float));

    if (!leftRow || !rightRow || !leftCol || !rightCol || !leftVal || !rightVal) {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    // Copy data to temporary arrays
    for (int i = 0; i < n1; i++) {
        leftRow[i] = row[left + i];
        leftCol[i] = col[left + i];
        leftVal[i] = val[left + i];
    }
    for (int j = 0; j < n2; j++) {
        rightRow[j] = row[mid + 1 + j];
        rightCol[j] = col[mid + 1 + j];
        rightVal[j] = val[mid + 1 + j];
    }

    // Merge the temporary arrays back into row[], col[], and val[]
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (leftRow[i] <= rightRow[j]) {
            row[k] = leftRow[i];
            col[k] = leftCol[i];
            val[k] = leftVal[i];
            i++;
        } else {
            row[k] = rightRow[j];
            col[k] = rightCol[j];
            val[k] = rightVal[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of left arrays, if any
    while (i < n1) {
        row[k] = leftRow[i];
        col[k] = leftCol[i];
        val[k] = leftVal[i];
        i++;
        k++;
    }

    // Copy the remaining elements of right arrays, if any
    while (j < n2) {
        row[k] = rightRow[j];
        col[k] = rightCol[j];
        val[k] = rightVal[j];
        j++;
        k++;
    }

    // Free allocated memory
    free(leftRow);
    free(rightRow);
    free(leftCol);
    free(rightCol);
    free(leftVal);
    free(rightVal);
}

// The subarray to be sorted is in the index range [left-right]
void mergeSort(size_t *row, size_t *col, float *val, int left, int right) {
    if (left < right) {
      
        // Calculate the midpoint
        int mid = left + (right - left) / 2;

        // Sort first and second halves
        mergeSort(row, col, val, left, mid);
        mergeSort(row, col, val, mid + 1, right);

        // Merge the sorted halves
        merge(row, col, val, left, mid, right);
    }
}


bool check_data(float *check, float *base, size_t size)
{
    for (int i = 0; i < size; i++)
    {
        if (abs(check[i] - base[i]) > EPS) {
            printf("\nRiga %d:\t", i);
            printf("%lf vs %lf\n\n", check[i], base[i]);
            return false;
        }
    }
    //
    return true;
}