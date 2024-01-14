#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 512

void generateRandomArray(int *arr, int size)
{
    srand(time(NULL));
    for (int i = 0; i < size; i++)
    {
        arr[i] = rand() % 1000; // Adjust the range as needed
    }
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(long *source, long *dest, long size, long width, long slices, dim3 *threads, dim3 *blocks)
{
    unsigned int idx = getIdx(threads, blocks);
    long start = width * idx * slices,
         middle,
         end;

    for (long slice = 0; slice < slices; slice++)
    {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(long *source, long *dest, long start, long middle, long end)
{
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++)
    {
        if (i < middle && (j >= end || source[i] < source[j]))
        {
            dest[k] = source[i];
            i++;
        }
        else
        {
            dest[k] = source[j];
            j++;
        }
    }
}

int main()
{
    const int arraySize = 100000;
    int arr[arraySize];
    generateRandomArray(arr, arraySize);

    int *d_arr;
    cudaMalloc((void **)&d_arr, arraySize * sizeof(int));
    cudaMemcpy(d_arr, arr, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    int *temp;
    cudaMalloc((void **)&temp, arraySize * sizeof(int));

    int blocks = (arraySize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Launch the parallel merge sort kernel
    bottomUpMergeSort<<<blocks, BLOCK_SIZE>>>(d_arr, temp, arraySize);

    // Copy the sorted array back to the host
    cudaMemcpy(arr, d_arr, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Free allocated memory
    cudaFree(d_arr);
    cudaFree(temp);

    // Print the sorted array
    printf("Sorted array: ");
    for (int i = 0; i < arraySize; i++)
    {
        printf("%d ", arr[i]);
    }

    return 0;
}
