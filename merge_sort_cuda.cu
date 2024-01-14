#include <iostream>
#include <sys/time.h>
#include <cstdlib>
#include <ctime>


// data[], size, threads, blocks, 
void mergesort(long*, long, dim3, dim3);
// A[]. B[], size, width, slices, nThreads
__global__ void gpu_mergesort(long*, long*, long, long, long, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(long*, long*, long, long, long);


void generateRandomNumbers(long* data, long size) {
    std::srand(std::time(0)); // Seed for random number generation

    for (long i = 0; i < size; ++i) {
        data[i] = std::rand() % 1000; // Modify this according to your range
    }
}

#define min(a, b) (a < b ? a : b)

void printHelp(char* program) {

    std::cout
            << "usage: " << program << " [-xyzXYZv]\n"
            << '\n'
            << "-x, -y, -z are numbers of threads in each dimension. On my machine\n"
            << "  the correct number is x*y*z = 32\n"
            << '\n'
            << "-X, -Y, -Z are numbers of blocks to use in each dimension. Each block\n"
            << "  holds x*y*z threads, so the total number of threads is:\n"
            << "  x*y*z*X*Y*Z\n"
            << '\n'
            << "-v prints out extra info\n"
            << '\n'
            << "? prints this message and exits\n"
            << '\n'
            << "example: ./mergesort -x 8 -Y 10 -v\n"
            << '\n'
            << "Reads in a list of integer numbers from stdin, and performs\n"
            << "a bottom-up merge sort:\n"
            << '\n'
            << "Input:          8 3 1 9 1 2 7 5 9 3 6 4 2 0 2 5\n"
            << "Threads: |    t1    |    t2    |    t3    |    t4    |\n"
            << "         | 8 3 1 9  | 1 2 7 5  | 9 3 6 4  | 2 0 2 5  |\n"
            << "         |  38 19   |  12 57   |  39 46   |  02 25   |\n"
            << "         |   1398   |   1257   |   3469   |   0225   |\n"
            << "         +----------+----------+----------+----------+\n"
            << "         |          t1         |          t2         |\n"
            << "         |       11235789      |       02234569      |\n"
            << "         +---------------------+---------------------+\n"
            << "         |                     t1                    |\n"
            << "         |      0 1 1 2 2 2 3 3 4 5 5 6 7 8 9 9      |\n"
            << '\n'
            << '\n';
}



void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    cudaMalloc((void**) &D_data, size * sizeof(long));
    cudaMalloc((void**) &D_swp, size * sizeof(long));

    // Copy from our input list into the first array
    cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice);
 
    //
    // Copy the thread / block info to the GPU as well
    //
    cudaMalloc((void**) &D_threads, sizeof(dim3));
    cudaMalloc((void**) &D_blocks, sizeof(dim3));

    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        std::cout << "mergeSort - width: " << width 
                    << ", slices: " << slices 
                    << ", nThreads: " << nThreads << '\n';


        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost);
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width*idx*slices, 
         middle, 
         end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

int main(int argc, char** argv) {

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 128;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 64;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    long size = 10000000;
    long* data = new long[size];
    generateRandomNumbers(data, size);

    // for (int i=0; i<size; i++){
    //     std::cout << data[i] << " ";
    // }
    // std::cout << "\n";

    mergesort(data, size, threadsPerBlock, blocksPerGrid);

    // for (int i = 0; i < size; i++) {
    //     std::cout << data[i];
    // } 
    // std::cout << "\n";

    bool is_sorted = true;
    for (int i = 0; i < size - 1; i++) {
        if (data[i] > data[i + 1]) {
            is_sorted = false;
            break;
        }
    }
    std::cout << "Array is sorted: " << (is_sorted ? "true" : "false") << std::endl;
}
