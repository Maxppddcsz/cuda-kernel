#include <cuda_runtime.h>

template <
    const int TILE_DIM>
__global__ void transpose_kernel(float *in, float *out, int M, int N)
{
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;
    int x1 = bx + threadIdx.x;
    int y1 = by + threadIdx.y;

    __shared__ s[TILE_DIM][TILE_DIM];

    if (x1 < N && y1 < M)
    {
        s[threadIdx.y][threadIdx.x ^ threadIdx.y] = in[y1 * N + x1];
    }
    __syncthreads();

    int x2 = by + threadIdx.x;
    int y2 = bx + threadIdx.y;
    if (y2 < N && x2 < M)
    {
        in[y2 * M + x2] = s[threadIdx.x][threadIdx.y ^ threadIdx.x];
    }
}