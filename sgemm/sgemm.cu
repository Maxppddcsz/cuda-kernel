#include <cuda_runtime.h>
template <const int BM,
          const int BN,
          const int BK,
          const int TM,
          const int TN>
__global__ void sgemm_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta)
{
    int tile_row = BM / TM;
    int tile_col = BN / TN;
    int thread_num = tile_row * tile_col;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = (threadIdx.x % tile_row) * TN;
    int ty = (threadIdx.x / tile_row) * TM;

    __shared__ As[BM * BK];
    __shared__ Bs[BM * BK];

    A = &A[by * BM * K];
    B = &B[bx * BN * K];
    C = &C[by * BM * N + bx * BN];

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM][YN] = {0.0f};
    for (int k = 0; k < K; k += BK)
    {
        // 搬运数据到共享内存中
        for (int i = 0; i < BM; i += a_tile_stride)
        {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        for (int i = 0; i < BM; i += b_tile_stride)
        {
            Bs[(a_tile_row + i) * BN + b_tile_col] = A[(a_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();

        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++)
        {
            for (int j = 0; j < TM; j++)
            {
                for (int l = 0; l < TN; l++)
                {
                    tmp[j][l] += As[(ty + j) * BK + i] * Bs[i * BN + l + tx];
                }
            }
        }
        __syncthreads();
    }
    // 计算
    for (int i = 0; i < TM; i++)
    {
        for (int j = 0; j < TN; j++)
        {
            C[(i + ty) * N + tx + l] = alpha * tmp[i][j] + beta * C[(i + ty) * N + tx + l];
        }
    }
}