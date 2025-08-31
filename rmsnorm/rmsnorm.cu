#include <cuda_runtime.h>
__device__ float block_reduce(float sum)
{
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    __shared__ float sdata[32];
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (laneId == 0)
    {
        sdata[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0)
    {
        sum = threadIdx.x < (blockDim.x + warpSize - 1) / warpSize ? sdata[threadIdx.x] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
    }
    else
    {
        sum = 0.0f;
    }
    return sum;
}
__global__ void rmsnorm_kernel(float *in, float *wei, float *out, int size, float eps)
{
    float *block_in = in + blockIdx.x * size;
    float *block_out = out + blockIdx.x * size;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        sum += block_in[i] * block_in[i];
    }

    __shared__ float shared_val;
    sum = block_reduce(sum);
    if (threadIdx.x == 0)
    {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        block_out[i] = block_in[i] / rsqrt(sum / static_cast<float>(size) + eps) * wei[i];
    }
}