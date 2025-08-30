#include <cuda_runtime.h>

__device__ static float atomicMax(float *address, float val)
{
    int *int_address = (int *)address;
    int old = *int_address;
    int assumed;
    do
    {
        assumed = old;
        old = atomicCAS(int_address, assumed, __float_as_int(fmax(val, __int_as_float(assumed))));
    } while (old != assumed);
    return __int_as_float(old);
}
__global__ void max_kernel(float *in, float *max_val, int N)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    __shared__ float sdata[32];

    if (tid > N)
    {
        return;
    }

    float val = in[tid];
    for (int offset = warpSize >> 1; offset >= 0; offset >>= 1)
    {
        val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }

    if (laneId == 0)
    {
        sdata[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0)
    {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? sdata[laneId] : (-INFINITY);
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        {

            val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if (laneId == 0)
        {
            atomicMax(max_val, val);
        }
    }
}

__global__ void sum_kernel(float *in, float *out, int N)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    __shared__ float sdata[32];
    float sum = tid < N ? in[tid] : 0.0f;
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
        int warpNum = blockDim.x / warpSize;
        sum = laneId < warpNum ? sdata[laneId] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (laneId == 0)
        {
            atomicAdd(out, sum);
        }
    }
}

__global__ void soft_max_kernel(float *in, float *out, float *sum, float *max_val, int N)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N)
    {
        out[tid] = expf(in[tid] - *max_val) / (*sum);
    }
}
