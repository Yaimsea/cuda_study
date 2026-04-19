#include <iostream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdlib>

//Run it on Windows:
//nvidia-smi -lgc 2850,2850
//nvidia-smi -rgc
//Run it on Windows or Linux:
//nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv

#define CHECK_CUDA(call) \
    do{ \
        cudaError_t err = (call); \
        if(err != cudaSuccess) \
        { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err \
                      << " (" << cudaGetErrorString(err) << ")" << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    }while(0)

const int testNum = 1;
const int warmupNum = 3;
const float eps = 1e-4;
const int N = 4096;
const int M1 = 4096;
const int M2 = 4096;
const int imod = 13;
const int jmod = 17;
const float base = 10.0f;
const int x_blockSize = 32;
const int y_blockSize = 64;
const int tileSize = 64; 
static_assert(tileSize == std::max(x_blockSize, y_blockSize));
static_assert(x_blockSize % y_blockSize == 0 || y_blockSize % x_blockSize == 0);
const int y_registerSize = 8;
static_assert(y_blockSize % y_registerSize == 0);
const int As_padding = 0;
const int Bs_padding = 0;
/*
    The condition must be met that
        tileSize == max(x_blockSize, y_blockSize) ,
        x_blockSize % y_blockSize == 0 or y_blockSize % x_blockSize == 0,
        x_blockSize % x_registerSize == 0 and
        y_blockSize % y_registerSize == 0 .
    It is very important!

    One block computes an output tile of C with shape
    (y_blockSize, x_blockSize).

    One thread computes an output tile with shape
    (y_registerSize, x_registerSize).

    Therefore, the thread block shape should be
    (x_blockSize / x_registerSize, y_blockSize / y_registerSize).
*/
const bool debugEnabled = true;

__global__ void matrixProduct(const float *A,const float *B,float *C,int tileNum,int x_tileLoadPasses)
{
    __shared__ float As[y_blockSize][tileSize + As_padding];
    __shared__ float Bs[tileSize][x_blockSize + Bs_padding];
    float sum[y_registerSize] = {0.0f};

    for(int i = 0; i < tileNum; ++i)
    {
        for(int j = 0; j < x_tileLoadPasses; ++j)
        {
            int bAidx = i * tileSize + j * x_blockSize + threadIdx.x;
            int bAidy = blockIdx.y * y_blockSize + threadIdx.y * y_registerSize;
            for(int Ridy = 0; Ridy < y_registerSize; ++Ridy)
            {
                int Asidx = j * x_blockSize + threadIdx.x;
                int Asidy = threadIdx.y * y_registerSize + Ridy;
                int Aidx = bAidx;
                int Aidy = bAidy + Ridy;
                if(Aidx >= M1 || Aidy >= N)
                    As[Asidy][Asidx] = 0.0f;
                else
                    As[Asidy][Asidx] = A[Aidy * M1 + Aidx];
            }
        }
        int bBidx = blockIdx.x * x_blockSize + threadIdx.x;
        int bBidy = i * tileSize + threadIdx.y * y_registerSize;
        for(int Ridy = 0; Ridy < y_registerSize; ++Ridy)
        {
            int Bsidx = threadIdx.x;
            int Bsidy = threadIdx.y * y_registerSize + Ridy;
            int Bidx = bBidx;
            int Bidy = bBidy + Ridy;
            if(Bidx >= M2 || Bidy >= M1)
                Bs[Bsidy][Bsidx] = 0.0f;
            else
                Bs[Bsidy][Bsidx] = B[Bidy * M2 + Bidx];
        }
        __syncthreads();
        for(int j = 0; j < tileSize; ++j)
        {
            for(int Ridy = 0; Ridy < y_registerSize; ++Ridy)
            {
                int Asidy = threadIdx.y * y_registerSize + Ridy;
                int Bsidx = threadIdx.x;
                sum[Ridy] += As[Asidy][j] * Bs[j][Bsidx];
            }
        }
        __syncthreads();
    }

    for(int Ridy = 0; Ridy < y_registerSize; ++Ridy)
    {
        int Cidx = blockIdx.x * x_blockSize + threadIdx.x;
        int Cidy = blockIdx.y * y_blockSize + threadIdx.y * y_registerSize + Ridy;
        if(Cidx < M2 && Cidy < N)
            C[Cidy * M2 + Cidx] = sum[Ridy];
    }
    return;
}

void debugPrint(int i,int j,float *h_C,float targetNumber)
{
    std::cout << i << ' ' << j << std::endl;
    std::cout << h_C[i * M2 + j] << ' ' << targetNumber << std::endl;
    std::cout << fabs((h_C[i * M2 + j] - targetNumber) / targetNumber) << std::endl;
    return;
}

int main()
{
    size_t size1 = N * M1 * sizeof(float);
    size_t size2 = M1 * M2 * sizeof(float);
    size_t size3 = N * M2 * sizeof(float);

    float *h_A = (float*)malloc(size1);
    float *h_B = (float*)malloc(size2);
    float *h_C = (float*)malloc(size3);
    for(int i = 0; i < N; ++i)
        for(int j = 0;j < M1; ++j)
            h_A[i * M1 + j] = (float)(i % imod) / base + 0.1f;
    for(int i = 0; i < M1; ++i)
        for(int j = 0;j < M2; ++j)
            h_B[i * M2 + j] = (float)(j % jmod) / base + 0.1f;
    for(int i = 0; i < N; ++i)
        for(int j = 0;j < M2; ++j)
            h_C[i * M2 + j] = 0.0f;

    float *d_A,*d_B,*d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size1));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size2));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size3));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size1, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, size3, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(x_blockSize, y_blockSize / y_registerSize);
    dim3 blockPerGird((M2 + x_blockSize - 1) / x_blockSize, (N + y_blockSize - 1) / y_blockSize);
    int tileNum = (M1 + tileSize - 1) / tileSize;
    int x_tileLoadPasses = std::max(1, y_blockSize / x_blockSize);

    for(int i = 0; i < warmupNum; ++i)
    {
        CHECK_CUDA(cudaMemcpy(d_C, h_C, size3, cudaMemcpyHostToDevice));

        matrixProduct<<<blockPerGird,threadsPerBlock>>>(d_A, d_B, d_C, tileNum, x_tileLoadPasses);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    float elapsedMs[testNum] = {0.0f};

    for(int i = 0; i < testNum; ++i)
    {
        CHECK_CUDA(cudaMemcpy(d_C, h_C, size3, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));

        matrixProduct<<<blockPerGird,threadsPerBlock>>>(d_A, d_B, d_C, tileNum, x_tileLoadPasses);
        
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs[i], start, stop));

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    std::sort(elapsedMs, elapsedMs + testNum);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, size3, cudaMemcpyDeviceToHost));

    bool flag = true;
    float maxRelativeError = 0.0f;
    int i_maxRelativeError = -1, j_maxRelativeError = -1;
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0;j < M2; ++j)
        {
            float targetNumber = ((float)(i % imod) / base + 0.1f) * ((float)(j % jmod) / base + 0.1f) * M1;
            float relativeError = fabs((h_C[i * M2 + j] - targetNumber) / targetNumber);
            if(relativeError > maxRelativeError)
            {
                maxRelativeError = relativeError;
                i_maxRelativeError = i;
                j_maxRelativeError = j;
            }
            if(relativeError > eps)
            {
                flag = false;
                if(debugEnabled)
                    debugPrint(i, j, h_C, targetNumber);
                break;
            }
        }
        if(!flag)
            break;
    }

    std::cout << (flag ? "OK! Correct!" : "WA! Check your code!") << std::endl;
    std::cout << "Max relative error is " << maxRelativeError << std::endl;
    std::cout << "Obtain it when i = " << i_maxRelativeError << " and j = " << j_maxRelativeError << std::endl;
    std::cout << "Median kernel time: " << elapsedMs[testNum / 2] << " ms" << std::endl;
    std::cout << "The kernel time for these tests:" << std::endl;
    for(int i = 0; i < testNum; ++i)
        std::cout << elapsedMs[i] << " ms" << std::endl;

    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);

    return 0;
}