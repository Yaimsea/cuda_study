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

const int testNum = 9;
const int warmupNum = 5;
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
const int threadNum = 128;
const int As_x_loadSize = 4;
const int As_y_loadSize = 4;
static_assert(x_blockSize % As_x_loadSize == 0);
static_assert(y_blockSize % As_y_loadSize == 0);
static_assert(As_x_loadSize % 4 == 0);
static_assert(M1 % 4 == 0);
const int Bs_x_loadSize = 4;
const int Bs_y_loadSize = 4;
static_assert(x_blockSize % Bs_x_loadSize == 0);
static_assert(y_blockSize % Bs_y_loadSize == 0);
static_assert(Bs_x_loadSize % 4 == 0);
static_assert(M2 % 4 == 0);
const int x_computeSize = 2;
const int y_computeSize = 8;
static_assert(x_blockSize % x_computeSize == 0);
static_assert(y_blockSize % y_computeSize == 0);
static_assert(As_x_loadSize * As_y_loadSize == Bs_x_loadSize * Bs_y_loadSize);
static_assert(As_x_loadSize * As_y_loadSize == x_computeSize * y_computeSize);
static_assert(As_x_loadSize * As_y_loadSize * threadNum == x_blockSize * y_blockSize);
const int As_padding = 0;
const int Bs_padding = 0;
const bool debugEnabled = true;

__global__ void matrixProduct(const float *A,const float *B,float *C,int tileNum,int x_tileLoadPasses,int y_tileLoadPasses)
{
    __shared__ float As[y_blockSize][tileSize + As_padding];
    __shared__ float Bs[tileSize][x_blockSize + Bs_padding];
    float sum[y_computeSize][x_computeSize] = {0.0f};
    int As_loadIdx = threadIdx.x % (x_blockSize / As_x_loadSize);
    int As_loadIdy = threadIdx.x / (x_blockSize / As_x_loadSize);
    int Bs_loadIdx = threadIdx.x % (x_blockSize / Bs_x_loadSize);
    int Bs_loadIdy = threadIdx.x / (x_blockSize / Bs_x_loadSize);
    int computeIdx = threadIdx.x % (x_blockSize / x_computeSize);
    int computeIdy = threadIdx.x / (x_blockSize / x_computeSize);

    for(int i = 0; i < tileNum; ++i)
    {
        for(int j = 0; j < x_tileLoadPasses; ++j)
        {
            int bAidx = i * tileSize + j * x_blockSize + As_loadIdx * As_x_loadSize;
            int bAidy = blockIdx.y * y_blockSize + As_loadIdy * As_y_loadSize;
            for(int Ridy = 0; Ridy < As_y_loadSize; ++Ridy)
                for(int Ridx = 0; Ridx < As_x_loadSize; Ridx += 4)
                {
                    int Asidx = j * x_blockSize + As_loadIdx * As_x_loadSize + Ridx;
                    int Asidy = As_loadIdy * As_y_loadSize + Ridy;
                    int Aidx = bAidx + Ridx;
                    int Aidy = bAidy + Ridy;
                    if(Aidx >= M1 || Aidy >= N)
                        reinterpret_cast<float4*>(As[Asidy])[Asidx / 4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    else
                        reinterpret_cast<float4*>(As[Asidy])[Asidx / 4] = 
                        reinterpret_cast<const float4*>(A)[(Aidy * M1 + Aidx) / 4];
                }
        }
        int bBidx = blockIdx.x * x_blockSize + Bs_loadIdx * Bs_x_loadSize;
        int bBidy = i * tileSize + Bs_loadIdy * Bs_y_loadSize;
        for(int Ridy = 0; Ridy < Bs_y_loadSize; ++Ridy)
            for(int Ridx = 0; Ridx < Bs_x_loadSize; Ridx += 4)
            {
                int Bsidx = Bs_loadIdx * Bs_x_loadSize + Ridx;
                int Bsidy = Bs_loadIdy * Bs_y_loadSize + Ridy;
                int Bidx = bBidx + Ridx;
                int Bidy = bBidy + Ridy;
                if(Bidx >= M2 || Bidy >= M1)
                    reinterpret_cast<float4*>(Bs[Bsidy])[Bsidx / 4] = {0.0f, 0.0f, 0.0f, 0.0f};
                else
                    reinterpret_cast<float4*>(Bs[Bsidy])[Bsidx / 4] = 
                    reinterpret_cast<const float4*>(B)[(Bidy * M2 + Bidx) / 4];
            }
        __syncthreads();
        for(int j = 0; j < tileSize; ++j)
        {
            for(int Ridy = 0; Ridy < y_computeSize; ++Ridy)
                for(int Ridx = 0; Ridx < x_computeSize; ++Ridx)
                {
                    int Asidy = computeIdy * y_computeSize + Ridy;
                    int Bsidx = computeIdx * x_computeSize + Ridx;
                    sum[Ridy][Ridx] += As[Asidy][j] * Bs[j][Bsidx];
                }
        }
        __syncthreads();
    }

    for(int Ridy = 0; Ridy < y_computeSize; ++Ridy)
        for(int Ridx = 0; Ridx < x_computeSize; ++Ridx)
        {
            int Cidx = blockIdx.x * x_blockSize + computeIdx * x_computeSize + Ridx;
            int Cidy = blockIdx.y * y_blockSize + computeIdy * y_computeSize + Ridy;
            if(Cidx < M2 && Cidy < N)
                C[Cidy * M2 + Cidx] = sum[Ridy][Ridx];
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

    int threadsPerBlock = threadNum;
    dim3 blockPerGird((M2 + x_blockSize - 1) / x_blockSize, (N + y_blockSize - 1) / y_blockSize);
    int tileNum = (M1 + tileSize - 1) / tileSize;
    int x_tileLoadPasses = std::max(1, y_blockSize / x_blockSize);
    int y_tileLoadPasses = std::max(1, x_blockSize / y_blockSize);

    for(int i = 0; i < warmupNum; ++i)
    {
        CHECK_CUDA(cudaMemcpy(d_C, h_C, size3, cudaMemcpyHostToDevice));

        matrixProduct<<<blockPerGird,threadsPerBlock>>>(d_A, d_B, d_C, tileNum, x_tileLoadPasses, y_tileLoadPasses);

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

        matrixProduct<<<blockPerGird,threadsPerBlock>>>(d_A, d_B, d_C, tileNum, x_tileLoadPasses, y_tileLoadPasses);
        
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