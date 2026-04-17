#include <iostream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cublas_v2.h>

//Run it on Windows:
//nvidia-smi -lgc 2850,2850
//nvidia-smi -rgc
//Run it on Windows or Linux:
//nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv

//nvcc -O3 -lineinfo cublasSgemm.cu -o cublasSgemm -lcublas

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

#define CHECK_CUBLAS(call) \
    do{ \
        cublasStatus_t status = (call); \
        if(status != CUBLAS_STATUS_SUCCESS) \
        { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << status << std::endl; \
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
const bool debugEnabled = true;

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

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    for(int i = 0; i < warmupNum; ++i)
    {
        CHECK_CUDA(cudaMemcpy(d_C, h_C, size3, cudaMemcpyHostToDevice));

        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            M2,
            N,
            M1,
            &alpha,
            d_B,
            M2,
            d_A,
            M1,
            &beta,
            d_C,
            M2
        ));

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

        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            M2,
            N,
            M1,
            &alpha,
            d_B,
            M2,
            d_A,
            M1,
            &beta,
            d_C,
            M2
        ));

        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs[i], start, stop));

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUBLAS(cublasDestroy(handle));

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
    for(int i = 0; i < testNum; ++i)
        std::cout << elapsedMs[i] << " ms" << std::endl;

    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);

    return 0;
}