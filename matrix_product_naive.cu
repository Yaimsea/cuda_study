#include <iostream>
#include <cmath>

const double eps = 1e-10;
const int N = 4096;
const int M1 = 4096;
const int M2 = 4096;
const int blockSize = 16;
const bool debugEnabled = false;

__global__ void matrixProduct(const double *A,const double *B,double *C,int n,int m1,int m2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx < m2 && idy < n)
    {
        for(int i = 0; i < m1; ++i)
            C[idy * m2 + idx] += A[idy * m1 + i] * B[i * m2 + idx];
    }
    return;
}

void debugPrint(int i,int j,double *h_C,double targetNumber)
{
    std::cout << i << ' ' << j << std::endl;
    std::cout << h_C[i * M2 + j] << ' ' << targetNumber << std::endl;
    std::cout << fabs((h_C[i * M2 + j] - targetNumber) / targetNumber) << std::endl;
    return;
}

int main()
{
    size_t size1 = N * M1 * sizeof(double);
    size_t size2 = M1 * M2 * sizeof(double);
    size_t size3 = N * M2 * sizeof(double);

    double *h_A = (double*)malloc(size1);
    double *h_B = (double*)malloc(size2);
    double *h_C = (double*)malloc(size3);
    for(int i = 0; i < N; ++i)
        for(int j = 0;j < M1; ++j)
            h_A[i * M1 + j] = (double)(i + 1);
    for(int i = 0; i < M1; ++i)
        for(int j = 0;j < M2; ++j)
            h_B[i * M2 + j] = (double)(j + 1);
    for(int i = 0; i < N; ++i)
        for(int j = 0;j < M2; ++j)
            h_C[i * M2 + j] = 0.0;

    double *d_A,*d_B,*d_C;
    cudaMalloc((void**)&d_A, size1);
    cudaMalloc((void**)&d_B, size2);
    cudaMalloc((void**)&d_C, size3);

    cudaMemcpy(d_A, h_A, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size3, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blockPerGird((M2 + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    int tileSize = (M1 + blockSize - 1) / blockSize;
    matrixProduct<<<blockPerGird,threadsPerBlock>>>(d_A, d_B, d_C, N, M1, M2);

    cudaMemcpy(h_C, d_C, size3, cudaMemcpyDeviceToHost);

    bool flag = true;
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0;j < M2; ++j)
        {
            double targetNumber = ((double)((i + 1) * (j + 1))) * M1;
            if(fabs((h_C[i * M2 + j] - targetNumber) / targetNumber) > eps)
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

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}