#include <iostream>
#include <cmath>
#include <algorithm>

const float eps = 1e-4;
const int N = 4096;
const int M1 = 4096;
const int M2 = 4096;
const int imod = 13;
const int jmod = 17;
const float base = 10.0f;
const int x_blockSize = 32;
const int y_blockSize = 8;
const int tileSize = 32;
// The condition must be met that
//     x_blockSize >= y_blockSize ,
//     tileSize = x_blockSize and
//     x_blockSize % y_blockSize == 0 .
// It is very important!
static_assert(x_blockSize >= y_blockSize);
static_assert(x_blockSize == tileSize);
static_assert(x_blockSize % y_blockSize == 0);
const bool debugEnabled = true;

__global__ void matrixProduct(const float *A,const float *B,float *C,int n,int m1,int m2,int tileNum,int tileLoadPasses)
{
    __shared__ float As[y_blockSize][tileSize];
    __shared__ float Bs[tileSize][x_blockSize];
    float sum = 0.0;

    for(int i = 0; i < tileNum; ++i)
    {
        int Aidx = i * tileSize + threadIdx.x;
        int Aidy = blockIdx.y * blockDim.y + threadIdx.y;
        if(Aidx >= m1 || Aidy >= n)
            As[threadIdx.y][threadIdx.x] = 0.0f;
        else
            As[threadIdx.y][threadIdx.x] = A[Aidy * m1 + Aidx];
        for(int j = 0; j < tileLoadPasses; ++j)
        {
            int Bidx = blockIdx.x * blockDim.x + threadIdx.x;
            int Bidy = i * tileSize + j * y_blockSize + threadIdx.y;
            if(Bidx >= m2 || Bidy >= m1)
                Bs[j * y_blockSize + threadIdx.y][threadIdx.x] = 0.0f;
            else
                Bs[j * y_blockSize + threadIdx.y][threadIdx.x] = B[Bidy * m2 + Bidx];
        }
        __syncthreads();
        for(int j = 0; j < tileSize; ++j)
            sum += As[threadIdx.y][j] * Bs[j][threadIdx.x];
        __syncthreads();
    }

    int Cidx = blockIdx.x * blockDim.x + threadIdx.x;
    int Cidy = blockIdx.y * blockDim.y + threadIdx.y;
    if(Cidx < m2 && Cidy < n)
        C[Cidy * m2 + Cidx] = sum;
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
    cudaMalloc((void**)&d_A, size1);
    cudaMalloc((void**)&d_B, size2);
    cudaMalloc((void**)&d_C, size3);

    cudaMemcpy(d_A, h_A, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size3, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(x_blockSize, y_blockSize);
    dim3 blockPerGird((M2 + x_blockSize - 1) / x_blockSize, (N + y_blockSize - 1) / y_blockSize);
    int tileNum = (M1 + x_blockSize - 1) / x_blockSize;
    int tileLoadPasses = x_blockSize / y_blockSize;
    matrixProduct<<<blockPerGird,threadsPerBlock>>>(d_A, d_B, d_C, N, M1, M2, tileNum, tileLoadPasses);

    cudaMemcpy(h_C, d_C, size3, cudaMemcpyDeviceToHost);

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

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}