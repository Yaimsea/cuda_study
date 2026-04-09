#include <iostream>

const float eps=1e-5;

__global__ void vectoradd(const float *A,const float *B,float *C,int N)
{
    int i = blockIdx.x * blockDim.x + blockIdx.x;
    if(i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    int N = 50000000;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    for(int i = 0; i < N; ++i)
        h_A[i] = 1.0, h_B[i] = 2.0;
    
    float *d_A,*d_B,*d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectoradd<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    bool flag = false;
    for(int i = 0;i < N; ++i)
    {
        if(fabs(h_C[i] - 3.0f) > eps)
        {
            flag = true;
            break;
        }
    }

    std::cout << (flag ? "OK! Correct!" : "WA! Check your code.") << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
