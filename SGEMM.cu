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
const int imod = 13;
const int jmod = 17;
const float base = 10.0f;
const bool debugEnabled = true;
const bool printExactTime = false;

void debugPrint(int i,int j,int N,float *h_C,float targetNumber)
{
    std::cout << i << ' ' << j << std::endl;
    std::cout << h_C[i * N + j] << ' ' << targetNumber << std::endl;
    std::cout << fabs((h_C[i * N + j] - targetNumber) / targetNumber) << std::endl;
    return;
}

template<
    int blockM_,
    int blockN_,
    int blockK_,
    int AloadM_,
    int AloadN_,
    int BloadM_,
    int BloadN_,
    int computeM_,
    int computeN_,
    int Apadding_,
    int Bpadding_
>
struct sgemmConfig_float4{
    static constexpr int blockM = blockM_;
    static constexpr int blockN = blockN_;
    static constexpr int blockK = blockK_;
    static constexpr int AloadM = AloadM_;
    static constexpr int AloadN = AloadN_;
    static constexpr int BloadM = BloadM_;
    static constexpr int BloadN = BloadN_;
    static constexpr int computeM = computeM_;
    static constexpr int computeN = computeN_;
    static constexpr int Apadding = Apadding_;
    static constexpr int Bpadding = Bpadding_;
    static constexpr int tileLoadPassesM = blockK_ / blockM_;
    static constexpr int tileLoadPassesN = blockK_ / blockN_;
    static constexpr int threadNum = (blockM_ / AloadM_) * (blockN_ / AloadN_);
    static_assert(blockK % blockM == 0);
    static_assert(blockK % blockN == 0);
    static_assert(blockN % AloadN == 0);
    static_assert(blockM % AloadM == 0);
    static_assert(AloadN % 4 == 0);
    static_assert(blockN % BloadN == 0);
    static_assert(blockM % BloadM == 0);
    static_assert(BloadN % 4 == 0);
    static_assert(blockN % computeN == 0);
    static_assert(blockM % computeM == 0);
    static_assert(AloadN * AloadM == BloadN * BloadM);
    static_assert(AloadN * AloadM == computeN * computeM);
    static_assert(AloadN * AloadM * threadNum == blockN * blockM);
    static_assert((blockK + Apadding) % 4 == 0);
    static_assert((blockN + Bpadding) % 4 == 0);
};

template<
    int blockM_,
    int blockN_,
    int blockK_,
    int AloadM_,
    int AloadN_,
    int BloadM_,
    int BloadN_,
    int computeM_,
    int computeN_,
    int Apadding_,
    int Bpadding_
>
struct sgemmConfig_normal{
    static constexpr int blockM = blockM_;
    static constexpr int blockN = blockN_;
    static constexpr int blockK = blockK_;
    static constexpr int AloadM = AloadM_;
    static constexpr int AloadN = AloadN_;
    static constexpr int BloadM = BloadM_;
    static constexpr int BloadN = BloadN_;
    static constexpr int computeM = computeM_;
    static constexpr int computeN = computeN_;
    static constexpr int Apadding = Apadding_;
    static constexpr int Bpadding = Bpadding_;
    static constexpr int tileLoadPassesM = blockK_ / blockM_;
    static constexpr int tileLoadPassesN = blockK_ / blockN_;
    static constexpr int threadNum = (blockM_ / AloadM_) * (blockN_ / AloadN_);
    static_assert(blockK % blockM == 0);
    static_assert(blockK % blockN == 0);
    static_assert(blockN % AloadN == 0);
    static_assert(blockM % AloadM == 0);
    static_assert(blockN % BloadN == 0);
    static_assert(blockM % BloadM == 0);
    static_assert(blockN % computeN == 0);
    static_assert(blockM % computeM == 0);
    static_assert(AloadN * AloadM == BloadN * BloadM);
    static_assert(AloadN * AloadM == computeN * computeM);
    static_assert(AloadN * AloadM * threadNum == blockN * blockM);
};

template<class config>
__global__ void matrixProduct_float4_normal(const float *A,const float *B,float *C,int m,int n,int k)
{
    constexpr int blockM = config::blockM;
    constexpr int blockN = config::blockN;
    constexpr int blockK = config::blockK;
    constexpr int AloadM = config::AloadM;
    constexpr int AloadN = config::AloadN;
    constexpr int BloadM = config::BloadM;
    constexpr int BloadN = config::BloadN;
    constexpr int computeM = config::computeM;
    constexpr int computeN = config::computeN;
    constexpr int Apadding = config::Apadding;
    constexpr int Bpadding = config::Bpadding;
    constexpr int tileLoadPassesM = config::tileLoadPassesM;
    constexpr int tileLoadPassesN = config::tileLoadPassesN;

    __shared__ float As[blockM][blockK + Apadding];
    __shared__ float Bs[blockK][blockN + Bpadding];
    float sum[computeM][computeN] = {0.0f};
    float loadA[tileLoadPassesN][AloadM][AloadN];
    float loadB[tileLoadPassesM][BloadM][BloadN];
    int As_loadIdx = threadIdx.x % (blockN / AloadN);
    int As_loadIdy = threadIdx.x / (blockN / AloadN);
    int Bs_loadIdx = threadIdx.x % (blockN / BloadN);
    int Bs_loadIdy = threadIdx.x / (blockN / BloadN);
    int computeIdx = threadIdx.x % (blockN / computeN);
    int computeIdy = threadIdx.x / (blockN / computeN);

    for(int j = 0; j < tileLoadPassesN; ++j)
    {
        int bAidx = j * blockN + As_loadIdx * AloadN;
        int bAidy = blockIdx.y * blockM + As_loadIdy * AloadM;
        int base_load_Aidy = bAidy * k;
        for(int Ridy = 0; Ridy < AloadM; ++Ridy)
        {
            for(int Ridx = 0; Ridx < AloadN; Ridx += 4)
            {
                int Asidx = j * blockN + As_loadIdx * AloadN + Ridx;
                int Asidy = As_loadIdy * AloadM + Ridy;
                int Aidx = bAidx + Ridx;
                int Aidy = bAidy + Ridy;
                if(Aidx >= k || Aidy >= m)
                    reinterpret_cast<float4*>(As[Asidy])[Asidx / 4] = {0.0f, 0.0f, 0.0f, 0.0f};
                else
                    reinterpret_cast<float4*>(As[Asidy])[Asidx / 4] = 
                    reinterpret_cast<const float4*>(A)[(base_load_Aidy + Aidx) / 4];
            }
            base_load_Aidy += k;
        }
    }
    for(int j = 0; j < tileLoadPassesM; ++j)
    {
        int bBidx = blockIdx.x * blockN + Bs_loadIdx * BloadN;
        int bBidy = j * blockM + Bs_loadIdy * BloadM;
        int base_load_Bidy = bBidy * n;
        for(int Ridy = 0; Ridy < BloadM; ++Ridy)
        {
            for(int Ridx = 0; Ridx < BloadN; Ridx += 4)
            {
                int Bsidx = Bs_loadIdx * BloadN + Ridx;
                int Bsidy = j * blockM + Bs_loadIdy * BloadM + Ridy;
                int Bidx = bBidx + Ridx;
                int Bidy = bBidy + Ridy;
                if(Bidx >= n || Bidy >= k)
                    reinterpret_cast<float4*>(Bs[Bsidy])[Bsidx / 4] = {0.0f, 0.0f, 0.0f, 0.0f};
                else
                    reinterpret_cast<float4*>(Bs[Bsidy])[Bsidx / 4] = 
                    reinterpret_cast<const float4*>(B)[(base_load_Bidy + Bidx) / 4];
            }
            base_load_Bidy += n;
        }
    }
    __syncthreads();

    int base_compute_Asidy = computeIdy * computeM;
    int base_compute_Bsidx = computeIdx * computeN;
    for(int i = blockK; i < k; i += blockK)
    {
        for(int j = 0; j < tileLoadPassesN; ++j)
        {
            int bAidx = i + j * blockN + As_loadIdx * AloadN;
            int bAidy = blockIdx.y * blockM + As_loadIdy * AloadM;
            int base_load_Aidy = bAidy * k;
            for(int Ridy = 0; Ridy < AloadM; ++Ridy)
            {
                for(int Ridx = 0; Ridx < AloadN; Ridx += 4)
                {
                    int Aidx = bAidx + Ridx;
                    int Aidy = bAidy + Ridy;
                    if(Aidx >= k || Aidy >= m)
                        reinterpret_cast<float4*>(loadA[j][Ridy])[Ridx / 4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    else
                        reinterpret_cast<float4*>(loadA[j][Ridy])[Ridx / 4] = 
                        reinterpret_cast<const float4*>(A)[(base_load_Aidy + Aidx) / 4];
                }
                base_load_Aidy += k;
            }
        }
        for(int j = 0; j < tileLoadPassesM; ++j)
        {
            int bBidx = blockIdx.x * blockN + Bs_loadIdx * BloadN;
            int bBidy = i + j * blockM + Bs_loadIdy * BloadM;
            int base_load_Bidy = bBidy * n;
            for(int Ridy = 0; Ridy < BloadM; ++Ridy)
            {
                for(int Ridx = 0; Ridx < BloadN; Ridx += 4)
                {
                    int Bidx = bBidx + Ridx;
                    int Bidy = bBidy + Ridy;
                    if(Bidx >= n || Bidy >= k)
                        reinterpret_cast<float4*>(loadB[j][Ridy])[Ridx / 4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    else
                        reinterpret_cast<float4*>(loadB[j][Ridy])[Ridx / 4] = 
                        reinterpret_cast<const float4*>(B)[(base_load_Bidy + Bidx) / 4];
                }
                base_load_Bidy += n;
            }
        }

        for(int j = 0; j < blockK; ++j)
        {
            int Asidy = base_compute_Asidy;
            for(int Ridy = 0; Ridy < computeM; ++Ridy, ++Asidy)
            {
                int Bsidx = base_compute_Bsidx;
                for(int Ridx = 0; Ridx < computeN; ++Ridx, ++Bsidx)
                    sum[Ridy][Ridx] += As[Asidy][j] * Bs[j][Bsidx];
            }
        }
        __syncthreads();

        for(int j = 0; j < tileLoadPassesN; ++j)
        {
            for(int Ridy = 0; Ridy < AloadM; ++Ridy)
                for(int Ridx = 0; Ridx < AloadN; Ridx += 4)
                {
                    int Asidx = j * blockN + As_loadIdx * AloadN + Ridx;
                    int Asidy = As_loadIdy * AloadM + Ridy;
                    reinterpret_cast<float4*>(As[Asidy])[Asidx / 4] = 
                    reinterpret_cast<const float4*>(loadA[j][Ridy])[Ridx / 4];
                }
        }
        for(int j = 0; j < tileLoadPassesM; ++j)
        {
            for(int Ridy = 0; Ridy < BloadM; ++Ridy)
                for(int Ridx = 0; Ridx < BloadN; Ridx += 4)
                {
                    int Bsidx = Bs_loadIdx * BloadN + Ridx;
                    int Bsidy = j * blockM + Bs_loadIdy * BloadM + Ridy;
                    reinterpret_cast<float4*>(Bs[Bsidy])[Bsidx / 4] = 
                    reinterpret_cast<const float4*>(loadB[j][Ridy])[Ridx / 4];
                }
        }
        __syncthreads();
    }

    for(int j = 0; j < blockK; ++j)
    {
        int Asidy = base_compute_Asidy;
        for(int Ridy = 0; Ridy < computeM; ++Ridy, ++Asidy)
        {
            int Bsidx = base_compute_Bsidx;
            for(int Ridx = 0; Ridx < computeN; ++Ridx, ++Bsidx)
                sum[Ridy][Ridx] += As[Asidy][j] * Bs[j][Bsidx];
        }
    }

    for(int Ridy = 0; Ridy < computeM; ++Ridy)
        for(int Ridx = 0; Ridx < computeN; ++Ridx)
        {
            int Cidx = blockIdx.x * blockN + computeIdx * computeN + Ridx;
            int Cidy = blockIdx.y * blockM + computeIdy * computeM + Ridy;
            if(Cidx < n && Cidy < m)
                C[Cidy * n + Cidx] = sum[Ridy][Ridx];
        }
    return;
}

template<class config>
__global__ void matrixProduct_float4_padding(const float *A,const float *B,float *C,int m,int n,int k,int k_padding,int n_padding)
{
    constexpr int blockM = config::blockM;
    constexpr int blockN = config::blockN;
    constexpr int blockK = config::blockK;
    constexpr int AloadM = config::AloadM;
    constexpr int AloadN = config::AloadN;
    constexpr int BloadM = config::BloadM;
    constexpr int BloadN = config::BloadN;
    constexpr int computeM = config::computeM;
    constexpr int computeN = config::computeN;
    constexpr int Apadding = config::Apadding;
    constexpr int Bpadding = config::Bpadding;
    constexpr int tileLoadPassesM = config::tileLoadPassesM;
    constexpr int tileLoadPassesN = config::tileLoadPassesN;

    __shared__ float As[blockM][blockK + Apadding];
    __shared__ float Bs[blockK][blockN + Bpadding];
    float sum[computeM][computeN] = {0.0f};
    float loadA[tileLoadPassesN][AloadM][AloadN];
    float loadB[tileLoadPassesM][BloadM][BloadN];
    int As_loadIdx = threadIdx.x % (blockN / AloadN);
    int As_loadIdy = threadIdx.x / (blockN / AloadN);
    int Bs_loadIdx = threadIdx.x % (blockN / BloadN);
    int Bs_loadIdy = threadIdx.x / (blockN / BloadN);
    int computeIdx = threadIdx.x % (blockN / computeN);
    int computeIdy = threadIdx.x / (blockN / computeN);

    for(int j = 0; j < tileLoadPassesN; ++j)
    {
        int bAidx = j * blockN + As_loadIdx * AloadN;
        int bAidy = blockIdx.y * blockM + As_loadIdy * AloadM;
        int base_load_Aidy = bAidy * k_padding;
        for(int Ridy = 0; Ridy < AloadM; ++Ridy)
        {
            for(int Ridx = 0; Ridx < AloadN; Ridx += 4)
            {
                int Asidx = j * blockN + As_loadIdx * AloadN + Ridx;
                int Asidy = As_loadIdy * AloadM + Ridy;
                int Aidx = bAidx + Ridx;
                int Aidy = bAidy + Ridy;
                if(Aidx >= k || Aidy >= m)
                    reinterpret_cast<float4*>(As[Asidy])[Asidx / 4] = {0.0f, 0.0f, 0.0f, 0.0f};
                else
                    reinterpret_cast<float4*>(As[Asidy])[Asidx / 4] = 
                    reinterpret_cast<const float4*>(A)[(base_load_Aidy + Aidx) / 4];
            }
            base_load_Aidy += k_padding;
        }
    }
    for(int j = 0; j < tileLoadPassesM; ++j)
    {
        int bBidx = blockIdx.x * blockN + Bs_loadIdx * BloadN;
        int bBidy = j * blockM + Bs_loadIdy * BloadM;
        int base_load_Bidy = bBidy * n_padding;
        for(int Ridy = 0; Ridy < BloadM; ++Ridy)
        {
            for(int Ridx = 0; Ridx < BloadN; Ridx += 4)
            {
                int Bsidx = Bs_loadIdx * BloadN + Ridx;
                int Bsidy = j * blockM + Bs_loadIdy * BloadM + Ridy;
                int Bidx = bBidx + Ridx;
                int Bidy = bBidy + Ridy;
                if(Bidx >= n || Bidy >= k)
                    reinterpret_cast<float4*>(Bs[Bsidy])[Bsidx / 4] = {0.0f, 0.0f, 0.0f, 0.0f};
                else
                    reinterpret_cast<float4*>(Bs[Bsidy])[Bsidx / 4] = 
                    reinterpret_cast<const float4*>(B)[(base_load_Bidy + Bidx) / 4];
            }
            base_load_Bidy += n_padding;
        }
    }
    __syncthreads();

    int base_compute_Asidy = computeIdy * computeM;
    int base_compute_Bsidx = computeIdx * computeN;
    for(int i = blockK; i < k; i += blockK)
    {
        for(int j = 0; j < tileLoadPassesN; ++j)
        {
            int bAidx = i + j * blockN + As_loadIdx * AloadN;
            int bAidy = blockIdx.y * blockM + As_loadIdy * AloadM;
            int base_load_Aidy = bAidy * k_padding;
            for(int Ridy = 0; Ridy < AloadM; ++Ridy)
            {
                for(int Ridx = 0; Ridx < AloadN; Ridx += 4)
                {
                    int Aidx = bAidx + Ridx;
                    int Aidy = bAidy + Ridy;
                    if(Aidx >= k || Aidy >= m)
                        reinterpret_cast<float4*>(loadA[j][Ridy])[Ridx / 4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    else
                        reinterpret_cast<float4*>(loadA[j][Ridy])[Ridx / 4] = 
                        reinterpret_cast<const float4*>(A)[(base_load_Aidy + Aidx) / 4];
                }
                base_load_Aidy += k_padding;
            }
        }
        for(int j = 0; j < tileLoadPassesM; ++j)
        {
            int bBidx = blockIdx.x * blockN + Bs_loadIdx * BloadN;
            int bBidy = i + j * blockM + Bs_loadIdy * BloadM;
            int base_load_Bidy = bBidy * n_padding;
            for(int Ridy = 0; Ridy < BloadM; ++Ridy)
            {
                for(int Ridx = 0; Ridx < BloadN; Ridx += 4)
                {
                    int Bidx = bBidx + Ridx;
                    int Bidy = bBidy + Ridy;
                    if(Bidx >= n || Bidy >= k)
                        reinterpret_cast<float4*>(loadB[j][Ridy])[Ridx / 4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    else
                        reinterpret_cast<float4*>(loadB[j][Ridy])[Ridx / 4] = 
                        reinterpret_cast<const float4*>(B)[(base_load_Bidy + Bidx) / 4];
                }
                base_load_Bidy += n_padding;
            }
        }

        for(int j = 0; j < blockK; ++j)
        {
            int Asidy = base_compute_Asidy;
            for(int Ridy = 0; Ridy < computeM; ++Ridy, ++Asidy)
            {
                int Bsidx = base_compute_Bsidx;
                for(int Ridx = 0; Ridx < computeN; ++Ridx, ++Bsidx)
                    sum[Ridy][Ridx] += As[Asidy][j] * Bs[j][Bsidx];
            }
        }
        __syncthreads();

        for(int j = 0; j < tileLoadPassesN; ++j)
        {
            for(int Ridy = 0; Ridy < AloadM; ++Ridy)
                for(int Ridx = 0; Ridx < AloadN; Ridx += 4)
                {
                    int Asidx = j * blockN + As_loadIdx * AloadN + Ridx;
                    int Asidy = As_loadIdy * AloadM + Ridy;
                    reinterpret_cast<float4*>(As[Asidy])[Asidx / 4] = 
                    reinterpret_cast<const float4*>(loadA[j][Ridy])[Ridx / 4];
                }
        }
        for(int j = 0; j < tileLoadPassesM; ++j)
        {
            for(int Ridy = 0; Ridy < BloadM; ++Ridy)
                for(int Ridx = 0; Ridx < BloadN; Ridx += 4)
                {
                    int Bsidx = Bs_loadIdx * BloadN + Ridx;
                    int Bsidy = j * blockM + Bs_loadIdy * BloadM + Ridy;
                    reinterpret_cast<float4*>(Bs[Bsidy])[Bsidx / 4] = 
                    reinterpret_cast<const float4*>(loadB[j][Ridy])[Ridx / 4];
                }
        }
        __syncthreads();
    }

    for(int j = 0; j < blockK; ++j)
    {
        int Asidy = base_compute_Asidy;
        for(int Ridy = 0; Ridy < computeM; ++Ridy, ++Asidy)
        {
            int Bsidx = base_compute_Bsidx;
            for(int Ridx = 0; Ridx < computeN; ++Ridx, ++Bsidx)
                sum[Ridy][Ridx] += As[Asidy][j] * Bs[j][Bsidx];
        }
    }

    for(int Ridy = 0; Ridy < computeM; ++Ridy)
        for(int Ridx = 0; Ridx < computeN; ++Ridx)
        {
            int Cidx = blockIdx.x * blockN + computeIdx * computeN + Ridx;
            int Cidy = blockIdx.y * blockM + computeIdy * computeM + Ridy;
            if(Cidx < n && Cidy < m)
                C[Cidy * n + Cidx] = sum[Ridy][Ridx];
        }
    return;
}

template<class config>
__global__ void matrixProduct_normal(const float *A,const float *B,float *C,int m,int n,int k)
{
    constexpr int blockM = config::blockM;
    constexpr int blockN = config::blockN;
    constexpr int blockK = config::blockK;
    constexpr int AloadM = config::AloadM;
    constexpr int AloadN = config::AloadN;
    constexpr int BloadM = config::BloadM;
    constexpr int BloadN = config::BloadN;
    constexpr int computeM = config::computeM;
    constexpr int computeN = config::computeN;
    constexpr int Apadding = config::Apadding;
    constexpr int Bpadding = config::Bpadding;
    constexpr int tileLoadPassesM = config::tileLoadPassesM;
    constexpr int tileLoadPassesN = config::tileLoadPassesN;

    __shared__ float As[blockM][blockK + Apadding];
    __shared__ float Bs[blockK][blockN + Bpadding];
    float sum[computeM][computeN] = {0.0f};
    float loadA[tileLoadPassesN][AloadM][AloadN];
    float loadB[tileLoadPassesM][BloadM][BloadN];
    int As_loadIdx = threadIdx.x % (blockN / AloadN);
    int As_loadIdy = threadIdx.x / (blockN / AloadN);
    int Bs_loadIdx = threadIdx.x % (blockN / BloadN);
    int Bs_loadIdy = threadIdx.x / (blockN / BloadN);
    int computeIdx = threadIdx.x % (blockN / computeN);
    int computeIdy = threadIdx.x / (blockN / computeN);

    for(int j = 0; j < tileLoadPassesN; ++j)
    {
        int bAidx = j * blockN + As_loadIdx * AloadN;
        int bAidy = blockIdx.y * blockM + As_loadIdy * AloadM;
        int base_load_Aidy = bAidy * k;
        for(int Ridy = 0; Ridy < AloadM; ++Ridy)
        {
            for(int Ridx = 0; Ridx < AloadN; ++Ridx)
            {
                int Asidx = j * blockN + As_loadIdx * AloadN + Ridx;
                int Asidy = As_loadIdy * AloadM + Ridy;
                int Aidx = bAidx + Ridx;
                int Aidy = bAidy + Ridy;
                if(Aidx >= k || Aidy >= m)
                    As[Asidy][Asidx] = 0.0f;
                else
                    As[Asidy][Asidx] = A[base_load_Aidy + Aidx];
            }
            base_load_Aidy += k;
        }
    }
    for(int j = 0; j < tileLoadPassesM; ++j)
    {
        int bBidx = blockIdx.x * blockN + Bs_loadIdx * BloadN;
        int bBidy = j * blockM + Bs_loadIdy * BloadM;
        int base_load_Bidy = bBidy * n;
        for(int Ridy = 0; Ridy < BloadM; ++Ridy)
        {
            for(int Ridx = 0; Ridx < BloadN; ++Ridx)
            {
                int Bsidx = Bs_loadIdx * BloadN + Ridx;
                int Bsidy = j * blockM + Bs_loadIdy * BloadM + Ridy;
                int Bidx = bBidx + Ridx;
                int Bidy = bBidy + Ridy;
                if(Bidx >= n || Bidy >= k)
                    Bs[Bsidy][Bsidx] = 0.0f;
                else
                    Bs[Bsidy][Bsidx] = B[base_load_Bidy + Bidx];
            }
            base_load_Bidy += n;
        }
    }
    __syncthreads();

    int base_compute_Asidy = computeIdy * computeM;
    int base_compute_Bsidx = computeIdx * computeN;
    for(int i = blockK; i < k; i += blockK)
    {
        for(int j = 0; j < tileLoadPassesN; ++j)
        {
            int bAidx = i + j * blockN + As_loadIdx * AloadN;
            int bAidy = blockIdx.y * blockM + As_loadIdy * AloadM;
            int base_load_Aidy = bAidy * k;
            for(int Ridy = 0; Ridy < AloadM; ++Ridy)
            {
                for(int Ridx = 0; Ridx < AloadN; ++Ridx)
                {
                    int Aidx = bAidx + Ridx;
                    int Aidy = bAidy + Ridy;
                    if(Aidx >= k || Aidy >= m)
                        loadA[j][Ridy][Ridx] = 0.0f;
                    else
                        loadA[j][Ridy][Ridx] = A[base_load_Aidy + Aidx];
                }
                base_load_Aidy += k;
            }
        }
        for(int j = 0; j < tileLoadPassesM; ++j)
        {
            int bBidx = blockIdx.x * blockN + Bs_loadIdx * BloadN;
            int bBidy = i + j * blockM + Bs_loadIdy * BloadM;
            int base_load_Bidy = bBidy * n;
            for(int Ridy = 0; Ridy < BloadM; ++Ridy)
            {
                for(int Ridx = 0; Ridx < BloadN; ++Ridx)
                {
                    int Bidx = bBidx + Ridx;
                    int Bidy = bBidy + Ridy;
                    if(Bidx >= n || Bidy >= k)
                        loadB[j][Ridy][Ridx] = 0.0f;
                    else
                        loadB[j][Ridy][Ridx] = B[base_load_Bidy + Bidx];
                }
                base_load_Bidy += n;
            }
        }

        for(int j = 0; j < blockK; ++j)
        {
            int Asidy = base_compute_Asidy;
            for(int Ridy = 0; Ridy < computeM; ++Ridy, ++Asidy)
            {
                int Bsidx = base_compute_Bsidx;
                for(int Ridx = 0; Ridx < computeN; ++Ridx, ++Bsidx)
                    sum[Ridy][Ridx] += As[Asidy][j] * Bs[j][Bsidx];
            }
        }
        __syncthreads();

        for(int j = 0; j < tileLoadPassesN; ++j)
        {
            for(int Ridy = 0; Ridy < AloadM; ++Ridy)
                for(int Ridx = 0; Ridx < AloadN; ++Ridx)
                {
                    int Asidx = j * blockN + As_loadIdx * AloadN + Ridx;
                    int Asidy = As_loadIdy * AloadM + Ridy;
                    As[Asidy][Asidx] = loadA[j][Ridy][Ridx];
                }
        }
        for(int j = 0; j < tileLoadPassesM; ++j)
        {
            for(int Ridy = 0; Ridy < BloadM; ++Ridy)
                for(int Ridx = 0; Ridx < BloadN; ++Ridx)
                {
                    int Bsidx = Bs_loadIdx * BloadN + Ridx;
                    int Bsidy = j * blockM + Bs_loadIdy * BloadM + Ridy;
                    Bs[Bsidy][Bsidx] = loadB[j][Ridy][Ridx];
                }
        }
        __syncthreads();
    }

    for(int j = 0; j < blockK; ++j)
    {
        int Asidy = base_compute_Asidy;
        for(int Ridy = 0; Ridy < computeM; ++Ridy, ++Asidy)
        {
            int Bsidx = base_compute_Bsidx;
            for(int Ridx = 0; Ridx < computeN; ++Ridx, ++Bsidx)
                sum[Ridy][Ridx] += As[Asidy][j] * Bs[j][Bsidx];
        }
    }

    for(int Ridy = 0; Ridy < computeM; ++Ridy)
        for(int Ridx = 0; Ridx < computeN; ++Ridx)
        {
            int Cidx = blockIdx.x * blockN + computeIdx * computeN + Ridx;
            int Cidy = blockIdx.y * blockM + computeIdy * computeM + Ridy;
            if(Cidx < n && Cidy < m)
                C[Cidy * n + Cidx] = sum[Ridy][Ridx];
        }
    return;
}

void SGEMM(const float *d_A,const float *d_B,float *d_C,int M,int N,int K)
{
    if(M >= 1024 && N >= 1024 && K >= 1024)
    {
        using config = sgemmConfig_float4<64, 64, 64, 8, 4, 8, 4, 8, 4, 0, 0>;
        if(K % 4 || N % 4)
        {
            int K_padding = (K + 3) / 4 * 4;
            int N_padding = (N + 3) / 4 * 4;
            float *d_A_padding, *d_B_padding;
            CHECK_CUDA(cudaMalloc((void**)&d_A_padding , M * K_padding * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_A_padding, 0, M * K_padding * sizeof(float)));
            CHECK_CUDA(cudaMemcpy2D(
                d_A_padding, K_padding * sizeof(float),
                d_A, K * sizeof(float), 
                K * sizeof(float),
                M,
                cudaMemcpyDeviceToDevice
            ));
            CHECK_CUDA(cudaMalloc((void**)&d_B_padding , K * N_padding * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_B_padding, 0, K * N_padding * sizeof(float)));
            CHECK_CUDA(cudaMemcpy2D(
                d_B_padding, N_padding * sizeof(float),
                d_B, N * sizeof(float), 
                N * sizeof(float),
                K,
                cudaMemcpyDeviceToDevice
            ));
            int threadsPerBlock = config::threadNum;
            dim3 blockPerGird((N + config::blockN - 1) / config::blockN,
                            (M + config::blockM - 1) / config::blockM);
            matrixProduct_float4_padding<config><<<blockPerGird,threadsPerBlock>>>(d_A_padding, d_B_padding, d_C, M, N, K, K_padding, N_padding);
            CHECK_CUDA(cudaFree(d_A_padding)); CHECK_CUDA(cudaFree(d_B_padding));
        }
        else
        {
            int threadsPerBlock = config::threadNum;
            dim3 blockPerGird((N + config::blockN - 1) / config::blockN,
                            (M + config::blockM - 1) / config::blockM);
            matrixProduct_float4_normal<config><<<blockPerGird,threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
        }
    }
    else    if(M >= 16 && N >= 16 && K >= 16)
    {
        using config = sgemmConfig_normal<16, 16, 16, 4, 1, 4, 1, 4, 1, 0, 0>;
        int threadsPerBlock = config::threadNum;
        dim3 blockPerGird((N + config::blockN - 1) / config::blockN,
                        (M + config::blockM - 1) / config::blockM);
        matrixProduct_normal<config><<<blockPerGird,threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    }
    else
    {
        using config = sgemmConfig_normal<1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0>;
        int threadsPerBlock = config::threadNum;
        dim3 blockPerGird((N + config::blockN - 1) / config::blockN,
                        (M + config::blockM - 1) / config::blockM);
        matrixProduct_normal<config><<<blockPerGird,threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    }
    return;
}

void solve(int M, int N, int K,int testId)
{
    std::cout << "Test #" << testId << ":" << std::endl;

    if(M <= 0 || N <= 0 || K <= 0)
    {
        std::cout << "The matrix size is invalid!" << std::endl;
        return;
    }

    size_t size1 = M * K * sizeof(float);
    size_t size2 = K * N * sizeof(float);
    size_t size3 = M * N * sizeof(float);

    float *h_A = (float*)malloc(size1);
    float *h_B = (float*)malloc(size2);
    float *h_C = (float*)malloc(size3);
    for(int i = 0; i < M; ++i)
        for(int j = 0;j < K; ++j)
            h_A[i * K + j] = (float)(i % imod) / base + 0.1f;
    for(int i = 0; i < K; ++i)
        for(int j = 0;j < N; ++j)
            h_B[i * N + j] = (float)(j % jmod) / base + 0.1f;
    for(int i = 0; i < M; ++i)
        for(int j = 0;j < N; ++j)
            h_C[i * N + j] = 0.0f;

    float *d_A,*d_B,*d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size1));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size2));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size3));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size1, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, size3, cudaMemcpyHostToDevice));

    for(int i = 0; i < warmupNum; ++i)
    {
        CHECK_CUDA(cudaMemcpy(d_C, h_C, size3, cudaMemcpyHostToDevice));

        SGEMM(d_A, d_B, d_C, M, N, K);

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

        SGEMM(d_A, d_B, d_C, M, N, K);
        
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
    float maxRelativeError = -1.0f;
    int i_maxRelativeError = -1, j_maxRelativeError = -1;
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0;j < N; ++j)
        {
            float targetNumber = ((float)(i % imod) / base + 0.1f) * ((float)(j % jmod) / base + 0.1f) * K;
            float relativeError = fabs((h_C[i * N + j] - targetNumber) / targetNumber);
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
                    debugPrint(i, j, N, h_C, targetNumber);
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
    if(printExactTime)
    {
        std::cout << "The kernel time for these tests:" << std::endl;
        for(int i = 0; i < testNum; ++i)
            std::cout << elapsedMs[i] << " ms" << std::endl;
    }
    std::cout << std::endl;

    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);

    return;
}

int main()
{
    solve(4096, 4096, 4096, 1);
    solve(4093, 4095, 4094, 2);
    solve(4095, 2047, 1025, 3);
    solve(256, 256, 256, 4);
    solve(255, 127, 256, 5);
    solve(511, 255, 129, 6);
    solve(8, 8, 8, 7);
    solve(8, 7, 9, 8);
    solve(1, 1, 1, 9);
    solve(0, 100, 200, 10);
    return 0;
}