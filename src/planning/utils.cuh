#pragma once

#include "Robots.hh"
#include "src/collision/environment.hh"
#include "src/collision/shapes.hh"
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cassert>

/* All device utils and collision functions */
#define M 4

namespace ppln::device_utils {
    using namespace collision;

    const int PANDA_SPHERE_COUNT = 59;
    const int PANDA_STOP_SELF_CC = 29;

    const int PANDA_APPROX_SPHERE_COUNT = 11;

    __device__ __constant__ int panda_link_index[PANDA_SPHERE_COUNT] = {
        0,  // panda_link0
        1, 1, 1, 1,  // panda_link1
        2, 2, 2, 2,  // panda_link2
        3, 3, 3, 3,  // panda_link3
        4, 4, 4, 4,  // panda_link4
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  // panda_link5
        6, 6, 6,  // panda_link6
        7, 7, 7, 7, 7,  // panda_link7
        9, 9, 9, 9, 9, 9,  // panda_hand (first 6)
        9, 9, 9, 9, 9, 9,  // panda_hand (next 6)
        9, 9, 9, 9, 9, 9,  // panda_hand (final 6)
        10, 10,  // panda_leftfinger
        11, 11   // panda_rightfinger
    };

    __device__ __constant__ int panda_approx_link_index[PANDA_APPROX_SPHERE_COUNT] = {
        0,  // panda_link0
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11        
    };

    __device__ __constant__ int panda_approx_startCC_ind[PANDA_APPROX_SPHERE_COUNT]={
        5,
        5,
        5, 
        12,
        12,
        7,
        12,
        12,
        12,
        12,
        12,
    };

    // for each sphere, from which sphere should self-collision check start looking at
    __device__ __constant__ int panda_startCC_ind[PANDA_SPHERE_COUNT] = {
        17,  // panda_link0
        17, 17, 17, 17,  // panda_link1
        17, 17, 17, 17,  // panda_link2
        59, 59, 59, 59,  // panda_link3
        59, 59, 59, 59,  // panda_link4
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,  // panda_link5
        59, 59, 59,  // panda_link6
        59, 59, 59, 59, 59,  // panda_link7
        59, 59, 59, 59, 59, 59,  // panda_hand (first 6)
        59, 59, 59, 59, 59, 59,  // panda_hand (next 6)
        59, 59, 59, 59, 59, 59,  // panda_hand (final 6)
        59, 59,  // panda_leftfinger
        59, 59   // panda_rightfinger
    };


    __device__ __constant__ float4 panda_local_xyz[PANDA_SPHERE_COUNT] = {
        {0.000000f, 0.000000f, 0.050000f, 0.080000f},
        {0.000000f, -0.080000f, 0.000000f, 0.060000f},
        {0.000000f, -0.030000f, 0.000000f, 0.060000f},
        {0.000000f, 0.000000f, -0.120000f, 0.060000f},
        {0.000000f, 0.000000f, -0.170000f, 0.060000f},
        {0.000000f, 0.000000f, 0.030000f, 0.060000f},
        {0.000000f, 0.000000f, 0.080000f, 0.060000f},
        {0.000000f, -0.120000f, 0.000000f, 0.060000f},
        {0.000000f, -0.170000f, 0.000000f, 0.060000f},
        {0.000000f, 0.000000f, -0.100000f, 0.060000f},
        {0.000000f, 0.000000f, -0.060000f, 0.050000f},
        {0.080000f, 0.060000f, 0.000000f, 0.055000f},
        {0.080000f, 0.020000f, 0.000000f, 0.055000f},
        { -0.080000f, 0.095000f, 0.000000f, 0.060000f},
        {0.000000f, 0.000000f, 0.020000f, 0.055000f},
        {0.000000f, 0.000000f, 0.060000f, 0.055000f},
        { -0.080000f, 0.060000f, 0.000000f, 0.055000f},
        {0.000000f, 0.055000f, 0.000000f, 0.060000f},
        {0.000000f, 0.075000f, 0.000000f, 0.060000f},
        {0.000000f, 0.000000f, -0.220000f, 0.060000f},
        {0.000000f, 0.050000f, -0.180000f, 0.050000f},
        {0.010000f, 0.080000f, -0.140000f, 0.025000f},
        {0.010000f, 0.085000f, -0.110000f, 0.025000f},
        {0.010000f, 0.090000f, -0.080000f, 0.025000f},
        {0.010000f, 0.095000f, -0.050000f, 0.025000f},
        {-0.010000f, 0.080000f, -0.140000f, 0.025000f},
        {-0.010000f, 0.085000f, -0.110000f, 0.025000f},
        {-0.010000f, 0.090000f, -0.080000f, 0.025000f},
        {-0.010000f, 0.095000f, -0.050000f, 0.025000f},
        {0.000000f, 0.000000f, 0.000000f, 0.050000f},
        {0.080000f, -0.010000f, 0.000000f, 0.050000f},
        {0.080000f, 0.035000f, 0.000000f, 0.052000f},
        {0.000000f, 0.000000f, 0.070000f, 0.050000f},
        {0.020000f, 0.040000f, 0.080000f, 0.025000f},
        {0.040000f, 0.020000f, 0.080000f, 0.025000f},
        {0.040000f, 0.060000f, 0.085000f, 0.020000f},
        {0.060000f, 0.040000f, 0.085000f, 0.020000f},
        {0.000000f, -0.075000f, 0.010000f, 0.028000f},
        {0.000000f, -0.045000f, 0.010000f, 0.028000f},
        {0.000000f, -0.015000f, 0.010000f, 0.028000f},
        {0.000000f, 0.015000f, 0.010000f, 0.028000f},
        {0.000000f, 0.045000f, 0.010000f, 0.028000f},
        {0.000000f, 0.075000f, 0.010000f, 0.028000f},
        {0.000000f, -0.075000f, 0.030000f, 0.026000f},
        {0.000000f, -0.045000f, 0.030000f, 0.026000f},
        {0.000000f, -0.015000f, 0.030000f, 0.026000f},
        {0.000000f, 0.015000f, 0.030000f, 0.026000f},
        {0.000000f, 0.045000f, 0.030000f, 0.026000f},
        {0.000000f, 0.075000f, 0.030000f, 0.026000f},
        {0.000000f, -0.075000f, 0.050000f, 0.024000f},
        {0.000000f, -0.045000f, 0.050000f, 0.024000f},
        {0.000000f, -0.015000f, 0.050000f, 0.024000f},
        {0.000000f, 0.015000f, 0.050000f, 0.024000f},
        {0.000000f, 0.045000f, 0.050000f, 0.024000f},
        {0.000000f, 0.075000f, 0.050000f, 0.024000f},
        {0.000000f, 0.015000f, 0.022000f, 0.012000f},
        {0.000000f, 0.008000f, 0.044000f, 0.012000f},
        {0.000000f, -0.015000f, 0.022000f, 0.012000f},
        {0.000000f, -0.008000f, 0.044000f, 0.012000f}
    };

    __device__ __constant__ float4 panda_approx_local_xyz[PANDA_APPROX_SPHERE_COUNT]={
        { 0.000f,   0.000f,   0.050f,   0.080f },   // panda_link0
        {-0.001f,  -0.039f,  -0.085f,   0.154f },   // panda_link1
        { 0.000f,  -0.085f,   0.040f,   0.154f },   // panda_link2
        { 0.039f,   0.028f,  -0.052f,   0.128f },   // panda_link3
        {-0.042f,   0.049f,   0.029f,   0.126f },   // panda_link4
        {-0.001f,   0.037f,  -0.110f,   0.176f },   // panda_link5
        { 0.042f,   0.014f,   0.000f,   0.095f },   // panda_link6
        { 0.015f,   0.015f,   0.075f,   0.072f },   // panda_link7
        { 0.000f,   0.000f,   0.022f,   0.104f },   // panda_hand
        { 0.000f,   0.012f,   0.033f,   0.024f },   // panda_leftfinger
        { 0.000f,  -0.012f,   0.033f,   0.024f },   // panda_rightfinger
    };

    __device__ __constant__ float4 approx_spheres_array[11] = {
        { 0.0f, 0.0f, 0.05f, 0.08f },
        { -0.001f, -0.039f, -0.085f, 0.154f },
        { 0.0f, -0.085f, 0.04f, 0.154f },
        { 0.039f, 0.028f, -0.052f, 0.128f },
        { -0.042f, 0.049f, 0.029f, 0.126f },
        { -0.001f, 0.037f, -0.11f, 0.176f },
        { 0.042f, 0.014f, 0.0f, 0.095f },
        { 0.015f, 0.015f, 0.075f, 0.072f },
        { 0.0f, 0.0f, 0.129f, 0.104f },
        { 0.054447f, 0.054447f, 0.1984f, 0.024f },
        { -0.054447f, -0.054447f, 0.1984f, 0.024f }
    };
    
    /* math utils */
    __device__ __forceinline__ constexpr float dot_2(const float &ax, const float &ay, const float &bx, const float &by) 
    {
        return (ax * bx) + (ay * by);
    }

     __device__ __forceinline__ constexpr float dot_3(
    const float ax,
    const float ay,
    const float az,
    const float bx,
    const float by,
    const float bz)
    {
        return ax * bx + ay * by + az * bz;
    }

    __device__ __forceinline__ constexpr float sql2_3(
        const float &ax,
        const float &ay,
        const float &az,
        const float &bx,
        const float &by,
        const float &bz)
    {
        const float xs = (ax - bx);
        const float ys = (ay - by);
        const float zs = (az - bz);

        return dot_3(xs, ys, zs, xs, ys, zs);
    }

    __device__ __forceinline__ constexpr float clamp(const float &v, const float &lower, const float &upper) 
    {
        return fmaxf(fminf(v, upper), lower);
    }
    /* end math utils */


    /* Sphere collision utils*/
    __device__ __forceinline__ constexpr float sphere_sphere_sql2(
        const float ax,
        const float ay,
        const float az,
        const float ar,
        const float bx,
        const float by,
        const float bz,
        const float br)
    {
        float sum = sql2_3(ax, ay, az, bx, by, bz);
        float rs = ar + br;
        return sum - rs * rs;
    }

    __device__ __forceinline__ constexpr float sphere_sphere_sql2(
        const Sphere<float> &a,
        const float &x,
        const float &y,
        const float &z,
        const float &r) 
    {
        return sphere_sphere_sql2(a.x, a.y, a.z, a.r, x, y, z, r);
    }

    __device__ __forceinline__ constexpr float sphere_sphere_self_collision(float ax, float ay, float az, float ar, float bx, float by, float bz, float br)
    {
        return (sphere_sphere_sql2(ax, ay, az, ar, bx, by, bz, br) < 0);
    }

    // returns l2 distance between two configs
    __device__ __forceinline__ float l2_dist(float *config_a, float *config_b, const int dim) {
        float ans = 0;
        float diff;
        #pragma unroll
        for (int i = 0; i < dim; i++) {
            diff = config_a[i] - config_b[i];
            ans += diff * diff;
        }
        return sqrt(ans);
    }

    __device__ __forceinline__ float sq_l2_dist(float *config_a, float *config_b, const int dim) {
        float ans = 0;
        float diff;
        #pragma unroll
        for (int i = 0; i < dim; i++) {
            diff = config_a[i] - config_b[i];
            ans += diff * diff;
        }
        return ans;
    }
    /* End Sphere collision utils*/

    /* Capsule collision utils */
    __device__ __forceinline__ constexpr float sphere_capsule(
        const Capsule<float> &c,
        const float &x,
        const float &y,
        const float &z,
        const float &r) noexcept
    {
        float dot = dot_3(x - c.x1, y - c.y1, z - c.z1, c.xv, c.yv, c.zv);
        float cdf = clamp((dot * c.rdv), 0.F, 1.F);

        float sum = sql2_3(x, y, z, c.x1 + c.xv * cdf, c.y1 + c.yv * cdf, c.z1 + c.zv * cdf);
        float rs = r + c.r;
        return sum - rs * rs;
    }

    
    __device__ __forceinline__ constexpr float sphere_capsule(const Capsule<float> &c, const Sphere<float> &s) noexcept 
    {
        return sphere_capsule(c, s.x, s.y, s.z, s.r);
    }

    
    __device__ __forceinline__ constexpr float sphere_z_aligned_capsule(
        const Capsule<float> &c,
        const float &x,
        const float &y,
        const float &z,
        const float &r) noexcept
    {
        float dot = (z - c.z1) * c.zv;
        float cdf = clamp((dot * c.rdv), 0.F, 1.F);

        float sum = sql2_3(x, y, z, c.x1, c.y1, c.z1 + c.zv * cdf);
        float rs = r + c.r;
        return sum - rs * rs;
    }

    __device__ __forceinline__ constexpr float sphere_z_aligned_capsule(const Capsule<float> &c, const Sphere<float> &s) noexcept
        
    {
        return sphere_z_aligned_capsule(c, s.x, s.y, s.z, s.r);
    }
    /* End Capsule collision utils*/

    /* Cuboid collision utils*/
    __device__ __forceinline__ constexpr float sphere_cuboid(
        const Cuboid<float> &c,
        const float &x,
        const float &y,
        const float &z,
        const float &rsq) noexcept
    {
        float xs = x - c.x;
        float ys = y - c.y;
        float zs = z - c.z;

        float a1 = fmaxf(0., abs(dot_3(c.axis_1_x, c.axis_1_y, c.axis_1_z, xs, ys, zs)) - c.axis_1_r);
        float a2 = fmaxf(0., abs(dot_3(c.axis_2_x, c.axis_2_y, c.axis_2_z, xs, ys, zs)) - c.axis_2_r);
        float a3 = fmaxf(0., abs(dot_3(c.axis_3_x, c.axis_3_y, c.axis_3_z, xs, ys, zs)) - c.axis_3_r);

        float sum = dot_3(a1, a2, a3, a1, a2, a3);
        return sum - rsq;
    }

    
    __device__ __forceinline__ constexpr float sphere_cuboid(const Cuboid<float> &c, const Sphere<float> &s) noexcept 
    {
        return sphere_cuboid(c, s.x, s.y, s.z, s.r * s.r);
    }

    
    __device__ __forceinline__ constexpr float sphere_z_aligned_cuboid(
        const Cuboid<float> &c,
        const float &x,
        const float &y,
        const float &z,
        const float &rsq) noexcept
    {
        float xs = x - c.x;
        float ys = y - c.y;
        float zs = z - c.z;

        float a1 = fmaxf(0., (abs(dot_2(c.axis_1_x, c.axis_1_y, xs, ys)) - c.axis_1_r));
        float a2 = fmaxf(0., (abs(dot_2(c.axis_2_x, c.axis_2_y, xs, ys)) - c.axis_2_r));
        float a3 = fmaxf(0, (abs(zs) - c.axis_3_r));

        float sum = dot_3(a1, a2, a3, a1, a2, a3);
        return sum - rsq;
    }

    
    __device__ __forceinline__ constexpr float sphere_z_aligned_cuboid(const Cuboid<float> &c, const Sphere<float> &s) noexcept
        
    {
        return sphere_z_aligned_cuboid(c, s.x, s.y, s.z, s.r * s.r);
    }
    /* End Cuboid collision util*/

    __device__ __forceinline__ bool sphere_environment_in_collision(ppln::collision::Environment<float> *env, float sx_, float sy_, float sz_, float sr_)
    {
        const float rsq = sr_ * sr_;
        bool in_collision = false;

        for (unsigned int i = 0; i < env->num_spheres && !in_collision; i++)
        {
            in_collision |= (sphere_sphere_sql2(env->spheres[i], sx_, sy_, sz_, sr_) < 0);
            
        }

        for (unsigned int i = 0; i < env->num_capsules && !in_collision; i++)
        {
            in_collision |= (sphere_capsule(env->capsules[i], sx_, sy_, sz_, sr_) < 0);
            
        }

        for (unsigned int i = 0; i < env->num_z_aligned_capsules && !in_collision; i++)
        {
            in_collision |= (sphere_z_aligned_capsule(env->z_aligned_capsules[i], sx_, sy_, sz_, sr_) < 0);
        }

        for (unsigned int i = 0; i < env->num_cuboids && !in_collision; i++)
        {
            in_collision |= (sphere_cuboid(env->cuboids[i], sx_, sy_, sz_, rsq) < 0);
            
        }

        for (unsigned int i = 0; i < env->num_z_aligned_cuboids && !in_collision; i++)
        {
            in_collision |= (sphere_z_aligned_cuboid(env->z_aligned_cuboids[i], sx_, sy_, sz_, rsq) < 0);
        }

        return in_collision;
    }

    __global__ void init_rng(curandState* states, unsigned long seed);
}


// Error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

inline void printCUDADeviceInfo() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "Error getting device count: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)\n\n";
    
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, device);
        
        if (error != cudaSuccess) {
            std::cerr << "Error getting device properties: " << cudaGetErrorString(error) << std::endl;
            continue;
        }
        
        std::cout << "Device " << device << ": " << prop.name << "\n";
        std::cout << "============================================\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
        
        // Memory Information
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB\n";
        std::cout << "Total Constant Memory: " << prop.totalConstMem / 1024.0 << " KB\n";
        std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
        std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024.0 << " KB\n";
        
        // Thread Information
        std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Max Threads Per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "Number of Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "Warp Size: " << prop.warpSize << "\n";
        
        // Block Dimensions
        std::cout << "Max Block Dimensions: " 
                << prop.maxThreadsDim[0] << " x "
                << prop.maxThreadsDim[1] << " x "
                << prop.maxThreadsDim[2] << "\n";
        
        // Grid Dimensions
        std::cout << "Max Grid Dimensions: "
                << prop.maxGridSize[0] << " x "
                << prop.maxGridSize[1] << " x "
                << prop.maxGridSize[2] << "\n";
        
        // Memory Clock and Bus Width
        std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000.0 << " MHz\n";
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
        
        // Additional Features
        std::cout << "Unified Addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << "\n";
        std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << "\n";
        std::cout << "ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << "\n";
        
        std::cout << "\n";
    }
}


/* Collision checking backend implementations for different robots */
namespace ppln::collision {
    using namespace device_utils;
    
    // fkcc -> checks if the config is "good"
    // cc returns false if the config does collide with an obstacle, returns true if the config does not collide

    template <typename Robot>
    __device__ __forceinline__ void fk(const float *config, volatile float* sphere_pos, float *T, const int tid);

    template <typename Robot>
    __device__ __forceinline__ bool self_collision_check(volatile float* sphere_pos, volatile int* link_approx_CC, const int tid);

    template <typename Robot>
    __device__ __forceinline__ bool env_collision_check(volatile float* sphere_pos, volatile int* link_approx_CC, ppln::collision::Environment<float> *env, const int tid);

    template <typename Robot>
    __device__ __forceinline__ void fk_approx(const float *config, volatile float* sphere_pos_approx, float *T, const int tid);

    template <typename Robot>
    __device__ __forceinline__ bool self_collision_check_approx(volatile float* sphere_pos_approx, volatile int* link_approx_CC, const int tid);

    template <typename Robot>
    __device__ __forceinline__ bool env_collision_check_approx(volatile float* sphere_pos_approx, volatile int* link_approx_CC, ppln::collision::Environment<float> *env, const int tid);

    template <typename Robot>
    __device__ __forceinline__ bool fkcc(volatile float *config, ppln::collision::Environment<float> *env, int tid);


    __device__ __forceinline__ void matmul(const float A[4][4], const float B[4][4], float C[4][4]) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                C[r][c] = 0;
                for (int k = 0; k < 4; ++k)
                    C[r][c] += A[r][k] * B[k][c];
            }
        }
    }

    
    /* panda approx constants */
    __device__ __constant__ float4 panda_approx_spheres_array[11] = {
        { 0.0f, 0.0f, 0.05f, 0.08f },
        { -0.001f, -0.039f, -0.085f, 0.154f },
        { 0.0f, -0.085f, 0.04f, 0.154f },
        { 0.039f, 0.028f, -0.052f, 0.128f },
        { -0.042f, 0.049f, 0.029f, 0.126f },
        { -0.001f, 0.037f, -0.11f, 0.176f },
        { 0.042f, 0.014f, 0.0f, 0.095f },
        { 0.015f, 0.015f, 0.075f, 0.072f },
        { 0.0f, 0.0f, 0.129f, 0.104f },
        { 0.054447f, 0.054447f, 0.1984f, 0.024f },
        { -0.054447f, -0.054447f, 0.1984f, 0.024f }
    };
    
    __device__ __constant__ int panda_approx_self_cc_ranges[4][3] = {
        { 0, 5, 10 },
        { 1, 5, 10 },
        { 2, 5, 5 },
        { 2, 7, 10 }
    };
    
    
    
    __device__ __constant__ int panda_approx_sphere_to_joint[] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        7,
        7,
        7
    };
    
    __device__ __constant__ int panda_approx_joint_types[] = {
        0,
        2,
        2,
        2,
        2,
        2,
        2,
        2
    };
    /* end panda approx constants */

    /* panda full constants */
    __device__ __constant__ float4 panda_spheres_array[59] = {
        { 0.0f, 0.0f, 0.05f, 0.08f },
        { 0.0f, -0.08f, 0.0f, 0.06f },
        { 0.0f, -0.03f, 0.0f, 0.06f },
        { 0.0f, 0.0f, -0.12f, 0.06f },
        { 0.0f, 0.0f, -0.17f, 0.06f },
        { 0.0f, 0.0f, 0.03f, 0.06f },
        { 0.0f, 0.0f, 0.08f, 0.06f },
        { 0.0f, -0.12f, 0.0f, 0.06f },
        { 0.0f, -0.17f, 0.0f, 0.06f },
        { 0.0f, 0.0f, -0.1f, 0.06f },
        { 0.0f, 0.0f, -0.06f, 0.05f },
        { 0.08f, 0.06f, 0.0f, 0.055f },
        { 0.08f, 0.02f, 0.0f, 0.055f },
        { -0.08f, 0.095f, 0.0f, 0.06f },
        { 0.0f, 0.0f, 0.02f, 0.055f },
        { 0.0f, 0.0f, 0.06f, 0.055f },
        { -0.08f, 0.06f, 0.0f, 0.055f },
        { 0.0f, 0.055f, 0.0f, 0.06f },
        { 0.0f, 0.075f, 0.0f, 0.06f },
        { 0.0f, 0.0f, -0.22f, 0.06f },
        { 0.0f, 0.05f, -0.18f, 0.05f },
        { 0.01f, 0.08f, -0.14f, 0.025f },
        { 0.01f, 0.085f, -0.11f, 0.025f },
        { 0.01f, 0.09f, -0.08f, 0.025f },
        { 0.01f, 0.095f, -0.05f, 0.025f },
        { -0.01f, 0.08f, -0.14f, 0.025f },
        { -0.01f, 0.085f, -0.11f, 0.025f },
        { -0.01f, 0.09f, -0.08f, 0.025f },
        { -0.01f, 0.095f, -0.05f, 0.025f },
        { 0.0f, 0.0f, 0.0f, 0.05f },
        { 0.08f, -0.01f, 0.0f, 0.05f },
        { 0.08f, 0.035f, 0.0f, 0.052f },
        { 0.0f, 0.0f, 0.07f, 0.05f },
        { 0.02f, 0.04f, 0.08f, 0.025f },
        { 0.04f, 0.02f, 0.08f, 0.025f },
        { 0.04f, 0.06f, 0.085f, 0.02f },
        { 0.06f, 0.04f, 0.085f, 0.02f },
        { -0.053033f, -0.053033f, 0.117f, 0.028f },
        { -0.03182f, -0.03182f, 0.117f, 0.028f },
        { -0.010607f, -0.010607f, 0.117f, 0.028f },
        { 0.010607f, 0.010607f, 0.117f, 0.028f },
        { 0.03182f, 0.03182f, 0.117f, 0.028f },
        { 0.053033f, 0.053033f, 0.117f, 0.028f },
        { -0.053033f, -0.053033f, 0.137f, 0.026f },
        { -0.03182f, -0.03182f, 0.137f, 0.026f },
        { -0.010607f, -0.010607f, 0.137f, 0.026f },
        { 0.010607f, 0.010607f, 0.137f, 0.026f },
        { 0.03182f, 0.03182f, 0.137f, 0.026f },
        { 0.053033f, 0.053033f, 0.137f, 0.026f },
        { -0.053033f, -0.053033f, 0.157f, 0.024f },
        { -0.03182f, -0.03182f, 0.157f, 0.024f },
        { -0.010607f, -0.010607f, 0.157f, 0.024f },
        { 0.010607f, 0.010607f, 0.157f, 0.024f },
        { 0.03182f, 0.03182f, 0.157f, 0.024f },
        { 0.053033f, 0.053033f, 0.157f, 0.024f },
        { 0.056569f, 0.056569f, 0.1874f, 0.012f },
        { 0.051619f, 0.051619f, 0.2094f, 0.012f },
        { -0.056569f, -0.056569f, 0.1874f, 0.012f },
        { -0.051619f, -0.051619f, 0.2094f, 0.012f }
    };








    __device__ __constant__ int panda_self_cc_ranges[24][3] = {
        { 0, 17, 58 },
        { 1, 17, 58 },
        { 2, 17, 58 },
        { 3, 17, 58 },
        { 4, 17, 58 },
        { 5, 17, 28 },
        { 5, 32, 58 },
        { 6, 17, 28 },
        { 6, 32, 58 },
        { 7, 17, 28 },
        { 7, 32, 58 },
        { 8, 17, 28 },
        { 8, 32, 58 },
        { 17, 32, 58 },
        { 18, 32, 58 },
        { 19, 32, 58 },
        { 20, 32, 58 },
        { 21, 32, 58 },
        { 22, 32, 58 },
        { 23, 32, 58 },
        { 24, 32, 58 },
        { 25, 32, 58 },
        { 26, 32, 58 },
        { 27, 32, 58 }
    };


    __device__ __constant__ float panda_fixed_transforms[] = {
        // joint 0
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 1
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.333,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 2
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 3
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, -0.316,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 4
        1.0, 0.0, 0.0, 0.0825,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 5
        1.0, 0.0, 0.0, -0.0825,
        0.0, 0.0, 1.0, 0.384,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 6
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 7
        1.0, 0.0, 0.0, 0.088,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        
    };

    __device__ __constant__ int panda_sphere_to_joint[] = {
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7
    };

    __device__ __constant__ int panda_joint_types[] = {
        0,
        2,
        2,
        2,
        2,
        2,
        2,
        2
    };
    /* end panda full constants */
    

    __device__ __forceinline__ void fixed_joint_fn(
        const float *fixed_transform,
        float *T_step_col
    )
    {
        T_step_col[0] = fixed_transform[0];
        T_step_col[1] = fixed_transform[M];
        T_step_col[2] = fixed_transform[M * 2];
        T_step_col[3] = fixed_transform[M * 3];
    }

    __device__ __forceinline__ void xrot_fn(
        const float *fixed_transforms,
        const float angle,
        const int col_idx,
        float *T_step_col
    )
    {
      // we found no change in convergence between fast approximate and IEEE sin,
      // cos functions using fast approximate method saves 5 registers per thread.
      float cos   = __cosf(angle);
      float sin   = __sinf(angle);
      float n_sin = -1 * sin;

      int bit1         = col_idx & 0x1;
      int bit2         = (col_idx & 0x2) >> 1;
      int _xor         = bit1 ^ bit2;  // 0 for threads 0 and 3, 1 for threads 1 and 2
      int col_idx_by_2 =
        col_idx / 2;                   // 0 for threads 0 and 1, 1 for threads 2 and 3

      float f1 = (1 - col_idx_by_2) * cos +
                 col_idx_by_2 * n_sin; // thread 1 get cos , thread 2 gets n_sin
      float f2 = (1 - col_idx_by_2) * sin +
                 col_idx_by_2 * cos;   // thread 1 get sin, thread 2 gets cos

      f1 = _xor * f1 + (1 - _xor) * 1; // threads 1 and 2 will get f1; the other
                                       // two threads will get 1
      f2 = _xor *
           f2;                         // threads 1 and 2 will get f2, the other two threads will
                                       // get 0.0
      float f3 = 1 - _xor;

      int addr_offset =
        _xor + (1 - _xor) *
        col_idx; // 1 for threads 1 and 2, col_idx for threads 0 and 3

      T_step_col[0] = fixed_transforms[0 + addr_offset] * f1 + f2 * fixed_transforms[2];
      T_step_col[1] = fixed_transforms[M + addr_offset] * f1 + f2 * fixed_transforms[M + 2];
      T_step_col[2] =
        fixed_transforms[M + M + addr_offset] * f1 + f2 * fixed_transforms[M + M + 2];
      T_step_col[3] = fixed_transforms[M + M + M + addr_offset] *
              f3; // threads 1 and 2 get 0.0, remaining two get fixed_transforms[3M];
    }

    // version with no control flow
    __device__ __forceinline__ void yrot_fn(
        const float *fixed_transforms,
        const float angle,
        const int col_idx,
        float *T_step_col
    )
    {
      float cos   = __cosf(angle);
      float sin   = __sinf(angle);
      float n_sin = -1 * sin;

      int col_idx_per_2 =
        col_idx % 2;                 // threads 0 and 2 will be 0 and threads 1 and 3 will be 1.
      int col_idx_by_2 =
        col_idx / 2;                 // threads 0 and 1 will be 0 and threads 2 and 3 will be 1.

      float f1 = (1 - col_idx_by_2) * cos +
                 col_idx_by_2 * sin; // thread 0 get cos , thread 2 gets sin
      float f2 = (1 - col_idx_by_2) * n_sin +
                 col_idx_by_2 * cos; // thread 0 get n_sin, thread 2 gets cos

      f1 = (1 - col_idx_per_2) * f1 +
           col_idx_per_2 * 1;        // threads 0 and 2 will get f1; the other two
                                     // threads will get 1
      f2 = (1 - col_idx_per_2) *
           f2;                       // threads 0 and 2 will get f2, the other two threads will get
                                     // 0.0
      float f3 =
        col_idx_per_2;               // threads 0 and 2 will be 0 and threads 1 and 3 will be 1.

      int addr_offset =
        col_idx_per_2 *
        col_idx; // threads 0 and 2 will get 0, the other two will get col_idx.

      T_step_col[0] = fixed_transforms[0 + addr_offset] * f1 + f2 * fixed_transforms[2];
      T_step_col[1] = fixed_transforms[M + addr_offset] * f1 + f2 * fixed_transforms[M + 2];
      T_step_col[2] =
        fixed_transforms[M + M + addr_offset] * f1 + f2 * fixed_transforms[M + M + 2];
      T_step_col[3] = fixed_transforms[M + M + M + addr_offset] *
              f3; // threads 0 and 2 threads get 0.0, remaining two get
                  // fixed_transforms[3M];
    }
    
    __device__ __forceinline__ void zrot_fn(
        const float *fixed_transforms,
        const float angle,
        const int col_idx,
        float *T_step_col
    ) {
        float cos = __cosf(angle);
        float sin = __sinf(angle);
        float n_sin = -1 * sin;

        int col_idx_by_2 =
            col_idx / 2; // first two threads will be 0 and the next two will be 1.
        int col_idx_per_2 =
            col_idx % 2; // first thread will be 0 and the second thread will be 1.
        float f1 = (1 - col_idx_per_2) * cos +
                   col_idx_per_2 * n_sin; // thread 0 get cos , thread 1 gets n_sin
        float f2 = (1 - col_idx_per_2) * sin +
                   col_idx_per_2 * cos; // thread 0 get sin, thread 1 gets cos

        f1 = (1 - col_idx_by_2) * f1 +
             col_idx_by_2 * 1; // first two threads get f1, other two threads get 1
        f2 = (1 - col_idx_by_2) *
             f2; // first two threads get f2, other two threads get 0.0

        int addr_offset =
            col_idx_by_2 *
            col_idx; // first 2 threads will get 0, the other two will get col_idx.

        T_step_col[0] = fixed_transforms[0 + addr_offset] * f1 + f2 * fixed_transforms[1];
        T_step_col[1] = fixed_transforms[M + addr_offset] * f1 + f2 * fixed_transforms[M + 1];
        T_step_col[2] = fixed_transforms[M + M + addr_offset] * f1 + f2 * fixed_transforms[M + M + 1];
        T_step_col[3] = fixed_transforms[M + M + M + addr_offset] * col_idx_by_2; // first two threads get 0.0, remaining two get fixed_transforms[3M];
    }
    
    __device__ __forceinline__ float dot4(float *a, float *b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    }

    __device__ __forceinline__ float dot4_col(float *a_col, float *b) {
        return a_col[0] * b[0] + a_col[M] * b[1] + a_col[M*2] * b[2] + a_col[M*3] * b[3];
    }

    template <>
    __device__ void fk_approx<ppln::robots::Panda>(
        const float* q,
        volatile float* sphere_pos_approx, // 11 spheres x 16 robots x 3 coordinates (each column is a robot)
        float *T, // 16 robots x 4x4 transform matrix , column major
        const int tid
    ) {
        // every 4 threads are responsible for one column of the transform matrix T
        // make_transform will calculate the necessary column of T_step needed for the thread
        const int col_ind = tid%4;
        const int batch_ind = tid/4;
        int transformed_sphere_ind = 0;

        int T_offset = batch_ind * 16;
        float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
        float *T_base = T + T_offset; // 4x4 transform matrix for the batch
        float *T_col = T_base + col_ind*4; // 1x4 column (column major) of the transform matrix for this thread

        for (int r=0; r<4; r++){
            T_col[r] = 0;
        }
        T_col[col_ind] = 1;

        if (col_ind == 0) {
            sphere_pos_approx[batch_ind * 16 * 3 + 0] = 0;
            sphere_pos_approx[batch_ind * 16 * 3 + 1] = 0;
            sphere_pos_approx[batch_ind * 16 * 3 + 2] = 0.05;
        }
        transformed_sphere_ind++;

        // loop through each joint, accumulate transformation matrix, and update sphere positions
        for (int i = 1; i < 8; ++i) {
            // printf("i: %d, q[i - 1]: %f\n", i, q[i - 1]);
            int ft_addr_start = i * 16;
            if (panda_joint_types[i] == 0) {
                xrot_fn(&panda_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }
            else if (panda_joint_types[i] == 1) {
                yrot_fn(&panda_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }
            else if (panda_joint_types[i] == 2) {
                zrot_fn(&panda_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }

            for (int r=0; r<4; r++){
                T_col[r] = dot4_col(&T_base[r], T_step_col);
            }

            __syncthreads();
            // if (tid == 0) {
            //     printf("approx i: %d, q[i]: %f, T: %f %f %f %f, %f %f %f %f, %f %f %f %f, %f %f %f %f\n", i, q[i], T_base[0], T_base[1], T_base[2], T_base[3], T_base[4], T_base[5], T_base[6], T_base[7], T_base[8], T_base[9], T_base[10], T_base[11], T_base[12], T_base[13], T_base[14], T_base[15]);
            // }

            while (panda_approx_sphere_to_joint[transformed_sphere_ind] == i) {
                if (col_ind < 3) {
                    // sphere transformed_sphere_ind, robot batch_ind (16 robots), coord col_ind
                    sphere_pos_approx[transformed_sphere_ind * 16 * 3 + batch_ind * 3 + col_ind] = 
                        T_base[col_ind] * panda_approx_spheres_array[transformed_sphere_ind].x +
                        T_base[col_ind + M] * panda_approx_spheres_array[transformed_sphere_ind].y +
                        T_base[col_ind + M*2] * panda_approx_spheres_array[transformed_sphere_ind].z +
                        T_base[col_ind + M*3];
                }
                transformed_sphere_ind++;
            }
        }
    }

    template <>
    __device__ void fk<ppln::robots::Panda>(
        const float* q,
        volatile float* sphere_pos, // 16 robots x 59 spheres x 3 coordinates
        float *T, // 16 robots x 4x4 transform matrix
        const int tid
    ) {
        // every 4 threads are responsible for one column of the transform matrix T
        // make_transform will calculate the necessary column of T_step needed for the thread
        const int col_ind = tid%4;
        const int batch_ind = tid/4;
        int transformed_sphere_ind = 0;

        int T_offset = batch_ind * 16;
        float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
        float *T_base = T + T_offset; // 4x4 transform matrix for the batch
        float *T_col = T_base + col_ind*4; // 1x4 column (column major) of the transform matrix for this thread

        for (int r=0; r<4; r++){
            T_col[r] = 0;
        }
        T_col[col_ind] = 1;

        if (col_ind == 0) {
           sphere_pos[batch_ind * 16 * 3 + 0] = 0;
           sphere_pos[batch_ind * 16 * 3 + 1] = 0;
           sphere_pos[batch_ind * 16 * 3 + 2] = 0.05;
        }
        transformed_sphere_ind++;

        // loop through each joint, accumulate transformation matrix, and update sphere positions
        for (int i = 1; i < 8; ++i) {
            int ft_addr_start = i * 16;
            if (panda_joint_types[i] == 0) {
                xrot_fn(&panda_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }
            else if (panda_joint_types[i] == 1) {
                yrot_fn(&panda_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }
            else if (panda_joint_types[i] == 2) {
                zrot_fn(&panda_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }

            for (int r=0; r<4; r++){
                T_col[r] = dot4_col(&T_base[r], T_step_col);
            }

            // __syncthreads();
            // if (tid == 0) {
            //     printf("i: %d, q[i]: %f, T: %f %f %f %f, %f %f %f %f, %f %f %f %f, %f %f %f %f\n", i, q[i], T_base[0], T_base[1], T_base[2], T_base[3], T_base[4], T_base[5], T_base[6], T_base[7], T_base[8], T_base[9], T_base[10], T_base[11], T_base[12], T_base[13], T_base[14], T_base[15]);
            // }

            while (panda_sphere_to_joint[transformed_sphere_ind]==i) {
                if (col_ind < 3) {
                    // sphere transformed_sphere_ind, robot batch_ind (16 robots), coord col_ind
                    sphere_pos[transformed_sphere_ind * 16 * 3 + batch_ind * 3 + col_ind] = 
                        T_base[col_ind] * panda_spheres_array[transformed_sphere_ind].x +
                        T_base[col_ind + M] * panda_spheres_array[transformed_sphere_ind].y +
                        T_base[col_ind + M*2] * panda_spheres_array[transformed_sphere_ind].z +
                        T_base[col_ind + M*3];
                }
                transformed_sphere_ind++;
            }
        }
    }

    

    // 4 threads per discretized motion for self-collision check
    template <>
    __device__ bool self_collision_check_approx<ppln::robots::Panda>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, const int tid){
        const int thread_ind = tid%4;
        const int batch_ind = tid/4;

        // for (int i = thread_ind; i < 21; i+=4) {
        //     int sphere_1_ind = panda_approx_self_cc_pairs[i][0];
        //     int sphere_2_ind = panda_approx_self_cc_pairs[i][1];
        //     if (sphere_sphere_self_collision(
        //         sphere_pos_approx[sphere_1_ind * 16 * 3 + batch_ind * 3 + 0],
        //         sphere_pos_approx[sphere_1_ind * 16 * 3 + batch_ind * 3 + 1],
        //         sphere_pos_approx[sphere_1_ind * 16 * 3 + batch_ind * 3 + 2],
        //         panda_approx_spheres_array[sphere_1_ind].w,
        //         sphere_pos_approx[sphere_2_ind * 16 * 3 + batch_ind * 3 + 0],
        //         sphere_pos_approx[sphere_2_ind * 16 * 3 + batch_ind * 3 + 1],
        //         sphere_pos_approx[sphere_2_ind * 16 * 3 + batch_ind * 3 + 2],
        //         panda_approx_spheres_array[sphere_2_ind].w
        //     )) {
        //         // if (tid == 0) {
        //         //     printf("collision between %d and %d\n", sphere_1_ind, sphere_2_ind);
        //         //     printf("sphere %d: %f %f %f\n", sphere_1_ind, sphere_pos_approx[sphere_1_ind * 16 * 3 + batch_ind * 3 + 0], sphere_pos_approx[sphere_1_ind * 16 * 3 + batch_ind * 3 + 1], sphere_pos_approx[sphere_1_ind * 16 * 3 + batch_ind * 3 + 2]);
        //         //     printf("sphere %d: %f %f %f\n", sphere_2_ind, sphere_pos_approx[sphere_2_ind * 16 * 3 + batch_ind * 3 + 0], sphere_pos_approx[sphere_2_ind * 16 * 3 + batch_ind * 3 + 1], sphere_pos_approx[sphere_2_ind * 16 * 3 + batch_ind * 3 + 2]);
        //         // }
        //         atomicAdd((int*)&joint_in_collision[20*batch_ind + panda_approx_sphere_to_joint[sphere_1_ind]], 1);
        //         // atomicAdd((int*)&joint_in_collision[20*batch_ind + panda_approx_sphere_to_joint[sphere_2_ind]], 1);
        //         return false;
        //     }
        // }
        // return true;

        for (int i = thread_ind; i < 4; i+=4) {
            int sphere_1_ind = panda_approx_self_cc_ranges[i][0];
            float sphere_1[3] = {
                sphere_pos_approx[sphere_1_ind * 16 * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[sphere_1_ind * 16 * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[sphere_1_ind * 16 * 3 + batch_ind * 3 + 2]
            };
            for (int j = panda_approx_self_cc_ranges[i][1]; j <= panda_approx_self_cc_ranges[i][2]; j++) {
                float sphere_2[3] = {
                    sphere_pos_approx[j * 16 * 3 + batch_ind * 3 + 0],
                    sphere_pos_approx[j * 16 * 3 + batch_ind * 3 + 1],
                    sphere_pos_approx[j * 16 * 3 + batch_ind * 3 + 2]
                };
                if (sphere_sphere_self_collision(
                    sphere_1[0], sphere_1[1], sphere_1[2], panda_approx_spheres_array[sphere_1_ind].w,
                    sphere_2[0], sphere_2[1], sphere_2[2], panda_approx_spheres_array[j].w
                )){
                    atomicAdd((int*)&joint_in_collision[20*batch_ind + panda_approx_sphere_to_joint[sphere_1_ind]], 1);
                    return false;
                }
            } 
        }
        return true;
    }

    // 4 threads per discretized motion for self-collision check
    template <>
    __device__ bool self_collision_check<ppln::robots::Panda>(volatile float* sphere_pos, volatile int* joint_in_collision, const int tid){
        const int thread_ind = tid%4;
        const int batch_ind = tid/4;

        // add approximate link check (self-collision)
        
        // #pragma unroll
        // for (int i=thread_ind; i<PANDA_STOP_SELF_CC; i+=4){
        //     if (joint_in_collision[20*batch_ind +panda_link_index[i]]==0) continue;
        //     // sphere i, robot batch_ind (16 robots)
        //     float check_sphere[3] = {sphere_pos[i * 16 * 3 + batch_ind * 3 + 0], sphere_pos[i * 16 * 3 + batch_ind * 3 + 1], sphere_pos[i * 16 * 3 + batch_ind * 3 + 2]};
            
        //     for (int j=panda_startCC_ind[i]; j<PANDA_SPHERE_COUNT; j++){
        //         //if (panda_link_index[i]==2 && panda_link_index[j]==6) continue;
        //         // sphere j, robot batch_ind (16 robots)
        //         float check_pos[3] = {sphere_pos[j * 16 * 3 + batch_ind * 3 + 0], sphere_pos[j * 16 * 3 + batch_ind * 3 + 1], sphere_pos[j * 16 * 3 + batch_ind * 3 + 2]};
        //         if (sphere_sphere_self_collision(check_sphere[0], check_sphere[1], check_sphere[2], panda_local_xyz[i].w, check_pos[0], check_pos[1], check_pos[2], panda_local_xyz[j].w)){ 
        //             return false;
        //         }
        //     }
        // }
        // return true;



        for (int i = thread_ind; i < 24; i+=4) {
            int sphere_1_ind = panda_self_cc_ranges[i][0];
            if (joint_in_collision[20*batch_ind + panda_sphere_to_joint[sphere_1_ind]] == 0) continue;
            float sphere_1[3] = {
                sphere_pos[sphere_1_ind * 16 * 3 + batch_ind * 3 + 0],
                sphere_pos[sphere_1_ind * 16 * 3 + batch_ind * 3 + 1],
                sphere_pos[sphere_1_ind * 16 * 3 + batch_ind * 3 + 2]
            };
            for (int j = panda_self_cc_ranges[i][1]; j <= panda_self_cc_ranges[i][2]; j++) {
                float sphere_2[3] = {
                    sphere_pos[j * 16 * 3 + batch_ind * 3 + 0],
                    sphere_pos[j * 16 * 3 + batch_ind * 3 + 1],
                    sphere_pos[j * 16 * 3 + batch_ind * 3 + 2]
                };
                if (sphere_sphere_self_collision(
                    sphere_1[0], sphere_1[1], sphere_1[2], panda_spheres_array[sphere_1_ind].w,
                    sphere_2[0], sphere_2[1], sphere_2[2], panda_spheres_array[j].w
                )){
                    return false;
                }
            }
        }
        return true;

    }

    // 4 threads per discretized motion for env collision check
    template <>
    __device__ bool env_collision_check<ppln::robots::Panda>(volatile float* sphere_pos, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
        const int thread_ind = tid%4;
        const int batch_ind = tid/4;

        
        // env check
        #pragma unroll
        // int i=PANDA_SPHERE_COUNT/4*thread_ind; i<PANDA_SPHERE_COUNT/4*(thread_ind+1); i++
        for (int i=thread_ind; i<PANDA_SPHERE_COUNT; i+=4){
            // sphere i, robot batch_ind (16 robots)
            if (joint_in_collision[20*batch_ind +panda_sphere_to_joint[i]] > 0 && 
                sphere_environment_in_collision(
                    env,
                    sphere_pos[i * 16 * 3 + batch_ind * 3 + 0],
                    sphere_pos[i * 16 * 3 + batch_ind * 3 + 1],
                    sphere_pos[i * 16 * 3 + batch_ind * 3 + 2],
                    panda_spheres_array[i].w
                )
            ) {
                return false;
            } 
        }

        //int i = PANDA_SPHERE_COUNT-1-thread_ind;
        //if (link_approx_CC[20*batch_ind +panda_link_index[i]]>0 && sphere_environment_in_collision(env, sphere_pos[batch_ind * PANDA_SPHERE_COUNT * 3+i*3], sphere_pos[batch_ind * PANDA_SPHERE_COUNT * 3+i*3+1], sphere_pos[batch_ind * PANDA_SPHERE_COUNT * 3+i*3+2], panda_local_xyz[i].w)){
        //    return false;
        //}
        return true;
    }

    // 4 threads per discretized motion for env collision check
    template <>
    __device__ bool env_collision_check_approx<ppln::robots::Panda>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
        const int thread_ind = tid%4;
        const int batch_ind = tid/4;
        bool out=true;
        // env check
        
        #pragma unroll
        for (int i=PANDA_APPROX_SPHERE_COUNT/4*thread_ind; i<PANDA_APPROX_SPHERE_COUNT/4*(thread_ind+1); i++){
            // sphere i, robot batch_ind (16 robots)
            if (sphere_environment_in_collision(
                env,
                sphere_pos_approx[i * 16 * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[i * 16 * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[i * 16 * 3 + batch_ind * 3 + 2],
                panda_approx_spheres_array[i].w
            )) {
                atomicAdd((int*)&joint_in_collision[20*batch_ind +panda_approx_sphere_to_joint[i]],1);
                out=false;
            } 
        }

        int i = PANDA_APPROX_SPHERE_COUNT-1-thread_ind;
        if (sphere_environment_in_collision(
            env,
            sphere_pos_approx[i * 16 * 3 + batch_ind * 3 + 0],
            sphere_pos_approx[i * 16 * 3 + batch_ind * 3 + 1],
            sphere_pos_approx[i * 16 * 3 + batch_ind * 3 + 2],
            panda_approx_spheres_array[i].w
        )) {
            atomicAdd((int*)&joint_in_collision[20*batch_ind +panda_approx_sphere_to_joint[i]],1);
            out=false;
        }
        return out;
    }


    template <>
    __device__  __forceinline__ bool fkcc<ppln::robots::Panda>(volatile float *q, ppln::collision::Environment<float> *environment, int tid)
    {
        // Ignore static frame collisions - needed for some evaluation problems
        // if (/*panda_link0*/ sphere_environment_in_collision(environment, 0.0, 0.0, 0.05, 0.08))
        // {
        //     return false;
        // }  // (0, 0)
        //printf("approx link 0: %f %f %f\n", 0.0, 0.0, 0.05);
        float INPUT_0 = q[0];
        float DIV_8 = INPUT_0 * 0.5;
        float SIN_9 = sin(DIV_8);
        float COS_15 = cos(DIV_8);
        float MUL_1575 = COS_15 * SIN_9;
        float MUL_1594 = MUL_1575 * 2.0;
        float MUL_1625 = MUL_1594 * 0.039;
        float MUL_1574 = SIN_9 * SIN_9;
        float MUL_1584 = MUL_1574 * 2.0;
        float SUB_1587 = 1.0 - MUL_1584;
        float MUL_1614 = SUB_1587 * 0.001;
        float SUB_1641 = MUL_1625 - MUL_1614; // intermediate
        float MUL_1628 = SUB_1587 * 0.039;
        float MUL_1618 = MUL_1594 * 0.001;
        float ADD_1642 = MUL_1618 + MUL_1628;
        float NEGATE_1643 = -ADD_1642; // intermediate
        float MUL_1656 = MUL_1594 * 0.08; // intermediate
        float MUL_1659 = SUB_1587 * 0.08;
        float NEGATE_1660 = -MUL_1659; // intermediate
        float MUL_1680 = MUL_1594 * 0.03;  // intermediate
        float MUL_1683 = SUB_1587 * 0.03;
        float NEGATE_1684 = -MUL_1683; // intermediate
        // var_cache[tid][0] = SUB_1641;
        // var_cache[tid][1] = NEGATE_1643;
        // var_cache[tid][2] = MUL_1656;
        // var_cache[tid][3] = NEGATE_1660;
        // var_cache[tid][4] = MUL_1680;
        // var_cache[tid][5] = NEGATE_1684;
        // if (tid%32 == 0) printf("here fkcc: %d", tid);
        if (/*panda_link1*/ sphere_environment_in_collision(environment, SUB_1641, NEGATE_1643, 0.248, 0.154))
        {
            if (sphere_environment_in_collision(environment, MUL_1656, NEGATE_1660, 0.333, 0.06))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, MUL_1680, NEGATE_1684, 0.333, 0.06))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, 0.0, 0.0, 0.213, 0.06))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, 0.0, 0.0, 0.163, 0.06))
            {
                return false;
            }
        }  // (0, 22)
        printf("approx link 1: %f %f %f\n", SUB_1641, NEGATE_1643, 0.248);
        float MUL_74 = COS_15 * 0.7071068;
        float MUL_72 = SIN_9 * 0.7071068;
        float INPUT_1 = q[1];
        float DIV_117 = INPUT_1 * 0.5;
        float SIN_118 = sin(DIV_117);
        float COS_124 = cos(DIV_117);
        float MUL_143 = MUL_72 * COS_124;
        float MUL_128 = MUL_72 * SIN_118;
        float MUL_126 = MUL_74 * COS_124;
        float SUB_149 = MUL_126 - MUL_128;
        float ADD_130 = MUL_126 + MUL_128;
        float MUL_140 = MUL_74 * SIN_118;
        float SUB_138 = MUL_140 - MUL_143;
        float ADD_144 = MUL_140 + MUL_143;
        float MUL_1756 = SUB_149 * ADD_144;
        float MUL_1757 = SUB_149 * SUB_138;
        float MUL_1763 = ADD_130 * ADD_144;
        float SUB_1796 = MUL_1757 - MUL_1763;
        float MUL_1798 = SUB_1796 * 2.0;
        float MUL_1827 = MUL_1798 * 0.04;
        float MUL_1761 = ADD_130 * SUB_138;
        float ADD_1781 = MUL_1761 + MUL_1756;
        float MUL_1784 = ADD_1781 * 2.0;
        float MUL_1817 = MUL_1784 * 0.085;
        float ADD_1832 = MUL_1817 + MUL_1827;
        float MUL_1759 = SUB_149 * ADD_130;
        float MUL_1755 = ADD_144 * ADD_144;
        float MUL_1765 = SUB_138 * ADD_144;
        float ADD_1799 = MUL_1765 + MUL_1759;
        float MUL_1801 = ADD_1799 * 2.0;
        float MUL_1829 = MUL_1801 * 0.04;
        float MUL_1758 = ADD_130 * ADD_130;
        float ADD_1786 = MUL_1755 + MUL_1758;
        float MUL_1789 = ADD_1786 * 2.0;
        float SUB_1792 = 1.0 - MUL_1789;
        float MUL_1820 = SUB_1792 * 0.085;
        float SUB_1833 = MUL_1829 - MUL_1820;
        float SUB_1793 = MUL_1765 - MUL_1759;
        float MUL_1795 = SUB_1793 * 2.0;
        float MUL_1824 = MUL_1795 * 0.085;
        float MUL_1754 = SUB_138 * SUB_138;
        float ADD_1802 = MUL_1754 + MUL_1758;
        float MUL_1805 = ADD_1802 * 2.0;
        float SUB_1808 = 1.0 - MUL_1805;
        float MUL_1831 = SUB_1808 * 0.04;
        float SUB_1834 = MUL_1831 - MUL_1824;
        float ADD_1835 = 0.333 + SUB_1834;
        float MUL_1849 = MUL_1798 * 0.03;
        float MUL_1851 = MUL_1801 * 0.03;
        float MUL_1853 = SUB_1808 * 0.03;
        float ADD_1854 = 0.333 + MUL_1853;
        float MUL_1868 = MUL_1798 * 0.08;
        float MUL_1870 = MUL_1801 * 0.08;
        float MUL_1872 = SUB_1808 * 0.08;
        float ADD_1873 = 0.333 + MUL_1872;
        float MUL_1882 = MUL_1784 * 0.12;
        float MUL_1885 = SUB_1792 * 0.12;
        float NEGATE_1886 = -MUL_1885;
        float MUL_1889 = MUL_1795 * 0.12;
        float SUB_1897 = 0.333 - MUL_1889;
        float MUL_1906 = MUL_1784 * 0.17;
        float MUL_1909 = SUB_1792 * 0.17;
        float NEGATE_1910 = -MUL_1909;
        float MUL_1913 = MUL_1795 * 0.17;
        float SUB_1921 = 0.333 - MUL_1913;
        if (/*panda_link2*/ sphere_environment_in_collision(environment, ADD_1832, SUB_1833, ADD_1835, 0.154))
        {
            if (sphere_environment_in_collision(environment, MUL_1849, MUL_1851, ADD_1854, 0.06))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, MUL_1868, MUL_1870, ADD_1873, 0.06))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, MUL_1882, NEGATE_1886, SUB_1897, 0.06))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, MUL_1906, NEGATE_1910, SUB_1921, 0.06))
            {
                return false;
            }
        }  // (22, 87)
        printf("approx link 2: %f %f %f\n", ADD_1832, SUB_1833, ADD_1835);
        float MUL_182 = SUB_149 * 0.7071068;
        float MUL_198 = ADD_144 * 0.7071068;
        float MUL_196 = SUB_138 * 0.7071068;
        float SUB_209 = MUL_198 - MUL_196;
        float ADD_199 = MUL_196 + MUL_198;
        float MUL_184 = ADD_130 * 0.7071068;
        float SUB_186 = MUL_182 - MUL_184;
        float ADD_215 = MUL_182 + MUL_184;
        float MUL_224 = ADD_144 * 0.316;
        float MUL_235 = SUB_149 * MUL_224;
        float MUL_228 = ADD_130 * 0.316;
        float MUL_236 = SUB_138 * MUL_228;
        float ADD_237 = MUL_235 + MUL_236;
        float MUL_240 = ADD_237 * 2.0;
        float INPUT_2 = q[2];
        float DIV_262 = INPUT_2 * 0.5;
        float SIN_263 = sin(DIV_262);
        float COS_269 = cos(DIV_262);
        float MUL_286 = ADD_215 * COS_269;
        float MUL_281 = ADD_215 * SIN_263;
        float MUL_284 = SUB_209 * COS_269;
        float ADD_285 = MUL_281 + MUL_284;
        float MUL_1933 = ADD_285 * ADD_285;
        float MUL_289 = SUB_209 * SIN_263;
        float SUB_290 = MUL_286 - MUL_289;
        float MUL_1934 = SUB_290 * ADD_285;
        float MUL_271 = SUB_186 * COS_269;
        float MUL_276 = SUB_186 * SIN_263;
        float MUL_278 = ADD_199 * COS_269;
        float SUB_279 = MUL_278 - MUL_276;
        float MUL_1935 = SUB_290 * SUB_279;
        float MUL_1932 = SUB_279 * SUB_279;
        float ADD_1941 = MUL_1932 + MUL_1933;
        float MUL_1944 = ADD_1941 * 2.0;
        float SUB_1947 = 1.0 - MUL_1944;
        float MUL_1981 = SUB_1947 * 0.039;
        float MUL_272 = ADD_199 * SIN_263;
        float ADD_273 = MUL_271 + MUL_272;
        float MUL_1939 = ADD_273 * ADD_285;
        float ADD_1967 = MUL_1939 + MUL_1935;
        float MUL_1969 = ADD_1967 * 2.0;
        float MUL_1994 = MUL_1969 * 0.052;
        float MUL_1938 = ADD_273 * SUB_279;
        float SUB_1954 = MUL_1938 - MUL_1934;
        float MUL_1956 = SUB_1954 * 2.0;
        float MUL_1987 = MUL_1956 * 0.028;
        float ADD_2004 = MUL_1981 + MUL_1987;
        float SUB_2007 = ADD_2004 - MUL_1994;
        float ADD_2010 = MUL_240 + SUB_2007;
        float ADD_1948 = MUL_1938 + MUL_1934;
        float MUL_1950 = ADD_1948 * 2.0;
        float MUL_1983 = MUL_1950 * 0.039;
        float MUL_1937 = SUB_290 * ADD_273;
        float MUL_1940 = SUB_279 * ADD_285;
        float SUB_1970 = MUL_1940 - MUL_1937;
        float MUL_1972 = SUB_1970 * 2.0;
        float MUL_1998 = MUL_1972 * 0.052;
        float MUL_1936 = ADD_273 * ADD_273;
        float ADD_1957 = MUL_1933 + MUL_1936;
        float MUL_1960 = ADD_1957 * 2.0;
        float SUB_1963 = 1.0 - MUL_1960;
        float MUL_1989 = SUB_1963 * 0.028;
        float ADD_2005 = MUL_1983 + MUL_1989;
        float SUB_2008 = ADD_2005 - MUL_1998;
        float MUL_246 = ADD_144 * MUL_224;
        float MUL_244 = ADD_130 * MUL_228;
        float ADD_247 = MUL_244 + MUL_246;
        float MUL_249 = ADD_247 * 2.0;
        float SUB_252 = MUL_249 - 0.316;
        float ADD_2011 = SUB_252 + SUB_2008;
        float SUB_1951 = MUL_1939 - MUL_1935;
        float ADD_1964 = MUL_1940 + MUL_1937;
        float ADD_1973 = MUL_1932 + MUL_1936;
        float MUL_1976 = ADD_1973 * 2.0;
        float SUB_1979 = 1.0 - MUL_1976;
        float MUL_2002 = SUB_1979 * 0.052;
        float MUL_1966 = ADD_1964 * 2.0;
        float MUL_1991 = MUL_1966 * 0.028;
        float MUL_1953 = SUB_1951 * 2.0;
        float MUL_1985 = MUL_1953 * 0.039;
        float ADD_2006 = MUL_1985 + MUL_1991;
        float SUB_2009 = ADD_2006 - MUL_2002;
        float MUL_253 = SUB_149 * MUL_228;
        float MUL_255 = SUB_138 * MUL_224;
        float SUB_256 = MUL_253 - MUL_255;
        float MUL_258 = SUB_256 * 2.0;
        float ADD_260 = 0.333 + MUL_258;
        float ADD_2012 = ADD_260 + SUB_2009;
        float MUL_2027 = MUL_1969 * 0.1;
        float SUB_2037 = MUL_240 - MUL_2027;
        float MUL_2031 = MUL_1972 * 0.1;
        float SUB_2038 = SUB_252 - MUL_2031;
        float MUL_2035 = SUB_1979 * 0.1;
        float SUB_2039 = ADD_260 - MUL_2035;
        float MUL_2054 = MUL_1969 * 0.06;
        float SUB_2064 = MUL_240 - MUL_2054;
        float MUL_2058 = MUL_1972 * 0.06;
        float SUB_2065 = SUB_252 - MUL_2058;
        float MUL_2062 = SUB_1979 * 0.06;
        float SUB_2066 = ADD_260 - MUL_2062;
        float MUL_2074 = MUL_1956 * 0.06;
        float MUL_2068 = SUB_1947 * 0.08;
        float ADD_2085 = MUL_2068 + MUL_2074;
        float ADD_2088 = MUL_240 + ADD_2085;
        float MUL_2076 = SUB_1963 * 0.06;
        float MUL_2070 = MUL_1950 * 0.08;
        float ADD_2086 = MUL_2070 + MUL_2076;
        float ADD_2089 = SUB_252 + ADD_2086;
        float MUL_2078 = MUL_1966 * 0.06;
        float MUL_2072 = MUL_1953 * 0.08;
        float ADD_2087 = MUL_2072 + MUL_2078;
        float ADD_2090 = ADD_260 + ADD_2087;
        float MUL_2098 = MUL_1956 * 0.02;
        float ADD_2109 = MUL_2068 + MUL_2098;
        float ADD_2112 = MUL_240 + ADD_2109;
        float MUL_2100 = SUB_1963 * 0.02;
        float ADD_2110 = MUL_2070 + MUL_2100;
        float ADD_2113 = SUB_252 + ADD_2110;
        float MUL_2102 = MUL_1966 * 0.02;
        float ADD_2111 = MUL_2072 + MUL_2102;
        float ADD_2114 = ADD_260 + ADD_2111;
        if (/*panda_link3*/ sphere_environment_in_collision(environment, ADD_2010, ADD_2011, ADD_2012, 0.128))
        {
            if (sphere_environment_in_collision(environment, SUB_2037, SUB_2038, SUB_2039, 0.06))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, SUB_2064, SUB_2065, SUB_2066, 0.05))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2088, ADD_2089, ADD_2090, 0.055))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2112, ADD_2113, ADD_2114, 0.055))
            {
                return false;
            }
        }  // (87, 208)
        printf("approx link 3: %f %f %f\n", ADD_2010, ADD_2011, ADD_2012);
        float MUL_323 = SUB_290 * 0.7071068;
        float MUL_338 = ADD_285 * 0.7071068;
        float MUL_336 = SUB_279 * 0.7071068;
        float SUB_349 = MUL_338 - MUL_336;
        float ADD_339 = MUL_336 + MUL_338;
        float MUL_325 = ADD_273 * 0.7071068;
        float SUB_354 = MUL_323 - MUL_325;
        float ADD_326 = MUL_323 + MUL_325;
        float MUL_371 = ADD_285 * 0.0825;
        float MUL_376 = ADD_285 * MUL_371;
        float MUL_366 = SUB_279 * 0.0825;
        float MUL_374 = SUB_279 * MUL_366;
        float ADD_378 = MUL_374 + MUL_376;
        float MUL_381 = ADD_378 * 2.0;
        float SUB_384 = 0.0825 - MUL_381;
        float ADD_403 = MUL_240 + SUB_384;
        float INPUT_3 = q[3];
        float DIV_407 = INPUT_3 * 0.5;
        float SIN_408 = sin(DIV_407);
        float COS_414 = cos(DIV_407);
        float MUL_431 = SUB_354 * COS_414;
        float MUL_426 = SUB_354 * SIN_408;
        float MUL_429 = SUB_349 * COS_414;
        float ADD_430 = MUL_426 + MUL_429;
        float MUL_2126 = ADD_430 * ADD_430;
        float MUL_434 = SUB_349 * SIN_408;
        float SUB_435 = MUL_431 - MUL_434;
        float MUL_2127 = SUB_435 * ADD_430;
        float MUL_416 = ADD_326 * COS_414;
        float MUL_421 = ADD_326 * SIN_408;
        float MUL_423 = ADD_339 * COS_414;
        float SUB_424 = MUL_423 - MUL_421;
        float MUL_2128 = SUB_435 * SUB_424;
        float MUL_2125 = SUB_424 * SUB_424;
        float ADD_2134 = MUL_2125 + MUL_2126;
        float MUL_2137 = ADD_2134 * 2.0;
        float SUB_2140 = 1.0 - MUL_2137;
        float MUL_2175 = SUB_2140 * 0.042;
        float MUL_417 = ADD_339 * SIN_408;
        float ADD_418 = MUL_416 + MUL_417;
        float MUL_2132 = ADD_418 * ADD_430;
        float ADD_2160 = MUL_2132 + MUL_2128;
        float MUL_2162 = ADD_2160 * 2.0;
        float MUL_2192 = MUL_2162 * 0.029;
        float MUL_2131 = ADD_418 * SUB_424;
        float SUB_2147 = MUL_2131 - MUL_2127;
        float MUL_2149 = SUB_2147 * 2.0;
        float MUL_2186 = MUL_2149 * 0.049;
        float SUB_2197 = MUL_2186 - MUL_2175;
        float ADD_2200 = SUB_2197 + MUL_2192;
        float ADD_2203 = ADD_403 + ADD_2200;
        float ADD_2141 = MUL_2131 + MUL_2127;
        float MUL_2143 = ADD_2141 * 2.0;
        float MUL_2179 = MUL_2143 * 0.042;
        float MUL_2130 = SUB_435 * ADD_418;
        float MUL_2133 = SUB_424 * ADD_430;
        float SUB_2163 = MUL_2133 - MUL_2130;
        float MUL_2165 = SUB_2163 * 2.0;
        float MUL_2194 = MUL_2165 * 0.029;
        float MUL_2129 = ADD_418 * ADD_418;
        float ADD_2150 = MUL_2126 + MUL_2129;
        float MUL_2153 = ADD_2150 * 2.0;
        float SUB_2156 = 1.0 - MUL_2153;
        float MUL_2188 = SUB_2156 * 0.049;
        float SUB_2198 = MUL_2188 - MUL_2179;
        float ADD_2201 = SUB_2198 + MUL_2194;
        float MUL_386 = SUB_290 * MUL_371;
        float MUL_387 = ADD_273 * MUL_366;
        float ADD_389 = MUL_386 + MUL_387;
        float MUL_392 = ADD_389 * 2.0;
        float ADD_404 = SUB_252 + MUL_392;
        float ADD_2204 = ADD_404 + ADD_2201;
        float SUB_2144 = MUL_2132 - MUL_2128;
        float ADD_2157 = MUL_2133 + MUL_2130;
        float ADD_2166 = MUL_2125 + MUL_2129;
        float MUL_2169 = ADD_2166 * 2.0;
        float SUB_2172 = 1.0 - MUL_2169;
        float MUL_2196 = SUB_2172 * 0.029;
        float MUL_2159 = ADD_2157 * 2.0;
        float MUL_2190 = MUL_2159 * 0.049;
        float MUL_2146 = SUB_2144 * 2.0;
        float MUL_2183 = MUL_2146 * 0.042;
        float SUB_2199 = MUL_2190 - MUL_2183;
        float ADD_2202 = SUB_2199 + MUL_2196;
        float MUL_394 = SUB_290 * MUL_366;
        float MUL_396 = ADD_273 * MUL_371;
        float SUB_398 = MUL_396 - MUL_394;
        float MUL_401 = SUB_398 * 2.0;
        float ADD_405 = ADD_260 + MUL_401;
        float ADD_2205 = ADD_405 + ADD_2202;
        float MUL_2208 = SUB_2140 * 0.08;
        float MUL_2219 = MUL_2149 * 0.095;
        float SUB_2230 = MUL_2219 - MUL_2208;
        float ADD_2233 = ADD_403 + SUB_2230;
        float MUL_2221 = SUB_2156 * 0.095;
        float MUL_2212 = MUL_2143 * 0.08;
        float SUB_2231 = MUL_2221 - MUL_2212;
        float ADD_2234 = ADD_404 + SUB_2231;
        float MUL_2223 = MUL_2159 * 0.095;
        float MUL_2216 = MUL_2146 * 0.08;
        float SUB_2232 = MUL_2223 - MUL_2216;
        float ADD_2235 = ADD_405 + SUB_2232;
        float MUL_2249 = MUL_2162 * 0.02;
        float ADD_2254 = ADD_403 + MUL_2249;
        float MUL_2251 = MUL_2165 * 0.02;
        float ADD_2255 = ADD_404 + MUL_2251;
        float MUL_2253 = SUB_2172 * 0.02;
        float ADD_2256 = ADD_405 + MUL_2253;
        float MUL_2270 = MUL_2162 * 0.06;
        float ADD_2275 = ADD_403 + MUL_2270;
        float MUL_2272 = MUL_2165 * 0.06;
        float ADD_2276 = ADD_404 + MUL_2272;
        float MUL_2274 = SUB_2172 * 0.06;
        float ADD_2277 = ADD_405 + MUL_2274;
        float MUL_2291 = MUL_2149 * 0.06;
        float SUB_2302 = MUL_2291 - MUL_2208;
        float ADD_2305 = ADD_403 + SUB_2302;
        float MUL_2293 = SUB_2156 * 0.06;
        float SUB_2303 = MUL_2293 - MUL_2212;
        float ADD_2306 = ADD_404 + SUB_2303;
        float MUL_2295 = MUL_2159 * 0.06;
        float SUB_2304 = MUL_2295 - MUL_2216;
        float ADD_2307 = ADD_405 + SUB_2304;
        if (/*panda_link4*/ sphere_environment_in_collision(environment, ADD_2203, ADD_2204, ADD_2205, 0.126))
        {
            if (sphere_environment_in_collision(environment, ADD_2233, ADD_2234, ADD_2235, 0.06))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2254, ADD_2255, ADD_2256, 0.055))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2275, ADD_2276, ADD_2277, 0.055))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2305, ADD_2306, ADD_2307, 0.055))
            {
                return false;
            }
        }  // (208, 331)
        printf("approx link 4: %f %f %f\n", ADD_2203, ADD_2204, ADD_2205);
        float MUL_469 = SUB_435 * 0.7071068;
        float MUL_486 = ADD_430 * 0.7071068;
        float MUL_527 = ADD_430 * 0.0825;
        float MUL_533 = ADD_430 * MUL_527;
        float MUL_483 = SUB_424 * 0.7071068;
        float SUB_488 = MUL_483 - MUL_486;
        float ADD_499 = MUL_483 + MUL_486;
        float MUL_520 = SUB_424 * 0.0825;
        float MUL_472 = ADD_418 * 0.7071068;
        float SUB_473 = MUL_472 - MUL_469;
        float ADD_506 = MUL_469 + MUL_472;
        float MUL_514 = ADD_430 * 0.384;
        float MUL_529 = SUB_435 * MUL_514;
        float MUL_517 = ADD_418 * 0.384;
        float ADD_522 = MUL_517 + MUL_520;
        float MUL_531 = SUB_424 * ADD_522;
        float SUB_532 = MUL_531 - MUL_529;
        float ADD_534 = SUB_532 + MUL_533;
        float MUL_536 = ADD_534 * 2.0;
        float SUB_539 = MUL_536 - 0.0825;
        float ADD_564 = ADD_403 + SUB_539;
        float INPUT_4 = q[4];
        float DIV_568 = INPUT_4 * 0.5;
        float SIN_569 = sin(DIV_568);
        float COS_575 = cos(DIV_568);
        float MUL_592 = ADD_506 * COS_575;
        float MUL_587 = ADD_506 * SIN_569;
        float MUL_590 = ADD_499 * COS_575;
        float ADD_591 = MUL_587 + MUL_590;
        float MUL_2319 = ADD_591 * ADD_591;
        float MUL_595 = ADD_499 * SIN_569;
        float SUB_596 = MUL_592 - MUL_595;
        float MUL_2320 = SUB_596 * ADD_591;
        float MUL_577 = SUB_473 * COS_575;
        float MUL_582 = SUB_473 * SIN_569;
        float MUL_584 = SUB_488 * COS_575;
        float SUB_585 = MUL_584 - MUL_582;
        float MUL_2321 = SUB_596 * SUB_585;
        float MUL_2318 = SUB_585 * SUB_585;
        float ADD_2327 = MUL_2318 + MUL_2319;
        float MUL_2330 = ADD_2327 * 2.0;
        float SUB_2333 = 1.0 - MUL_2330;
        float MUL_2368 = SUB_2333 * 0.001;
        float MUL_578 = SUB_488 * SIN_569;
        float ADD_579 = MUL_577 + MUL_578;
        float MUL_2325 = ADD_579 * ADD_591;
        float ADD_2353 = MUL_2325 + MUL_2321;
        float MUL_2355 = ADD_2353 * 2.0;
        float MUL_2386 = MUL_2355 * 0.11;
        float MUL_2324 = ADD_579 * SUB_585;
        float SUB_2340 = MUL_2324 - MUL_2320;
        float MUL_2342 = SUB_2340 * 2.0;
        float MUL_2379 = MUL_2342 * 0.037;
        float SUB_2396 = MUL_2379 - MUL_2368;
        float SUB_2399 = SUB_2396 - MUL_2386;
        float ADD_2402 = ADD_564 + SUB_2399;
        float ADD_2334 = MUL_2324 + MUL_2320;
        float MUL_2336 = ADD_2334 * 2.0;
        float MUL_2372 = MUL_2336 * 0.001;
        float MUL_2323 = SUB_596 * ADD_579;
        float MUL_2326 = SUB_585 * ADD_591;
        float SUB_2356 = MUL_2326 - MUL_2323;
        float MUL_2358 = SUB_2356 * 2.0;
        float MUL_2390 = MUL_2358 * 0.11;
        float MUL_2322 = ADD_579 * ADD_579;
        float ADD_2343 = MUL_2319 + MUL_2322;
        float MUL_2346 = ADD_2343 * 2.0;
        float SUB_2349 = 1.0 - MUL_2346;
        float MUL_2381 = SUB_2349 * 0.037;
        float SUB_2397 = MUL_2381 - MUL_2372;
        float SUB_2400 = SUB_2397 - MUL_2390;
        float MUL_541 = SUB_435 * MUL_527;
        float MUL_546 = ADD_430 * MUL_514;
        float MUL_543 = ADD_418 * ADD_522;
        float ADD_544 = MUL_541 + MUL_543;
        float ADD_548 = ADD_544 + MUL_546;
        float MUL_551 = ADD_548 * 2.0;
        float SUB_554 = 0.384 - MUL_551;
        float ADD_565 = ADD_404 + SUB_554;
        float ADD_2403 = ADD_565 + SUB_2400;
        float SUB_2337 = MUL_2325 - MUL_2321;
        float ADD_2350 = MUL_2326 + MUL_2323;
        float ADD_2359 = MUL_2318 + MUL_2322;
        float MUL_2362 = ADD_2359 * 2.0;
        float SUB_2365 = 1.0 - MUL_2362;
        float MUL_2394 = SUB_2365 * 0.11;
        float MUL_2352 = ADD_2350 * 2.0;
        float MUL_2383 = MUL_2352 * 0.037;
        float MUL_2339 = SUB_2337 * 2.0;
        float MUL_2376 = MUL_2339 * 0.001;
        float SUB_2398 = MUL_2383 - MUL_2376;
        float SUB_2401 = SUB_2398 - MUL_2394;
        float MUL_555 = SUB_435 * ADD_522;
        float MUL_558 = SUB_424 * MUL_514;
        float MUL_556 = ADD_418 * MUL_527;
        float SUB_557 = MUL_555 - MUL_556;
        float ADD_560 = SUB_557 + MUL_558;
        float MUL_562 = ADD_560 * 2.0;
        float ADD_566 = ADD_405 + MUL_562;
        float ADD_2404 = ADD_566 + SUB_2401;
        float MUL_2412 = MUL_2342 * 0.055;
        float ADD_2423 = ADD_564 + MUL_2412;
        float MUL_2414 = SUB_2349 * 0.055;
        float ADD_2424 = ADD_565 + MUL_2414;
        float MUL_2416 = MUL_2352 * 0.055;
        float ADD_2425 = ADD_566 + MUL_2416;
        float MUL_2433 = MUL_2342 * 0.075;
        float ADD_2444 = ADD_564 + MUL_2433;
        float MUL_2435 = SUB_2349 * 0.075;
        float ADD_2445 = ADD_565 + MUL_2435;
        float MUL_2437 = MUL_2352 * 0.075;
        float ADD_2446 = ADD_566 + MUL_2437;
        float MUL_2461 = MUL_2355 * 0.22;
        float SUB_2471 = ADD_564 - MUL_2461;
        float MUL_2465 = MUL_2358 * 0.22;
        float SUB_2472 = ADD_565 - MUL_2465;
        float MUL_2469 = SUB_2365 * 0.22;
        float SUB_2473 = ADD_566 - MUL_2469;
        float MUL_2488 = MUL_2355 * 0.18;
        float MUL_2481 = MUL_2342 * 0.05;
        float SUB_2498 = MUL_2481 - MUL_2488;
        float ADD_2501 = ADD_564 + SUB_2498;
        float MUL_2492 = MUL_2358 * 0.18;
        float MUL_2483 = SUB_2349 * 0.05;
        float SUB_2499 = MUL_2483 - MUL_2492;
        float ADD_2502 = ADD_565 + SUB_2499;
        float MUL_2496 = SUB_2365 * 0.18;
        float MUL_2485 = MUL_2352 * 0.05;
        float SUB_2500 = MUL_2485 - MUL_2496;
        float ADD_2503 = ADD_566 + SUB_2500;
        float MUL_2511 = MUL_2342 * 0.08;
        float MUL_2518 = MUL_2355 * 0.14;
        float MUL_2505 = SUB_2333 * 0.01;
        float ADD_2528 = MUL_2505 + MUL_2511;
        float SUB_2531 = ADD_2528 - MUL_2518;
        float ADD_2534 = ADD_564 + SUB_2531;
        float MUL_2522 = MUL_2358 * 0.14;
        float MUL_2513 = SUB_2349 * 0.08;
        float MUL_2507 = MUL_2336 * 0.01;
        float ADD_2529 = MUL_2507 + MUL_2513;
        float SUB_2532 = ADD_2529 - MUL_2522;
        float ADD_2535 = ADD_565 + SUB_2532;
        float MUL_2526 = SUB_2365 * 0.14;
        float MUL_2515 = MUL_2352 * 0.08;
        float MUL_2509 = MUL_2339 * 0.01;
        float ADD_2530 = MUL_2509 + MUL_2515;
        float SUB_2533 = ADD_2530 - MUL_2526;
        float ADD_2536 = ADD_566 + SUB_2533;
        float MUL_2544 = MUL_2342 * 0.085;
        float ADD_2561 = MUL_2505 + MUL_2544;
        float SUB_2564 = ADD_2561 - MUL_2386;
        float ADD_2567 = ADD_564 + SUB_2564;
        float MUL_2546 = SUB_2349 * 0.085;
        float ADD_2562 = MUL_2507 + MUL_2546;
        float SUB_2565 = ADD_2562 - MUL_2390;
        float ADD_2568 = ADD_565 + SUB_2565;
        float MUL_2548 = MUL_2352 * 0.085;
        float ADD_2563 = MUL_2509 + MUL_2548;
        float SUB_2566 = ADD_2563 - MUL_2394;
        float ADD_2569 = ADD_566 + SUB_2566;
        float MUL_2584 = MUL_2355 * 0.08;
        float MUL_2577 = MUL_2342 * 0.09;
        float ADD_2594 = MUL_2505 + MUL_2577;
        float SUB_2597 = ADD_2594 - MUL_2584;
        float ADD_2600 = ADD_564 + SUB_2597;
        float MUL_2588 = MUL_2358 * 0.08;
        float MUL_2579 = SUB_2349 * 0.09;
        float ADD_2595 = MUL_2507 + MUL_2579;
        float SUB_2598 = ADD_2595 - MUL_2588;
        float ADD_2601 = ADD_565 + SUB_2598;
        float MUL_2592 = SUB_2365 * 0.08;
        float MUL_2581 = MUL_2352 * 0.09;
        float ADD_2596 = MUL_2509 + MUL_2581;
        float SUB_2599 = ADD_2596 - MUL_2592;
        float ADD_2602 = ADD_566 + SUB_2599;
        float MUL_2617 = MUL_2355 * 0.05;
        float MUL_2610 = MUL_2342 * 0.095;
        float ADD_2627 = MUL_2505 + MUL_2610;
        float SUB_2630 = ADD_2627 - MUL_2617;
        float ADD_2633 = ADD_564 + SUB_2630;
        float MUL_2621 = MUL_2358 * 0.05;
        float MUL_2612 = SUB_2349 * 0.095;
        float ADD_2628 = MUL_2507 + MUL_2612;
        float SUB_2631 = ADD_2628 - MUL_2621;
        float ADD_2634 = ADD_565 + SUB_2631;
        float MUL_2625 = SUB_2365 * 0.05;
        float MUL_2614 = MUL_2352 * 0.095;
        float ADD_2629 = MUL_2509 + MUL_2614;
        float SUB_2632 = ADD_2629 - MUL_2625;
        float ADD_2635 = ADD_566 + SUB_2632;
        float SUB_2666 = MUL_2511 - MUL_2505;
        float SUB_2669 = SUB_2666 - MUL_2518;
        float ADD_2672 = ADD_564 + SUB_2669;
        float SUB_2667 = MUL_2513 - MUL_2507;
        float SUB_2670 = SUB_2667 - MUL_2522;
        float ADD_2673 = ADD_565 + SUB_2670;
        float SUB_2668 = MUL_2515 - MUL_2509;
        float SUB_2671 = SUB_2668 - MUL_2526;
        float ADD_2674 = ADD_566 + SUB_2671;
        float SUB_2705 = MUL_2544 - MUL_2505;
        float SUB_2708 = SUB_2705 - MUL_2386;
        float ADD_2711 = ADD_564 + SUB_2708;
        float SUB_2706 = MUL_2546 - MUL_2507;
        float SUB_2709 = SUB_2706 - MUL_2390;
        float ADD_2712 = ADD_565 + SUB_2709;
        float SUB_2707 = MUL_2548 - MUL_2509;
        float SUB_2710 = SUB_2707 - MUL_2394;
        float ADD_2713 = ADD_566 + SUB_2710;
        float SUB_2744 = MUL_2577 - MUL_2505;
        float SUB_2747 = SUB_2744 - MUL_2584;
        float ADD_2750 = ADD_564 + SUB_2747;
        float SUB_2745 = MUL_2579 - MUL_2507;
        float SUB_2748 = SUB_2745 - MUL_2588;
        float ADD_2751 = ADD_565 + SUB_2748;
        float SUB_2746 = MUL_2581 - MUL_2509;
        float SUB_2749 = SUB_2746 - MUL_2592;
        float ADD_2752 = ADD_566 + SUB_2749;
        float SUB_2783 = MUL_2610 - MUL_2505;
        float SUB_2786 = SUB_2783 - MUL_2617;
        float ADD_2789 = ADD_564 + SUB_2786;
        float SUB_2784 = MUL_2612 - MUL_2507;
        float SUB_2787 = SUB_2784 - MUL_2621;
        float ADD_2790 = ADD_565 + SUB_2787;
        float SUB_2785 = MUL_2614 - MUL_2509;
        float SUB_2788 = SUB_2785 - MUL_2625;
        float ADD_2791 = ADD_566 + SUB_2788;
        if (/*panda_link0 vs. panda_link5*/ sphere_sphere_self_collision(
            0.0, 0.0, 0.05, 0.08, ADD_2402, ADD_2403, ADD_2404, 0.176))
        {
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2423, ADD_2424, ADD_2425, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2444, ADD_2445, ADD_2446, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, SUB_2471, SUB_2472, SUB_2473, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2501, ADD_2502, ADD_2503, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2534, ADD_2535, ADD_2536, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2567, ADD_2568, ADD_2569, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2600, ADD_2601, ADD_2602, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2633, ADD_2634, ADD_2635, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2672, ADD_2673, ADD_2674, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2711, ADD_2712, ADD_2713, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2750, ADD_2751, ADD_2752, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2789, ADD_2790, ADD_2791, 0.025))
            {
                return false;
            }
        }  // (331, 557)
        if (/*panda_link1 vs. panda_link5*/ sphere_sphere_self_collision(
            SUB_1641, NEGATE_1643, 0.248, 0.154, ADD_2402, ADD_2403, ADD_2404, 0.176))
        {
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2423, ADD_2424, ADD_2425, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2444, ADD_2445, ADD_2446, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, SUB_2471, SUB_2472, SUB_2473, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2501, ADD_2502, ADD_2503, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2534, ADD_2535, ADD_2536, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2567, ADD_2568, ADD_2569, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2600, ADD_2601, ADD_2602, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2633, ADD_2634, ADD_2635, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2672, ADD_2673, ADD_2674, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2711, ADD_2712, ADD_2713, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2750, ADD_2751, ADD_2752, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2789, ADD_2790, ADD_2791, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2423, ADD_2424, ADD_2425, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2444, ADD_2445, ADD_2446, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, SUB_2471, SUB_2472, SUB_2473, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2501, ADD_2502, ADD_2503, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2534, ADD_2535, ADD_2536, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2567, ADD_2568, ADD_2569, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2600, ADD_2601, ADD_2602, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2633, ADD_2634, ADD_2635, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2672, ADD_2673, ADD_2674, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2711, ADD_2712, ADD_2713, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2750, ADD_2751, ADD_2752, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2789, ADD_2790, ADD_2791, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2423, ADD_2424, ADD_2425, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2444, ADD_2445, ADD_2446, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, SUB_2471, SUB_2472, SUB_2473, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2501, ADD_2502, ADD_2503, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2534, ADD_2535, ADD_2536, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2567, ADD_2568, ADD_2569, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2600, ADD_2601, ADD_2602, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2633, ADD_2634, ADD_2635, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2672, ADD_2673, ADD_2674, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2711, ADD_2712, ADD_2713, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2750, ADD_2751, ADD_2752, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2789, ADD_2790, ADD_2791, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2423, ADD_2424, ADD_2425, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2444, ADD_2445, ADD_2446, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, SUB_2471, SUB_2472, SUB_2473, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2501, ADD_2502, ADD_2503, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2534, ADD_2535, ADD_2536, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2567, ADD_2568, ADD_2569, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2600, ADD_2601, ADD_2602, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2633, ADD_2634, ADD_2635, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2672, ADD_2673, ADD_2674, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2711, ADD_2712, ADD_2713, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2750, ADD_2751, ADD_2752, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2789, ADD_2790, ADD_2791, 0.025))
            {
                return false;
            }
        }  // (557, 557)
        if (/*panda_link2 vs. panda_link5*/ sphere_sphere_self_collision(
            ADD_1832, SUB_1833, ADD_1835, 0.154, ADD_2402, ADD_2403, ADD_2404, 0.176))
        {
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_2423, ADD_2424, ADD_2425, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_2444, ADD_2445, ADD_2446, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, SUB_2471, SUB_2472, SUB_2473, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_2501, ADD_2502, ADD_2503, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_2534, ADD_2535, ADD_2536, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_2567, ADD_2568, ADD_2569, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_2600, ADD_2601, ADD_2602, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_2633, ADD_2634, ADD_2635, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_2672, ADD_2673, ADD_2674, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_2711, ADD_2712, ADD_2713, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_2750, ADD_2751, ADD_2752, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_2789, ADD_2790, ADD_2791, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_2423, ADD_2424, ADD_2425, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_2444, ADD_2445, ADD_2446, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, SUB_2471, SUB_2472, SUB_2473, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_2501, ADD_2502, ADD_2503, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_2534, ADD_2535, ADD_2536, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_2567, ADD_2568, ADD_2569, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_2600, ADD_2601, ADD_2602, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_2633, ADD_2634, ADD_2635, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_2672, ADD_2673, ADD_2674, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_2711, ADD_2712, ADD_2713, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_2750, ADD_2751, ADD_2752, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_2789, ADD_2790, ADD_2791, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_2423, ADD_2424, ADD_2425, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_2444, ADD_2445, ADD_2446, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, SUB_2471, SUB_2472, SUB_2473, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_2501, ADD_2502, ADD_2503, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_2534, ADD_2535, ADD_2536, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_2567, ADD_2568, ADD_2569, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_2600, ADD_2601, ADD_2602, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_2633, ADD_2634, ADD_2635, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_2672, ADD_2673, ADD_2674, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_2711, ADD_2712, ADD_2713, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_2750, ADD_2751, ADD_2752, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_2789, ADD_2790, ADD_2791, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_2423, ADD_2424, ADD_2425, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_2444, ADD_2445, ADD_2446, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, SUB_2471, SUB_2472, SUB_2473, 0.06))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_2501, ADD_2502, ADD_2503, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_2534, ADD_2535, ADD_2536, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_2567, ADD_2568, ADD_2569, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_2600, ADD_2601, ADD_2602, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_2633, ADD_2634, ADD_2635, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_2672, ADD_2673, ADD_2674, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_2711, ADD_2712, ADD_2713, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_2750, ADD_2751, ADD_2752, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_2789, ADD_2790, ADD_2791, 0.025))
            {
                return false;
            }
        }  // (557, 557)
        printf("approx link 5: %f %f %f\n", ADD_2402, ADD_2403, ADD_2404);
        if (/*panda_link5*/ sphere_environment_in_collision(environment, ADD_2402, ADD_2403, ADD_2404, 0.176))
        {
            if (sphere_environment_in_collision(environment, ADD_2423, ADD_2424, ADD_2425, 0.06))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2444, ADD_2445, ADD_2446, 0.06))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, SUB_2471, SUB_2472, SUB_2473, 0.06))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2501, ADD_2502, ADD_2503, 0.05))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2534, ADD_2535, ADD_2536, 0.025))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2567, ADD_2568, ADD_2569, 0.025))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2600, ADD_2601, ADD_2602, 0.025))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2633, ADD_2634, ADD_2635, 0.025))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2672, ADD_2673, ADD_2674, 0.025))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2711, ADD_2712, ADD_2713, 0.025))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2750, ADD_2751, ADD_2752, 0.025))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2789, ADD_2790, ADD_2791, 0.025))
            {
                return false;
            }
        }  // (557, 557)
        float MUL_657 = SUB_596 * 0.7071068;
        float MUL_654 = ADD_591 * 0.7071068;
        float MUL_651 = SUB_585 * 0.7071068;
        float SUB_655 = MUL_654 - MUL_651;
        float ADD_645 = MUL_651 + MUL_654;
        float MUL_659 = ADD_579 * 0.7071068;
        float SUB_660 = MUL_657 - MUL_659;
        float ADD_632 = MUL_657 + MUL_659;
        float INPUT_5 = q[5];
        float DIV_697 = INPUT_5 * 0.5;
        float SIN_698 = sin(DIV_697);
        float COS_704 = cos(DIV_697);
        float MUL_716 = SUB_660 * SIN_698;
        float MUL_721 = SUB_660 * COS_704;
        float MUL_724 = SUB_655 * SIN_698;
        float SUB_725 = MUL_721 - MUL_724;
        float MUL_719 = SUB_655 * COS_704;
        float ADD_720 = MUL_716 + MUL_719;
        float MUL_2820 = SUB_725 * ADD_720;
        float MUL_2819 = ADD_720 * ADD_720;
        float MUL_711 = ADD_632 * SIN_698;
        float MUL_706 = ADD_632 * COS_704;
        float MUL_707 = ADD_645 * SIN_698;
        float ADD_708 = MUL_706 + MUL_707;
        float MUL_713 = ADD_645 * COS_704;
        float SUB_714 = MUL_713 - MUL_711;
        float MUL_2818 = SUB_714 * SUB_714;
        float ADD_2827 = MUL_2818 + MUL_2819;
        float MUL_2830 = ADD_2827 * 2.0;
        float SUB_2833 = 1.0 - MUL_2830;
        float MUL_2867 = SUB_2833 * 0.042;
        float MUL_2824 = ADD_708 * SUB_714;
        float SUB_2840 = MUL_2824 - MUL_2820;
        float MUL_2842 = SUB_2840 * 2.0;
        float MUL_2873 = MUL_2842 * 0.014;
        float ADD_2884 = MUL_2867 + MUL_2873;
        float ADD_2887 = ADD_564 + ADD_2884;
        float ADD_2834 = MUL_2824 + MUL_2820;
        float MUL_2836 = ADD_2834 * 2.0;
        float MUL_2869 = MUL_2836 * 0.042;
        float MUL_2822 = ADD_708 * ADD_708;
        float ADD_2843 = MUL_2819 + MUL_2822;
        float MUL_2846 = ADD_2843 * 2.0;
        float SUB_2849 = 1.0 - MUL_2846;
        float MUL_2875 = SUB_2849 * 0.014;
        float ADD_2885 = MUL_2869 + MUL_2875;
        float ADD_2888 = ADD_565 + ADD_2885;
        float MUL_2821 = SUB_725 * SUB_714;
        float MUL_2823 = SUB_725 * ADD_708;
        float MUL_2826 = SUB_714 * ADD_720;
        float ADD_2850 = MUL_2826 + MUL_2823;
        float MUL_2852 = ADD_2850 * 2.0;
        float MUL_2877 = MUL_2852 * 0.014;
        float MUL_2825 = ADD_708 * ADD_720;
        float SUB_2837 = MUL_2825 - MUL_2821;
        float MUL_2839 = SUB_2837 * 2.0;
        float MUL_2871 = MUL_2839 * 0.042;
        float ADD_2886 = MUL_2871 + MUL_2877;
        float ADD_2889 = ADD_566 + ADD_2886;
        float MUL_2916 = MUL_2842 * 0.01;
        float MUL_2909 = SUB_2833 * 0.08;
        float SUB_2932 = MUL_2909 - MUL_2916;
        float ADD_2935 = ADD_564 + SUB_2932;
        float MUL_2920 = SUB_2849 * 0.01;
        float MUL_2911 = MUL_2836 * 0.08;
        float SUB_2933 = MUL_2911 - MUL_2920;
        float ADD_2936 = ADD_565 + SUB_2933;
        float MUL_2924 = MUL_2852 * 0.01;
        float MUL_2913 = MUL_2839 * 0.08;
        float SUB_2934 = MUL_2913 - MUL_2924;
        float ADD_2937 = ADD_566 + SUB_2934;
        float MUL_2945 = MUL_2842 * 0.035;
        float ADD_2956 = MUL_2909 + MUL_2945;
        float ADD_2959 = ADD_564 + ADD_2956;
        float MUL_2947 = SUB_2849 * 0.035;
        float ADD_2957 = MUL_2911 + MUL_2947;
        float ADD_2960 = ADD_565 + ADD_2957;
        float MUL_2949 = MUL_2852 * 0.035;
        float ADD_2958 = MUL_2913 + MUL_2949;
        float ADD_2961 = ADD_566 + ADD_2958;
        if (/*panda_link0 vs. panda_link6*/ sphere_sphere_self_collision(
            0.0, 0.0, 0.05, 0.08, ADD_2887, ADD_2888, ADD_2889, 0.095))
        {
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_564, ADD_565, ADD_566, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2935, ADD_2936, ADD_2937, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_2959, ADD_2960, ADD_2961, 0.052))
            {
                return false;
            }
        }  // (557, 637)
        if (/*panda_link1 vs. panda_link6*/ sphere_sphere_self_collision(
            SUB_1641, NEGATE_1643, 0.248, 0.154, ADD_2887, ADD_2888, ADD_2889, 0.095))
        {
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_564, ADD_565, ADD_566, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2935, ADD_2936, ADD_2937, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_2959, ADD_2960, ADD_2961, 0.052))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_564, ADD_565, ADD_566, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2935, ADD_2936, ADD_2937, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_2959, ADD_2960, ADD_2961, 0.052))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_564, ADD_565, ADD_566, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2935, ADD_2936, ADD_2937, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_2959, ADD_2960, ADD_2961, 0.052))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_564, ADD_565, ADD_566, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2935, ADD_2936, ADD_2937, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_2959, ADD_2960, ADD_2961, 0.052))
            {
                return false;
            }
        }  // (637, 637)
        if (/*panda_link6*/ sphere_environment_in_collision(environment, ADD_2887, ADD_2888, ADD_2889, 0.095))
        {
            if (sphere_environment_in_collision(environment, ADD_564, ADD_565, ADD_566, 0.05))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2935, ADD_2936, ADD_2937, 0.05))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_2959, ADD_2960, ADD_2961, 0.052))
            {
                return false;
            }
        }  // (637, 637)
        printf("approx link 6: %f %f %f\n", ADD_2887, ADD_2888, ADD_2889);
        float MUL_758 = SUB_725 * 0.7071068;
        float MUL_773 = ADD_720 * 0.7071068;
        float MUL_771 = SUB_714 * 0.7071068;
        float SUB_784 = MUL_773 - MUL_771;
        float ADD_774 = MUL_771 + MUL_773;
        float MUL_760 = ADD_708 * 0.7071068;
        float SUB_789 = MUL_758 - MUL_760;
        float ADD_761 = MUL_758 + MUL_760;
        float MUL_806 = ADD_720 * 0.088;
        float MUL_811 = ADD_720 * MUL_806;
        float MUL_801 = SUB_714 * 0.088;
        float MUL_809 = SUB_714 * MUL_801;
        float ADD_813 = MUL_809 + MUL_811;
        float MUL_816 = ADD_813 * 2.0;
        float SUB_819 = 0.088 - MUL_816;
        float ADD_838 = ADD_564 + SUB_819;
        float INPUT_6 = q[6];
        float DIV_842 = INPUT_6 * 0.5;
        float SIN_843 = sin(DIV_842);
        float COS_849 = cos(DIV_842);
        float MUL_866 = SUB_789 * COS_849;
        float MUL_861 = SUB_789 * SIN_843;
        float MUL_864 = SUB_784 * COS_849;
        float ADD_865 = MUL_861 + MUL_864;
        float MUL_2971 = ADD_865 * ADD_865;
        float MUL_869 = SUB_784 * SIN_843;
        float SUB_870 = MUL_866 - MUL_869;
        float MUL_2972 = SUB_870 * ADD_865;
        float MUL_851 = ADD_761 * COS_849;
        float MUL_856 = ADD_761 * SIN_843;
        float MUL_858 = ADD_774 * COS_849;
        float SUB_859 = MUL_858 - MUL_856;
        float MUL_2973 = SUB_870 * SUB_859;
        float MUL_2970 = SUB_859 * SUB_859;
        float ADD_2979 = MUL_2970 + MUL_2971;
        float MUL_2982 = ADD_2979 * 2.0;
        float SUB_2985 = 1.0 - MUL_2982;
        float MUL_3019 = SUB_2985 * 0.015;
        float MUL_852 = ADD_774 * SIN_843;
        float ADD_853 = MUL_851 + MUL_852;
        float MUL_2977 = ADD_853 * ADD_865;
        float ADD_3005 = MUL_2977 + MUL_2973;
        float MUL_3007 = ADD_3005 * 2.0;
        float MUL_3031 = MUL_3007 * 0.075;
        float MUL_2976 = ADD_853 * SUB_859;
        float SUB_2992 = MUL_2976 - MUL_2972;
        float MUL_2994 = SUB_2992 * 2.0;
        float MUL_3025 = MUL_2994 * 0.015;
        float ADD_3036 = MUL_3019 + MUL_3025;
        float ADD_3039 = ADD_3036 + MUL_3031;
        float ADD_3042 = ADD_838 + ADD_3039;
        float ADD_2986 = MUL_2976 + MUL_2972;
        float MUL_2988 = ADD_2986 * 2.0;
        float MUL_3021 = MUL_2988 * 0.015;
        float MUL_2975 = SUB_870 * ADD_853;
        float MUL_2978 = SUB_859 * ADD_865;
        float SUB_3008 = MUL_2978 - MUL_2975;
        float MUL_3010 = SUB_3008 * 2.0;
        float MUL_3033 = MUL_3010 * 0.075;
        float MUL_2974 = ADD_853 * ADD_853;
        float ADD_2995 = MUL_2971 + MUL_2974;
        float MUL_2998 = ADD_2995 * 2.0;
        float SUB_3001 = 1.0 - MUL_2998;
        float MUL_3027 = SUB_3001 * 0.015;
        float ADD_3037 = MUL_3021 + MUL_3027;
        float ADD_3040 = ADD_3037 + MUL_3033;
        float MUL_821 = SUB_725 * MUL_806;
        float MUL_822 = ADD_708 * MUL_801;
        float ADD_824 = MUL_821 + MUL_822;
        float MUL_827 = ADD_824 * 2.0;
        float ADD_839 = ADD_565 + MUL_827;
        float ADD_3043 = ADD_839 + ADD_3040;
        float SUB_2989 = MUL_2977 - MUL_2973;
        float ADD_3002 = MUL_2978 + MUL_2975;
        float ADD_3011 = MUL_2970 + MUL_2974;
        float MUL_3014 = ADD_3011 * 2.0;
        float SUB_3017 = 1.0 - MUL_3014;
        float MUL_3035 = SUB_3017 * 0.075;
        float MUL_3004 = ADD_3002 * 2.0;
        float MUL_3029 = MUL_3004 * 0.015;
        float MUL_2991 = SUB_2989 * 2.0;
        float MUL_3023 = MUL_2991 * 0.015;
        float ADD_3038 = MUL_3023 + MUL_3029;
        float ADD_3041 = ADD_3038 + MUL_3035;
        float MUL_829 = SUB_725 * MUL_801;
        float MUL_831 = ADD_708 * MUL_806;
        float SUB_833 = MUL_831 - MUL_829;
        float MUL_836 = SUB_833 * 2.0;
        float ADD_840 = ADD_566 + MUL_836;
        float ADD_3044 = ADD_840 + ADD_3041;
        float MUL_3058 = MUL_3007 * 0.07;
        float ADD_3063 = ADD_838 + MUL_3058;
        float MUL_3060 = MUL_3010 * 0.07;
        float ADD_3064 = ADD_839 + MUL_3060;
        float MUL_3062 = SUB_3017 * 0.07;
        float ADD_3065 = ADD_840 + MUL_3062;
        float MUL_3079 = MUL_3007 * 0.08;
        float MUL_3073 = MUL_2994 * 0.04;
        float MUL_3067 = SUB_2985 * 0.02;
        float ADD_3084 = MUL_3067 + MUL_3073;
        float ADD_3087 = ADD_3084 + MUL_3079;
        float ADD_3090 = ADD_838 + ADD_3087;
        float MUL_3081 = MUL_3010 * 0.08;
        float MUL_3075 = SUB_3001 * 0.04;
        float MUL_3069 = MUL_2988 * 0.02;
        float ADD_3085 = MUL_3069 + MUL_3075;
        float ADD_3088 = ADD_3085 + MUL_3081;
        float ADD_3091 = ADD_839 + ADD_3088;
        float MUL_3083 = SUB_3017 * 0.08;
        float MUL_3077 = MUL_3004 * 0.04;
        float MUL_3071 = MUL_2991 * 0.02;
        float ADD_3086 = MUL_3071 + MUL_3077;
        float ADD_3089 = ADD_3086 + MUL_3083;
        float ADD_3092 = ADD_840 + ADD_3089;
        float MUL_3100 = MUL_2994 * 0.02;
        float MUL_3094 = SUB_2985 * 0.04;
        float ADD_3111 = MUL_3094 + MUL_3100;
        float ADD_3114 = ADD_3111 + MUL_3079;
        float ADD_3117 = ADD_838 + ADD_3114;
        float MUL_3102 = SUB_3001 * 0.02;
        float MUL_3096 = MUL_2988 * 0.04;
        float ADD_3112 = MUL_3096 + MUL_3102;
        float ADD_3115 = ADD_3112 + MUL_3081;
        float ADD_3118 = ADD_839 + ADD_3115;
        float MUL_3104 = MUL_3004 * 0.02;
        float MUL_3098 = MUL_2991 * 0.04;
        float ADD_3113 = MUL_3098 + MUL_3104;
        float ADD_3116 = ADD_3113 + MUL_3083;
        float ADD_3119 = ADD_840 + ADD_3116;
        float MUL_3133 = MUL_3007 * 0.085;
        float MUL_3127 = MUL_2994 * 0.06;
        float ADD_3138 = MUL_3094 + MUL_3127;
        float ADD_3141 = ADD_3138 + MUL_3133;
        float ADD_3144 = ADD_838 + ADD_3141;
        float MUL_3135 = MUL_3010 * 0.085;
        float MUL_3129 = SUB_3001 * 0.06;
        float ADD_3139 = MUL_3096 + MUL_3129;
        float ADD_3142 = ADD_3139 + MUL_3135;
        float ADD_3145 = ADD_839 + ADD_3142;
        float MUL_3137 = SUB_3017 * 0.085;
        float MUL_3131 = MUL_3004 * 0.06;
        float ADD_3140 = MUL_3098 + MUL_3131;
        float ADD_3143 = ADD_3140 + MUL_3137;
        float ADD_3146 = ADD_840 + ADD_3143;
        float MUL_3148 = SUB_2985 * 0.06;
        float ADD_3165 = MUL_3148 + MUL_3073;
        float ADD_3168 = ADD_3165 + MUL_3133;
        float ADD_3171 = ADD_838 + ADD_3168;
        float MUL_3150 = MUL_2988 * 0.06;
        float ADD_3166 = MUL_3150 + MUL_3075;
        float ADD_3169 = ADD_3166 + MUL_3135;
        float ADD_3172 = ADD_839 + ADD_3169;
        float MUL_3152 = MUL_2991 * 0.06;
        float ADD_3167 = MUL_3152 + MUL_3077;
        float ADD_3170 = ADD_3167 + MUL_3137;
        float ADD_3173 = ADD_840 + ADD_3170;
        if (/*panda_link0 vs. panda_link7*/ sphere_sphere_self_collision(
            0.0, 0.0, 0.05, 0.08, ADD_3042, ADD_3043, ADD_3044, 0.072))
        {
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
        }  // (637, 793)
        if (/*panda_link1 vs. panda_link7*/ sphere_sphere_self_collision(
            SUB_1641, NEGATE_1643, 0.248, 0.154, ADD_3042, ADD_3043, ADD_3044, 0.072))
        {
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
        }  // (793, 793)
        if (/*panda_link2 vs. panda_link7*/ sphere_sphere_self_collision(
            ADD_1832, SUB_1833, ADD_1835, 0.154, ADD_3042, ADD_3043, ADD_3044, 0.072))
        {
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
        }  // (793, 793)
        if (/*panda_link5 vs. panda_link7*/ sphere_sphere_self_collision(
            ADD_2402, ADD_2403, ADD_2404, 0.176, ADD_3042, ADD_3043, ADD_3044, 0.072))
        {
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
        }  // (793, 793)

        // if (local_cc_result[0]>0) return false;
        printf("approx link 7: %f %f %f\n", ADD_3042, ADD_3043, ADD_3044);

        if (/*panda_link7*/ sphere_environment_in_collision(environment, ADD_3042, ADD_3043, ADD_3044, 0.072))
        {
            if (sphere_environment_in_collision(environment, ADD_3063, ADD_3064, ADD_3065, 0.05))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3090, ADD_3091, ADD_3092, 0.025))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3117, ADD_3118, ADD_3119, 0.025))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3144, ADD_3145, ADD_3146, 0.02))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3171, ADD_3172, ADD_3173, 0.02))
            {
                return false;
            }
        }  // (793, 793)
        float MUL_1065 = SUB_870 * 0.9238795;
        float MUL_1062 = ADD_865 * 0.9238795;
        float MUL_1049 = SUB_859 * 0.9238795;
        float MUL_1034 = ADD_853 * 0.9238795;
        float MUL_1055 = SUB_870 * 0.3826834;
        float SUB_1063 = MUL_1062 - MUL_1055;
        float MUL_1072 = ADD_865 * 0.3826834;
        float ADD_1074 = MUL_1065 + MUL_1072;
        float MUL_1037 = SUB_859 * 0.3826834;
        float SUB_1039 = MUL_1034 - MUL_1037;
        float MUL_3241 = SUB_1039 * SUB_1063;
        float MUL_1046 = ADD_853 * 0.3826834;
        float ADD_1050 = MUL_1046 + MUL_1049;
        float MUL_3237 = ADD_1074 * ADD_1050;
        float ADD_3269 = MUL_3241 + MUL_3237;
        float MUL_3271 = ADD_3269 * 2.0;
        float MUL_931 = SUB_859 * 0.107;
        float MUL_942 = SUB_870 * MUL_931;
        float MUL_939 = ADD_853 * 0.107;
        float MUL_944 = ADD_865 * MUL_939;
        float ADD_945 = MUL_942 + MUL_944;
        float MUL_947 = ADD_945 * 2.0;
        float ADD_969 = ADD_838 + MUL_947;
        float MUL_3295 = MUL_3271 * 0.022;
        float ADD_3300 = ADD_969 + MUL_3295;
        float MUL_3239 = ADD_1074 * SUB_1039;
        float MUL_3242 = ADD_1050 * SUB_1063;
        float SUB_3272 = MUL_3242 - MUL_3239;
        float MUL_3274 = SUB_3272 * 2.0;
        float MUL_3297 = MUL_3274 * 0.022;
        float MUL_950 = SUB_870 * MUL_939;
        float MUL_953 = ADD_865 * MUL_931;
        float SUB_954 = MUL_953 - MUL_950;
        float MUL_956 = SUB_954 * 2.0;
        float ADD_970 = ADD_839 + MUL_956;
        float ADD_3301 = ADD_970 + MUL_3297;
        float MUL_3238 = SUB_1039 * SUB_1039;
        float MUL_3234 = ADD_1050 * ADD_1050;
        float ADD_3275 = MUL_3234 + MUL_3238;
        float MUL_3278 = ADD_3275 * 2.0;
        float SUB_3281 = 1.0 - MUL_3278;
        float MUL_3299 = SUB_3281 * 0.022;
        float MUL_961 = SUB_859 * MUL_931;
        float MUL_959 = ADD_853 * MUL_939;
        float ADD_962 = MUL_959 + MUL_961;
        float MUL_965 = ADD_962 * 2.0;
        float SUB_968 = 0.107 - MUL_965;
        float ADD_971 = ADD_840 + SUB_968;
        float ADD_3302 = ADD_971 + MUL_3299;
        float MUL_3322 = MUL_3271 * 0.01;
        float MUL_3236 = ADD_1074 * SUB_1063;
        float MUL_3240 = SUB_1039 * ADD_1050;
        float SUB_3256 = MUL_3240 - MUL_3236;
        float MUL_3258 = SUB_3256 * 2.0;
        float MUL_3311 = MUL_3258 * 0.075;
        float SUB_3327 = MUL_3322 - MUL_3311;
        float ADD_3330 = ADD_969 + SUB_3327;
        float MUL_3324 = MUL_3274 * 0.01;
        float MUL_3235 = SUB_1063 * SUB_1063;
        float ADD_3259 = MUL_3235 + MUL_3238;
        float MUL_3262 = ADD_3259 * 2.0;
        float SUB_3265 = 1.0 - MUL_3262;
        float MUL_3315 = SUB_3265 * 0.075;
        float SUB_3328 = MUL_3324 - MUL_3315;
        float ADD_3331 = ADD_970 + SUB_3328;
        float ADD_3266 = MUL_3242 + MUL_3239;
        float MUL_3326 = SUB_3281 * 0.01;
        float MUL_3268 = ADD_3266 * 2.0;
        float MUL_3319 = MUL_3268 * 0.075;
        float SUB_3329 = MUL_3326 - MUL_3319;
        float ADD_3332 = ADD_971 + SUB_3329;
        float MUL_3341 = MUL_3258 * 0.045;
        float SUB_3357 = MUL_3322 - MUL_3341;
        float ADD_3360 = ADD_969 + SUB_3357;
        float MUL_3345 = SUB_3265 * 0.045;
        float SUB_3358 = MUL_3324 - MUL_3345;
        float ADD_3361 = ADD_970 + SUB_3358;
        float MUL_3349 = MUL_3268 * 0.045;
        float SUB_3359 = MUL_3326 - MUL_3349;
        float ADD_3362 = ADD_971 + SUB_3359;
        float MUL_3371 = MUL_3258 * 0.015;
        float SUB_3387 = MUL_3322 - MUL_3371;
        float ADD_3390 = ADD_969 + SUB_3387;
        float MUL_3375 = SUB_3265 * 0.015;
        float SUB_3388 = MUL_3324 - MUL_3375;
        float ADD_3391 = ADD_970 + SUB_3388;
        float MUL_3379 = MUL_3268 * 0.015;
        float SUB_3389 = MUL_3326 - MUL_3379;
        float ADD_3392 = ADD_971 + SUB_3389;
        float ADD_3411 = MUL_3371 + MUL_3322;
        float ADD_3414 = ADD_969 + ADD_3411;
        float ADD_3412 = MUL_3375 + MUL_3324;
        float ADD_3415 = ADD_970 + ADD_3412;
        float ADD_3413 = MUL_3379 + MUL_3326;
        float ADD_3416 = ADD_971 + ADD_3413;
        float ADD_3435 = MUL_3341 + MUL_3322;
        float ADD_3438 = ADD_969 + ADD_3435;
        float ADD_3436 = MUL_3345 + MUL_3324;
        float ADD_3439 = ADD_970 + ADD_3436;
        float ADD_3437 = MUL_3349 + MUL_3326;
        float ADD_3440 = ADD_971 + ADD_3437;
        float ADD_3459 = MUL_3311 + MUL_3322;
        float ADD_3462 = ADD_969 + ADD_3459;
        float ADD_3460 = MUL_3315 + MUL_3324;
        float ADD_3463 = ADD_970 + ADD_3460;
        float ADD_3461 = MUL_3319 + MUL_3326;
        float ADD_3464 = ADD_971 + ADD_3461;
        float MUL_3484 = MUL_3271 * 0.03;
        float SUB_3489 = MUL_3484 - MUL_3311;
        float ADD_3492 = ADD_969 + SUB_3489;
        float MUL_3486 = MUL_3274 * 0.03;
        float SUB_3490 = MUL_3486 - MUL_3315;
        float ADD_3493 = ADD_970 + SUB_3490;
        float MUL_3488 = SUB_3281 * 0.03;
        float SUB_3491 = MUL_3488 - MUL_3319;
        float ADD_3494 = ADD_971 + SUB_3491;
        float SUB_3519 = MUL_3484 - MUL_3341;
        float ADD_3522 = ADD_969 + SUB_3519;
        float SUB_3520 = MUL_3486 - MUL_3345;
        float ADD_3523 = ADD_970 + SUB_3520;
        float SUB_3521 = MUL_3488 - MUL_3349;
        float ADD_3524 = ADD_971 + SUB_3521;
        float SUB_3549 = MUL_3484 - MUL_3371;
        float ADD_3552 = ADD_969 + SUB_3549;
        float SUB_3550 = MUL_3486 - MUL_3375;
        float ADD_3553 = ADD_970 + SUB_3550;
        float SUB_3551 = MUL_3488 - MUL_3379;
        float ADD_3554 = ADD_971 + SUB_3551;
        float ADD_3573 = MUL_3371 + MUL_3484;
        float ADD_3576 = ADD_969 + ADD_3573;
        float ADD_3574 = MUL_3375 + MUL_3486;
        float ADD_3577 = ADD_970 + ADD_3574;
        float ADD_3575 = MUL_3379 + MUL_3488;
        float ADD_3578 = ADD_971 + ADD_3575;
        float ADD_3597 = MUL_3341 + MUL_3484;
        float ADD_3600 = ADD_969 + ADD_3597;
        float ADD_3598 = MUL_3345 + MUL_3486;
        float ADD_3601 = ADD_970 + ADD_3598;
        float ADD_3599 = MUL_3349 + MUL_3488;
        float ADD_3602 = ADD_971 + ADD_3599;
        float ADD_3621 = MUL_3311 + MUL_3484;
        float ADD_3624 = ADD_969 + ADD_3621;
        float ADD_3622 = MUL_3315 + MUL_3486;
        float ADD_3625 = ADD_970 + ADD_3622;
        float ADD_3623 = MUL_3319 + MUL_3488;
        float ADD_3626 = ADD_971 + ADD_3623;
        float MUL_3646 = MUL_3271 * 0.05;
        float SUB_3651 = MUL_3646 - MUL_3311;
        float ADD_3654 = ADD_969 + SUB_3651;
        float MUL_3648 = MUL_3274 * 0.05;
        float SUB_3652 = MUL_3648 - MUL_3315;
        float ADD_3655 = ADD_970 + SUB_3652;
        float MUL_3650 = SUB_3281 * 0.05;
        float SUB_3653 = MUL_3650 - MUL_3319;
        float ADD_3656 = ADD_971 + SUB_3653;
        float SUB_3681 = MUL_3646 - MUL_3341;
        float ADD_3684 = ADD_969 + SUB_3681;
        float SUB_3682 = MUL_3648 - MUL_3345;
        float ADD_3685 = ADD_970 + SUB_3682;
        float SUB_3683 = MUL_3650 - MUL_3349;
        float ADD_3686 = ADD_971 + SUB_3683;
        float SUB_3711 = MUL_3646 - MUL_3371;
        float ADD_3714 = ADD_969 + SUB_3711;
        float SUB_3712 = MUL_3648 - MUL_3375;
        float ADD_3715 = ADD_970 + SUB_3712;
        float SUB_3713 = MUL_3650 - MUL_3379;
        float ADD_3716 = ADD_971 + SUB_3713;
        float ADD_3735 = MUL_3371 + MUL_3646;
        float ADD_3738 = ADD_969 + ADD_3735;
        float ADD_3736 = MUL_3375 + MUL_3648;
        float ADD_3739 = ADD_970 + ADD_3736;
        float ADD_3737 = MUL_3379 + MUL_3650;
        float ADD_3740 = ADD_971 + ADD_3737;
        float ADD_3759 = MUL_3341 + MUL_3646;
        float ADD_3762 = ADD_969 + ADD_3759;
        float ADD_3760 = MUL_3345 + MUL_3648;
        float ADD_3763 = ADD_970 + ADD_3760;
        float ADD_3761 = MUL_3349 + MUL_3650;
        float ADD_3764 = ADD_971 + ADD_3761;
        float ADD_3783 = MUL_3311 + MUL_3646;
        float ADD_3786 = ADD_969 + ADD_3783;
        float ADD_3784 = MUL_3315 + MUL_3648;
        float ADD_3787 = ADD_970 + ADD_3784;
        float ADD_3785 = MUL_3319 + MUL_3650;
        float ADD_3788 = ADD_971 + ADD_3785;
        printf("approx hand: %f %f %f\n", ADD_3300, ADD_3301, ADD_3302);
        if (/*panda_hand*/ sphere_environment_in_collision(environment, ADD_3300, ADD_3301, ADD_3302, 0.104))
        {
            if (sphere_environment_in_collision(environment, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            // printf("about to check panda hand collision:\n");
            if (sphere_environment_in_collision(environment, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                // printf("panda hand collision!");
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
        }  // (793, 978)
        if (/*panda_link0 vs. panda_hand*/ sphere_sphere_self_collision(
            0.0, 0.0, 0.05, 0.08, ADD_3300, ADD_3301, ADD_3302, 0.104))
        {
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
        }  // (978, 978)
        if (/*panda_link1 vs. panda_hand*/ sphere_sphere_self_collision(
            SUB_1641, NEGATE_1643, 0.248, 0.154, ADD_3300, ADD_3301, ADD_3302, 0.104))
        {
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
        }  // (978, 978)
        if (/*panda_link2 vs. panda_hand*/ sphere_sphere_self_collision(
            ADD_1832, SUB_1833, ADD_1835, 0.154, ADD_3300, ADD_3301, ADD_3302, 0.104))
        {
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
        }  // (978, 978)
        // if (local_cc_result[0]>0) return false;
        if (/*panda_link5 vs. panda_hand*/ sphere_sphere_self_collision(
            ADD_2402, ADD_2403, ADD_2404, 0.176, ADD_3300, ADD_3301, ADD_3302, 0.104))
        {
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3330, ADD_3331, ADD_3332, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3360, ADD_3361, ADD_3362, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3390, ADD_3391, ADD_3392, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3414, ADD_3415, ADD_3416, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3438, ADD_3439, ADD_3440, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3462, ADD_3463, ADD_3464, 0.028))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3492, ADD_3493, ADD_3494, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3522, ADD_3523, ADD_3524, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3552, ADD_3553, ADD_3554, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3576, ADD_3577, ADD_3578, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3600, ADD_3601, ADD_3602, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3624, ADD_3625, ADD_3626, 0.026))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3654, ADD_3655, ADD_3656, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3684, ADD_3685, ADD_3686, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3714, ADD_3715, ADD_3716, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3738, ADD_3739, ADD_3740, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3762, ADD_3763, ADD_3764, 0.024))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3786, ADD_3787, ADD_3788, 0.024))
            {
                return false;
            }
        }  // (978, 978)
        // if (local_cc_result[0]>0) return false;
        float MUL_3864 = ADD_3269 * 2.0;
        float MUL_3851 = SUB_3256 * 2.0;
        float MUL_1196 = SUB_1063 * 0.065;
        float MUL_1199 = SUB_1039 * 0.065;
        float MUL_1207 = ADD_1050 * MUL_1199;
        float MUL_1203 = SUB_1039 * 0.0584;
        float MUL_1209 = SUB_1063 * MUL_1203;
        float MUL_1194 = ADD_1050 * 0.0584;
        float SUB_1197 = MUL_1194 - MUL_1196;
        float MUL_1206 = ADD_1074 * SUB_1197;
        float ADD_1208 = MUL_1206 + MUL_1207;
        float ADD_1210 = ADD_1208 + MUL_1209;
        float MUL_1212 = ADD_1210 * 2.0;
        float ADD_1235 = ADD_969 + MUL_1212;
        float MUL_3888 = MUL_3864 * 0.033;
        float MUL_3882 = MUL_3851 * 0.012;
        float ADD_3893 = MUL_3882 + MUL_3888;
        float ADD_3896 = ADD_1235 + ADD_3893;
        float MUL_3867 = SUB_3272 * 2.0;
        float MUL_3890 = MUL_3867 * 0.033;
        float MUL_3855 = ADD_3259 * 2.0;
        float SUB_3858 = 1.0 - MUL_3855;
        float MUL_3884 = SUB_3858 * 0.012;
        float ADD_3894 = MUL_3884 + MUL_3890;
        float MUL_1215 = ADD_1074 * MUL_1203;
        float MUL_1220 = SUB_1063 * SUB_1197;
        float MUL_1217 = SUB_1039 * MUL_1199;
        float ADD_1218 = MUL_1215 + MUL_1217;
        float SUB_1221 = MUL_1220 - ADD_1218;
        float MUL_1223 = SUB_1221 * 2.0;
        float ADD_1225 = MUL_1223 + 0.065;
        float ADD_1236 = ADD_970 + ADD_1225;
        float ADD_3897 = ADD_1236 + ADD_3894;
        float MUL_3871 = ADD_3275 * 2.0;
        float SUB_3874 = 1.0 - MUL_3871;
        float MUL_3892 = SUB_3874 * 0.033;
        float MUL_3861 = ADD_3266 * 2.0;
        float MUL_3886 = MUL_3861 * 0.012;
        float ADD_3895 = MUL_3886 + MUL_3892;
        float MUL_1226 = ADD_1074 * MUL_1199;
        float MUL_1227 = SUB_1039 * MUL_1203;
        float SUB_1228 = MUL_1226 - MUL_1227;
        float MUL_1229 = ADD_1050 * SUB_1197;
        float SUB_1230 = SUB_1228 - MUL_1229;
        float MUL_1232 = SUB_1230 * 2.0;
        float ADD_1234 = MUL_1232 + 0.0584;
        float ADD_1237 = ADD_971 + ADD_1234;
        float ADD_3898 = ADD_1237 + ADD_3895;
        float MUL_3912 = MUL_3864 * 0.022;
        float MUL_3906 = MUL_3851 * 0.015;
        float ADD_3917 = MUL_3906 + MUL_3912;
        float ADD_3920 = ADD_1235 + ADD_3917;
        float MUL_3914 = MUL_3867 * 0.022;
        float MUL_3908 = SUB_3858 * 0.015;
        float ADD_3918 = MUL_3908 + MUL_3914;
        float ADD_3921 = ADD_1236 + ADD_3918;
        float MUL_3916 = SUB_3874 * 0.022;
        float MUL_3910 = MUL_3861 * 0.015;
        float ADD_3919 = MUL_3910 + MUL_3916;
        float ADD_3922 = ADD_1237 + ADD_3919;
        float MUL_3936 = MUL_3864 * 0.044;
        float MUL_3930 = MUL_3851 * 0.008;
        float ADD_3941 = MUL_3930 + MUL_3936;
        float ADD_3944 = ADD_1235 + ADD_3941;
        float MUL_3938 = MUL_3867 * 0.044;
        float MUL_3932 = SUB_3858 * 0.008;
        float ADD_3942 = MUL_3932 + MUL_3938;
        float ADD_3945 = ADD_1236 + ADD_3942;
        float MUL_3940 = SUB_3874 * 0.044;
        float MUL_3934 = MUL_3861 * 0.008;
        float ADD_3943 = MUL_3934 + MUL_3940;
        float ADD_3946 = ADD_1237 + ADD_3943;
        printf("approx left finger: %f %f %f\n", ADD_3896, ADD_3897, ADD_3898);
        if (/*panda_leftfinger*/ sphere_environment_in_collision(
            environment, ADD_3896, ADD_3897, ADD_3898, 0.024))
        {
            if (sphere_environment_in_collision(environment, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
        }  // (978, 1050)
        if (/*panda_link0 vs. panda_leftfinger*/ sphere_sphere_self_collision(
            0.0, 0.0, 0.05, 0.08, ADD_3896, ADD_3897, ADD_3898, 0.024))
        {
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
        }  // (1050, 1050)
        if (/*panda_link1 vs. panda_leftfinger*/ sphere_sphere_self_collision(
            SUB_1641, NEGATE_1643, 0.248, 0.154, ADD_3896, ADD_3897, ADD_3898, 0.024))
        {
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
        }  // (1050, 1050)
        if (/*panda_link2 vs. panda_leftfinger*/ sphere_sphere_self_collision(
            ADD_1832, SUB_1833, ADD_1835, 0.154, ADD_3896, ADD_3897, ADD_3898, 0.024))
        {
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
        }  // (1050, 1050)
        if (/*panda_link5 vs. panda_leftfinger*/ sphere_sphere_self_collision(
            ADD_2402, ADD_2403, ADD_2404, 0.176, ADD_3896, ADD_3897, ADD_3898, 0.024))
        {
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3920, ADD_3921, ADD_3922, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_3944, ADD_3945, ADD_3946, 0.012))
            {
                return false;
            }
        }  // (1050, 1050)
        float ADD_1331 = MUL_1194 + MUL_1196;
        float MUL_3990 = ADD_3269 * 2.0;
        float MUL_4020 = MUL_3990 * 0.033;
        float MUL_3977 = SUB_3256 * 2.0;
        float MUL_4009 = MUL_3977 * 0.012;
        float SUB_4025 = MUL_4020 - MUL_4009;
        float MUL_1342 = ADD_1074 * ADD_1331;
        float SUB_1345 = MUL_1342 - MUL_1207;
        float ADD_1347 = SUB_1345 + MUL_1209;
        float MUL_1349 = ADD_1347 * 2.0;
        float ADD_1377 = ADD_969 + MUL_1349;
        float ADD_4028 = ADD_1377 + SUB_4025;
        float SUB_1356 = MUL_1217 - MUL_1215;
        float MUL_3993 = SUB_3272 * 2.0;
        float MUL_4022 = MUL_3993 * 0.033;
        float MUL_3981 = ADD_3259 * 2.0;
        float SUB_3984 = 1.0 - MUL_3981;
        float MUL_4013 = SUB_3984 * 0.012;
        float SUB_4026 = MUL_4022 - MUL_4013;
        float MUL_1357 = SUB_1063 * ADD_1331;
        float ADD_1358 = SUB_1356 + MUL_1357;
        float MUL_1360 = ADD_1358 * 2.0;
        float SUB_1363 = MUL_1360 - 0.065;
        float ADD_1378 = ADD_970 + SUB_1363;
        float ADD_4029 = ADD_1378 + SUB_4026;
        float ADD_1367 = MUL_1226 + MUL_1227;
        float MUL_3997 = ADD_3275 * 2.0;
        float SUB_4000 = 1.0 - MUL_3997;
        float MUL_4024 = SUB_4000 * 0.033;
        float MUL_3987 = ADD_3266 * 2.0;
        float MUL_4017 = MUL_3987 * 0.012;
        float SUB_4027 = MUL_4024 - MUL_4017;
        float MUL_1369 = ADD_1050 * ADD_1331;
        float ADD_1370 = ADD_1367 + MUL_1369;
        float MUL_1373 = ADD_1370 * 2.0;
        float SUB_1376 = 0.0584 - MUL_1373;
        float ADD_1379 = ADD_971 + SUB_1376;
        float ADD_4030 = ADD_1379 + SUB_4027;
        float MUL_4050 = MUL_3990 * 0.022;
        float MUL_4039 = MUL_3977 * 0.015;
        float SUB_4055 = MUL_4050 - MUL_4039;
        float ADD_4058 = ADD_1377 + SUB_4055;
        float MUL_4052 = MUL_3993 * 0.022;
        float MUL_4043 = SUB_3984 * 0.015;
        float SUB_4056 = MUL_4052 - MUL_4043;
        float ADD_4059 = ADD_1378 + SUB_4056;
        float MUL_4054 = SUB_4000 * 0.022;
        float MUL_4047 = MUL_3987 * 0.015;
        float SUB_4057 = MUL_4054 - MUL_4047;
        float ADD_4060 = ADD_1379 + SUB_4057;
        float MUL_4080 = MUL_3990 * 0.044;
        float MUL_4069 = MUL_3977 * 0.008;
        float SUB_4085 = MUL_4080 - MUL_4069;
        float ADD_4088 = ADD_1377 + SUB_4085;
        float MUL_4082 = MUL_3993 * 0.044;
        float MUL_4073 = SUB_3984 * 0.008;
        float SUB_4086 = MUL_4082 - MUL_4073;
        float ADD_4089 = ADD_1378 + SUB_4086;
        float MUL_4084 = SUB_4000 * 0.044;
        float MUL_4077 = MUL_3987 * 0.008;
        float SUB_4087 = MUL_4084 - MUL_4077;
        float ADD_4090 = ADD_1379 + SUB_4087;
        if (/*panda_link0 vs. panda_rightfinger*/ sphere_sphere_self_collision(
            0.0, 0.0, 0.05, 0.08, ADD_4028, ADD_4029, ADD_4030, 0.024))
        {
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.05, 0.08, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
        }  // (1050, 1112)
        if (/*panda_link1 vs. panda_rightfinger*/ sphere_sphere_self_collision(
            SUB_1641, NEGATE_1643, 0.248, 0.154, ADD_4028, ADD_4029, ADD_4030, 0.024))
        {
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1656, NEGATE_1660, 0.333, 0.06, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1680, NEGATE_1684, 0.333, 0.06, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.213, 0.06, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    0.0, 0.0, 0.163, 0.06, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
        }  // (1112, 1112)
        if (/*panda_link2 vs. panda_rightfinger*/ sphere_sphere_self_collision(
            ADD_1832, SUB_1833, ADD_1835, 0.154, ADD_4028, ADD_4029, ADD_4030, 0.024))
        {
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1849, MUL_1851, ADD_1854, 0.06, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1868, MUL_1870, ADD_1873, 0.06, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1882, NEGATE_1886, SUB_1897, 0.06, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    MUL_1906, NEGATE_1910, SUB_1921, 0.06, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
        }  // (1112, 1112)
        if (/*panda_link5 vs. panda_rightfinger*/ sphere_sphere_self_collision(
            ADD_2402, ADD_2403, ADD_2404, 0.176, ADD_4028, ADD_4029, ADD_4030, 0.024))
        {
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2423, ADD_2424, ADD_2425, 0.06, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2444, ADD_2445, ADD_2446, 0.06, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    SUB_2471, SUB_2472, SUB_2473, 0.06, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2501, ADD_2502, ADD_2503, 0.05, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2534, ADD_2535, ADD_2536, 0.025, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2567, ADD_2568, ADD_2569, 0.025, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2600, ADD_2601, ADD_2602, 0.025, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2633, ADD_2634, ADD_2635, 0.025, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2672, ADD_2673, ADD_2674, 0.025, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2711, ADD_2712, ADD_2713, 0.025, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2750, ADD_2751, ADD_2752, 0.025, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_sphere_self_collision(
                    ADD_2789, ADD_2790, ADD_2791, 0.025, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
        }  // (1112, 1112)
        printf("approx right finger: %f %f %f\n", ADD_4028, ADD_4029, ADD_4030);
        if (/*panda_rightfinger*/ sphere_environment_in_collision(
            environment, ADD_4028, ADD_4029, ADD_4030, 0.024))
        {
            if (sphere_environment_in_collision(environment, ADD_4058, ADD_4059, ADD_4060, 0.012))
            {
                return false;
            }
            if (sphere_environment_in_collision(environment, ADD_4088, ADD_4089, ADD_4090, 0.012))
            {
                return false;
            }
        }  // (1112, 1112)

        return true;
    }
}
