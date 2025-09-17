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

#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5

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

    __device__ __forceinline__ bool warp_any_active_mask(bool pred) {
        // Active-lane mask: which threads are alive in this warp
        unsigned mask = __activemask();
        // Nonzero if any lane's pred is true
        return __any_sync(mask , pred);
    }

    __device__ __forceinline__ bool warp_any_full_mask(bool pred) {
        return __any_sync(0xffffffff , pred);
    }
    
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

    /* adapted from https://github.com/NVlabs/curobo/blob/0a50de1ba72db304195d59d9d0b1ed269696047f/src/curobo/curobolib/cpp/kinematics_fused_kernel.cu */
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

    // prism_fn withOUT control flow
    __device__ __forceinline__ void prism_fn(
        const float *fixed_transforms,
        const float angle,
        const int col_idx,
        float *T_step_col,
        const int xyz
    )
    {
        if (col_idx <= 2)
        {
            fixed_joint_fn(&fixed_transforms[col_idx], T_step_col);
        }
        else
        {
            T_step_col[0] = fixed_transforms[0 + xyz] * angle + fixed_transforms[3];     // FT_0[1];
            T_step_col[1] = fixed_transforms[M + xyz] * angle + fixed_transforms[M + 3]; // FT_1[1];
            T_step_col[2] = fixed_transforms[M + M + xyz] * angle +
                            fixed_transforms[M + M + 3];                                 // FT_2[1];
            T_step_col[3] = 1;
        }
    }
    
    __device__ __forceinline__ float dot4(float *a, float *b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    }

    __device__ __forceinline__ float dot4_col(float *a_col, float *b) {
        return a_col[0] * b[0] + a_col[M] * b[1] + a_col[M*2] * b[2] + a_col[M*3] * b[3];
    }
}

