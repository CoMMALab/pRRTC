#include "Planners.hh"
#include "Robots.hh"
#include "utils.cuh"
#include "pRRTC_settings.hh"
#include "src/collision/environment.hh"
#include "src/robots/panda.cuh"
#include "src/robots/fetch.cuh"
#include "src/robots/baxter.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>



/*
Parallelized RRTC: Each block works to add a config to the tree (either start or goal depending on balance)
*/


namespace pRRTC {
    using namespace ppln;
    __device__ volatile int solved = 0;
    __device__ volatile int atomic_free_index[2]; // separate for tree_a and tree_b
    __device__ volatile int nodes_size[2];
    __device__ volatile int completed_nodes[2]; // track completed nodes for each tree
    constexpr int MAX_PATH_SIZE = 5000;
    __device__ float path[2][MAX_PATH_SIZE]; // solution path segments for tree_a, and tree_b
    __device__ int path_size[2] = {0, 0};
    __device__ float cost = 0.0;
    __device__ int reached_goal_idx = 0;
    __device__ int solved_iters = 0; // value of iters in the block that solves the problem
    __constant__ pRRTC_settings d_settings;

    constexpr int MAX_GRANULARITY = 32;
    constexpr int MAX_THREADS_PER_BLOCK = 4*MAX_GRANULARITY;

    constexpr int BLOCK_SIZE = 64;
    constexpr float UNWRITTEN_VAL = -9999.0f;

    template<typename Robot>
    struct HaltonState {
        float b[Robot::dimension];   // bases
        float n[Robot::dimension];   // numerators
        float d[Robot::dimension];   // denominators
    };

    void __device__ shuffle_array(float *array, int n, curandState &state) {
        for (int i = n - 1; i > 0; i--) {
            int j = curand(&state) % (i + 1);
            float temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    template<typename Robot>
    __device__ void halton_initialize(HaltonState<Robot>& state, size_t skip_iterations, curandState& rng_state, int idx) {
        
        float primes[16] = {
            3.f, 5.f, 7.f, 11.f, 13.f, 17.f, 19.f, 23.f,
            29.f, 31.f, 37.f, 41.f, 43.f, 47.f, 53.f, 59.f
        };
        if (idx != 0) shuffle_array(primes, 16, rng_state);
        
        // Initialize bases from primes
        for (size_t i = 0; i < Robot::dimension; i++) {
            state.b[i] = primes[i];
            state.n[i] = 0.0f;
            state.d[i] = 1.0f;
        }
        
        // Skip iterations if requested
        volatile float temp_result[Robot::dimension];
        for (size_t i = 0; i < skip_iterations; i++) {
            halton_next(state, (float *)temp_result);
        }
    }

    template<typename Robot>
    __device__ void halton_next(HaltonState<Robot>& state, float* result) {
        for (size_t i = 0; i < Robot::dimension; i++) {
            float xf = state.d[i] - state.n[i];
            bool x_eq_1 = (xf == 1.0f);
            
            if (x_eq_1) {
                // x == 1 case
                state.d[i] = floorf(state.d[i] * state.b[i]);
                state.n[i] = 1.0f;
            } else {
                // x != 1 case
                float y = floorf(state.d[i] / state.b[i]);
                
                // Continue dividing by b until we find the right digit position
                while (xf <= y) {
                    y = floorf(y / state.b[i]);
                }
                
                state.n[i] = floorf((state.b[i] + 1.0f) * y) - xf;
            }
            
            result[i] = state.n[i] / state.d[i];
        }
    }

    __global__ void init_rng(curandState* states, unsigned long seed, int num_rng_states) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_rng_states) return;
        curand_init(seed + idx, idx, 0, &states[idx]);
    }

    template <typename Robot>
    __global__ void init_halton(HaltonState<Robot>* states, curandState* cr_states) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= d_settings.num_new_configs) return;
        // int skip = (curand_uniform(&cr_states[idx]) * 50000.0f);
        int skip = 0;
        if (idx == 0) skip = 0;
        halton_initialize(states[idx], skip, cr_states[idx], idx);
    }

    __device__ inline void print_config(volatile float *config, int dim) {
        for (int i = 0; i < dim; i++) {
            printf("%f ,", config[i]);
        }
        printf("\n");
    }

    inline void setup_environment_on_device(ppln::collision::Environment<float> *&d_env, 
                                      const ppln::collision::Environment<float> &h_env) {
        // allocate the environment struct
        cudaMalloc(&d_env, sizeof(ppln::collision::Environment<float>));
        
        // Initialize struct to zeros first
        cudaMemset(d_env, 0, sizeof(ppln::collision::Environment<float>));

        // Handle each primitive type separately
        if (h_env.num_spheres > 0) {
            // Allocate and copy spheres array
            ppln::collision::Sphere<float> *d_spheres;
            cudaMalloc(&d_spheres, sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres);
            cudaMemcpy(d_spheres, h_env.spheres, 
                    sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres, 
                    cudaMemcpyHostToDevice);
            
            // Update the struct fields directly
            cudaMemcpy(&(d_env->spheres), &d_spheres, sizeof(ppln::collision::Sphere<float>*), 
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_spheres), &h_env.num_spheres, sizeof(unsigned int), 
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_capsules > 0) {
            ppln::collision::Capsule<float> *d_capsules;
            cudaMalloc(&d_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules);
            cudaMemcpy(d_capsules, h_env.capsules,
                    sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->capsules), &d_capsules, sizeof(ppln::collision::Capsule<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_capsules), &h_env.num_capsules, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        // Repeat for each primitive type...
        if (h_env.num_z_aligned_capsules > 0) {
            ppln::collision::Capsule<float> *d_z_capsules;
            cudaMalloc(&d_z_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules);
            cudaMemcpy(d_z_capsules, h_env.z_aligned_capsules,
                    sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->z_aligned_capsules), &d_z_capsules, sizeof(ppln::collision::Capsule<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_capsules), &h_env.num_z_aligned_capsules, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_cylinders > 0) {
            ppln::collision::Cylinder<float> *d_cylinders;
            cudaMalloc(&d_cylinders, sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders);
            cudaMemcpy(d_cylinders, h_env.cylinders,
                    sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->cylinders), &d_cylinders, sizeof(ppln::collision::Cylinder<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cylinders), &h_env.num_cylinders, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_cuboids;
            cudaMalloc(&d_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids);
            cudaMemcpy(d_cuboids, h_env.cuboids,
                    sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->cuboids), &d_cuboids, sizeof(ppln::collision::Cuboid<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cuboids), &h_env.num_cuboids, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_z_aligned_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_z_cuboids;
            cudaMalloc(&d_z_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids);
            cudaMemcpy(d_z_cuboids, h_env.z_aligned_cuboids,
                    sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->z_aligned_cuboids), &d_z_cuboids, sizeof(ppln::collision::Cuboid<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_cuboids), &h_env.num_z_aligned_cuboids, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }
    }


    inline void cleanup_environment_on_device(ppln::collision::Environment<float> *d_env, 
                                        const ppln::collision::Environment<float> &h_env) {
        // Get the pointers from device struct before freeing
        ppln::collision::Sphere<float> *d_spheres = nullptr;
        ppln::collision::Capsule<float> *d_capsules = nullptr;
        ppln::collision::Capsule<float> *d_z_capsules = nullptr;
        ppln::collision::Cylinder<float> *d_cylinders = nullptr;
        ppln::collision::Cuboid<float> *d_cuboids = nullptr;
        ppln::collision::Cuboid<float> *d_z_cuboids = nullptr;

        // Copy each pointer from device memory
        if (h_env.num_spheres > 0) {
            cudaMemcpy(&d_spheres, &(d_env->spheres), sizeof(ppln::collision::Sphere<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_spheres);
        }
        
        if (h_env.num_capsules > 0) {
            cudaMemcpy(&d_capsules, &(d_env->capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_capsules);
        }
        
        if (h_env.num_z_aligned_capsules > 0) {
            cudaMemcpy(&d_z_capsules, &(d_env->z_aligned_capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_z_capsules);
        }
        
        if (h_env.num_cylinders > 0) {
            cudaMemcpy(&d_cylinders, &(d_env->cylinders), sizeof(ppln::collision::Cylinder<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_cylinders);
        }
        
        if (h_env.num_cuboids > 0) {
            cudaMemcpy(&d_cuboids, &(d_env->cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_cuboids);
        }
        
        if (h_env.num_z_aligned_cuboids > 0) {
            cudaMemcpy(&d_z_cuboids, &(d_env->z_aligned_cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_z_cuboids);
        }

        // Finally free the environment struct itself
        cudaFree(d_env);
    }

    __global__ void reset_device_variables_kernel() {
        solved = 0;
        
        atomic_free_index[0] = 0;
        atomic_free_index[1] = 0;
        nodes_size[0] = 0;
        nodes_size[1] = 0;
        completed_nodes[0] = 0;
        completed_nodes[1] = 0;
        
        path_size[0] = 0;
        path_size[1] = 0;
        
        for (int tree = 0; tree < 2; tree++) {
            for (int i = 0; i < MAX_PATH_SIZE; i++) {
                path[tree][i] = 0.0f;
            }
        }
        
        cost = 0.0f;
        reached_goal_idx = 0;
    }

    void reset_device_variables() {
        reset_device_variables_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
    }

    __device__ __forceinline__ void reset_to_unwritten_state(volatile float *buffer, int size, int tid) {
        if (tid == 0) {
            for (int i = 0; i < size; i++) {
                buffer[i] = UNWRITTEN_VAL;
            }
        }
        __syncthreads();
    }
    
    template <typename Robot>
    __global__ void
    // __launch_bounds__(128, 8)
    rrtc(
        float **nodes,
        int **parents,
        float **radii,
        HaltonState<Robot> *halton_states,
        curandState *rng_states,
        ppln::collision::Environment<float> *env
    )
    {
        static constexpr auto dim = Robot::dimension;
        const int tid = threadIdx.x;
        const int bid = blockIdx.x; // 0 ... NUM_NEW_CONFIGS
        __shared__ int t_tree_id; // this tree
        __shared__ int o_tree_id; // the other tree
        __shared__ float config[dim];
        __shared__ float sdata[MAX_THREADS_PER_BLOCK];
        __shared__ int sindex[MAX_THREADS_PER_BLOCK];
        __shared__ volatile unsigned int local_cc_result[1];
        __shared__ float *t_nodes;
        __shared__ float *o_nodes;
        __shared__ int *t_parents;
        __shared__ int *o_parents;
        __shared__ float scale;
        __shared__ float *nearest_node;
        __shared__ float delta[dim];
        __shared__ int index;
        __shared__ float vec[dim];
        __shared__ unsigned int n_extensions;
        __shared__ bool should_skip;
        __align__(16) __shared__ volatile float sphere_pos[6000]; // ~assuming max 120 spheres with granularity 32, each has x y z coordinates
        __align__(16) __shared__ volatile float sphere_pos_approx[2500]; // ~assuming 50 spheres with granularity 32, each has x y z coordinates
        __align__(16) __shared__ volatile int link_CC[640]; //assuming max granularity 32, max number of links 20
        __align__(16) __shared__ float T[16 * 2 * 16]; // 32 robots x 2x4x4 transform matrix

        int iter = 0;

        while (true) {
            if (tid == 0) {
                // printf("iter: %d\n", iter);
                // printf("tree size: %d\n", atomic_free_index[0]);
                iter++;
                if (iter > d_settings.max_iters) {
                    atomicCAS((int *)&solved, 0, -1);
                }

                if (d_settings.balance == 0 || iter == 1) {
                    t_tree_id = (bid < (d_settings.num_new_configs / 2))? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 1 && abs(atomic_free_index[0]-atomic_free_index[1]) < 1.5 * d_settings.num_new_configs) { // dynamic balance
                    float ratio = atomic_free_index[0] / (float)(atomic_free_index[0]+atomic_free_index[1]);
                    float balance_factor = 1 - ratio;
                    t_tree_id = (bid < (d_settings.num_new_configs * balance_factor))? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 1) {
                    float ratio = atomic_free_index[0] / (float)(atomic_free_index[0] + atomic_free_index[1]);
                    if (ratio < d_settings.tree_ratio) t_tree_id = 0;
                    else t_tree_id = 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 2) { // vamp balance
                    float ratio = abs(atomic_free_index[t_tree_id] - atomic_free_index[o_tree_id]) / (float) atomic_free_index[t_tree_id];
                    if (ratio < d_settings.tree_ratio)
                    {
                        t_tree_id = 1 - t_tree_id;
                        o_tree_id = 1 - t_tree_id;
                    }
                }

                t_nodes = nodes[t_tree_id];
                o_nodes = nodes[o_tree_id];
                t_parents = parents[t_tree_id];
                o_parents = parents[o_tree_id];
                
                halton_next(halton_states[bid], (float *)config);
                Robot::scale_cfg((float *)config);
                local_cc_result[0] = 0;
                // printf("config: %f %f %f %f %f %f %f\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6]);
                // 14 dim config for baxter
                // printf("config: %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8], config[9], config[10], config[11], config[12], config[13]);
                // // print out first 3 configs for testing
                // for (int i = 0; i < 4; i++) {
                //     float temp_config[dim];
                //     halton_next(halton_states[bid], (float *)temp_config);
                //     Robot::scale_cfg((float *)temp_config);
                //     printf("test q: %f %f %f %f %f %f %f\n", temp_config[0], temp_config[1], temp_config[2], temp_config[3], temp_config[4], temp_config[5], temp_config[6]);
                // }

            }

            // reset link_CC every iteration
            for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                link_CC[r]=0;
            }

            __syncthreads();

            // parallelized nearest neighbor search
            float local_min_dist = FLT_MAX;
            int local_near_idx = 0;
            float dist;
            int size = min(atomic_free_index[t_tree_id], completed_nodes[t_tree_id]);
            for (int i = tid; i < size; i += blockDim.x) {
                dist = device_utils::sq_l2_dist((float *)&t_nodes[i * dim], (float *) config, dim);
                if (dist < local_min_dist) {
                    local_min_dist = dist;
                    local_near_idx = i;
                }
            }
            sdata[tid] = local_min_dist;
            sindex[tid] = local_near_idx;
            __syncthreads();

            for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                float sdata_tid = sdata[tid];
                float sdata_tid_s = sdata[tid + s];
                __syncthreads();
                if (tid < s){
                    if (sdata_tid_s < sdata_tid) {
                        sdata[tid] = sdata[tid + s];
                        sindex[tid] = sindex[tid + s];
                    }
                }
                __syncthreads();
            }

            // nn index is in sindex[0], distance in sdata[0]
            if (tid == 0) {
                sdata[0] = sqrt(sdata[0]);
                scale = min(1.0f, d_settings.range / (sdata[0]));
                nearest_node = &t_nodes[sindex[0] * dim];

                should_skip = (d_settings.dynamic_domain && radii[t_tree_id][sindex[0]] < sdata[0]);
            }
            __syncthreads();

            if (should_skip) {
                // if (tid == 0) printf("skipping\n");
                continue;
            }
            __syncthreads();

            if (tid < dim) {
                config[tid] = nearest_node[tid] + ((config[tid] - nearest_node[tid]) * scale);
                delta[tid] = (config[tid] - nearest_node[tid]) / (float) d_settings.granularity;
            }
            __syncthreads();

            // validate edge
            float interp_cfg[dim];
            for (int i = 0; i < dim; i++) {
                interp_cfg[i] = nearest_node[i] + (int(tid/4 + 1) * delta[i]);
            }
            
            //approximate FK & CC first, if collision found then detailed FK & CC
            int detailed_FK=0;
            // if (tid == 0) {
            //     printf("q: %f %f %f %f %f %f %f\n", interp_cfg[0], interp_cfg[1], interp_cfg[2], interp_cfg[3], interp_cfg[4], interp_cfg[5], interp_cfg[6]);
            // }
            
            ppln::collision::fk_approx<Robot>(interp_cfg, sphere_pos_approx, T, tid);
            __syncthreads();
            bool config_in_collision2_approx = not ppln::collision::env_collision_check_approx<Robot>(sphere_pos_approx, link_CC, env, tid);
            atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2_approx ? 1u : 0u);
            
            __syncthreads();
            // if collision found in approx env check, proceed to detailed env check
            if (local_cc_result[0]==1){
                // if (tid == 0) printf("approx env collision\n");
                if (tid==0) local_cc_result[0]=0;
                __syncthreads();
                //reset_to_unwritten_state(sphere_pos, 4000, tid);
                ppln::collision::fk<Robot>(interp_cfg, sphere_pos, T, tid);
                detailed_FK=1;
                __syncthreads();
                bool config_in_collision2 = not ppln::collision::env_collision_check<Robot>(sphere_pos, link_CC, env, tid);
                // if (tid == 63) {
                //     printf("config_in_collision2: %d\n", config_in_collision2);
                // }
                atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2 ? 1u : 0u);
                __syncthreads();
            }
            
            for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                link_CC[r]=0;
            }
            __syncthreads();
            // if env check is collision free, proceed to self-collision check
            if (local_cc_result[0]==0){
                
                bool config_in_collision_approx = not ppln::collision::self_collision_check_approx<Robot>(sphere_pos_approx, link_CC, tid);
                atomicOr((unsigned int *)&local_cc_result[0], config_in_collision_approx ? 1u : 0u);
                __syncthreads();
                // if collision found in approx self check, proceed to detailed self check
                if (local_cc_result[0]==1){
                    // if (tid == 0) printf("approx self collision\n");
                    if (tid==0) local_cc_result[0]=0;
                    __syncthreads();
                    if (detailed_FK==0){
                        //reset_to_unwritten_state(sphere_pos, 4000, tid);
                        ppln::collision::fk<Robot>(interp_cfg, sphere_pos, T, tid);
                        detailed_FK=1;
                        __syncthreads();
                    }
                    bool config_in_collision = not ppln::collision::self_collision_check<Robot>(sphere_pos, link_CC, tid);
                    atomicOr((unsigned int *)&local_cc_result[0], config_in_collision ? 1u : 0u);
                    __syncthreads();
                }
                //if(blockIdx.x==0) printf("tid %d: env_collision - %d\n", tid, config_in_collision2);
            }

            bool edge_good = local_cc_result[0] == 0;
            __syncthreads();
            if (edge_good) {
                // grow tree
                if (tid == 0) {
                    // printf("edge good\n");
                    index = atomicAdd((int *)&atomic_free_index[t_tree_id], 1);
                    if (index >= d_settings.max_samples) solved = -1;
                    
                    t_parents[index] = sindex[0];
                    
                    if (d_settings.dynamic_domain) {
                        radii[t_tree_id][index] = FLT_MAX;
                        volatile float *radius_ptr = &radii[t_tree_id][sindex[0]];
                        float old_radius, new_radius;
                        int expected, desired;
                        do {
                            old_radius = *radius_ptr;
                            if (old_radius == FLT_MAX) break;
                            new_radius = old_radius * (1 + d_settings.dd_alpha);
                            expected = __float_as_int(old_radius);
                            desired = __float_as_int(new_radius);
                        } while (atomicCAS((int *)radius_ptr, expected, desired) != expected);
                    }
                    // float last_interp_cfg[dim];
                    // for (int i = 0; i < dim; i++) {
                    //     last_interp_cfg[i] = nearest_node[i] + (32 * delta[i]);
                    // }
                    // printf("last_interp_cfg: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", last_interp_cfg[0], last_interp_cfg[1], last_interp_cfg[2], last_interp_cfg[3], last_interp_cfg[4], last_interp_cfg[5], last_interp_cfg[6], last_interp_cfg[7], last_interp_cfg[8], last_interp_cfg[9], last_interp_cfg[10], last_interp_cfg[11], last_interp_cfg[12], last_interp_cfg[13]);
                    // printf("config added: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8], config[9], config[10], config[11], config[12], config[13]);
                }
                __syncthreads();

                if (tid < dim) {
                    t_nodes[index * dim + tid] = config[tid];
                }
                if (tid == 0) {
                    atomicAdd((int*)&completed_nodes[t_tree_id], 1);
                    __threadfence();
                }
                __syncthreads();

                // connect
                local_min_dist = FLT_MAX;
                local_near_idx = 0;
                int size = min(atomic_free_index[o_tree_id], completed_nodes[o_tree_id]);
                for (unsigned int i = tid; i < size; i += blockDim.x) {
                    dist = device_utils::sq_l2_dist((float *)&o_nodes[i * dim], (float *)config, dim);
                    if (dist < local_min_dist) {
                        local_min_dist = dist;
                        local_near_idx = i;
                    }
                }
                sdata[tid] = local_min_dist;
                sindex[tid] = local_near_idx;
                __syncthreads();
                
                for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                    float sdata_tid = sdata[tid];
                    float sdata_tid_s = sdata[tid + s];
                    __syncthreads();
                    if (tid < s){
                        if (sdata_tid_s < sdata_tid) {
                            sdata[tid] = sdata[tid + s];
                            sindex[tid] = sindex[tid + s];
                        }
                    }
                    __syncthreads();
                }
                
                
                if (tid == 0) {
                    sdata[0] = sqrt(sdata[0]);
                    nearest_node = &o_nodes[sindex[0] * dim];
                    n_extensions = ceil(sdata[0] / d_settings.range);
                    local_cc_result[0] = 0;
                }
                __syncthreads();

                if (tid < dim) {
                    vec[tid] = (nearest_node[tid] - config[tid]) / (float) n_extensions;
                }
                __syncthreads();

                // validate the edge to the nearest neighbor in opposite tree, go as far as we can
                int i_extensions = 0;
                int extension_parent_idx = index;
                while (i_extensions < n_extensions) {
                    for (int i = 0; i < dim; i++) {
                        interp_cfg[i] = config[i] + (int(tid/4 + 1) * (vec[i] / (float) d_settings.granularity));
                    }
                    __syncthreads();
                    
                    //approximate FK & CC first, if collision found then detailed FK & CC
                    int detailed_FK=0;
                    // if (tid == 0) {
                    //     printf("q: %f %f %f %f %f %f %f\n", interp_cfg[0], interp_cfg[1], interp_cfg[2], interp_cfg[3], interp_cfg[4], interp_cfg[5], interp_cfg[6]);
                    // }
                    // clear link_CC
                    for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                        link_CC[r]=0;
                    }
                    __syncthreads();
                    ppln::collision::fk_approx<Robot>(interp_cfg, sphere_pos_approx, T, tid);
                    __syncthreads();
                    bool config_in_collision2_approx = not ppln::collision::env_collision_check_approx<Robot>(sphere_pos_approx, link_CC, env, tid);
                    atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2_approx ? 1u : 0u);
                    __syncthreads();
                    // if collision found in approx env check, proceed to detailed env check
                    if (local_cc_result[0]==1){
                        // if (tid == 0) printf("approx env collision in extension\n");
                        if (tid==0) local_cc_result[0]=0;
                        __syncthreads();
                        //reset_to_unwritten_state(sphere_pos, 4000, tid);
                        ppln::collision::fk<Robot>(interp_cfg, sphere_pos, T, tid);
                        detailed_FK=1;
                        __syncthreads();
                        bool config_in_collision2 = not ppln::collision::env_collision_check<Robot>(sphere_pos, link_CC, env, tid);
                        atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2 ? 1u : 0u);
                        __syncthreads();
                    }
                    //if (tid==0) {
                        //printf("new round\n");
                        //ppln::collision::fkcc<Robot>(interp_cfg, env, tid);
                    //}
                    // if env check is collision free, proceed to self-collision check
                    for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                        link_CC[r]=0;
                    }
                    __syncthreads();
                    if (local_cc_result[0]==0){
                        bool config_in_collision_approx = not ppln::collision::self_collision_check_approx<Robot>(sphere_pos_approx, link_CC, tid);
                        atomicOr((unsigned int *)&local_cc_result[0], config_in_collision_approx ? 1u : 0u);
                        __syncthreads();
                        // if collision found in approx self check, proceed to detailed self check
                        if (local_cc_result[0]==1){
                            // if (tid == 0) printf("approx self collision in extension\n");
                            if (tid==0) local_cc_result[0]=0;
                            __syncthreads();
                            if (detailed_FK==0){
                                //reset_to_unwritten_state(sphere_pos, 4000, tid);
                                ppln::collision::fk<Robot>(interp_cfg, sphere_pos, T, tid);
                                detailed_FK=1;
                                __syncthreads();
                            }
                            bool config_in_collision = not ppln::collision::self_collision_check<Robot>(sphere_pos, link_CC, tid);
                            atomicOr((unsigned int *)&local_cc_result[0], config_in_collision ? 1u : 0u);
                            __syncthreads();
                        }
                        //if(blockIdx.x==0) printf("tid %d: env_collision - %d\n", tid, config_in_collision2);
                    }

                    bool ext_edge_good = local_cc_result[0] == 0;
                    if (!ext_edge_good) break;
                    if (tid == 0) {
                        index = atomicAdd((int *)&atomic_free_index[t_tree_id], 1);
                        if (index >= d_settings.max_samples) solved = -1;
                        t_parents[index] = extension_parent_idx;
                        radii[t_tree_id][index] = FLT_MAX;
                        extension_parent_idx = index;
                        local_cc_result[0] = 0;
                        // printf("config added (extension): %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8], config[9], config[10], config[11], config[12], config[13]);
                    }
                    __syncthreads();
                    if (tid < dim) {
                        config[tid] = config[tid] + vec[tid];
                        t_nodes[index * dim + tid] = config[tid];
                    }
                    if (tid == 0) {
                        atomicAdd((int*)&completed_nodes[t_tree_id], 1);
                        __threadfence();
                    }
                    __syncthreads();
                    i_extensions++;
                    __syncthreads();
                }
                if (i_extensions == n_extensions) { // connected
                    if (tid == 0 && atomicCAS((int *)&solved, 0, 1) == 0) {
                        // trace back to the start and goal.
                        int current = index;
                        int parent;
                        int t_path_size = 0;
                        int o_path_size = 0;
                        while (t_parents[current] != current) {
                            parent = t_parents[current];
                            cost += device_utils::l2_dist((float *)&t_nodes[current * dim], (float *)&t_nodes[parent * dim], dim);
                            for (int i = 0; i < dim; i++) path[t_tree_id][t_path_size * dim + i] = t_nodes[current * dim + i];
                            t_path_size++;
                            current = parent;
                            
                        }
                        if (t_tree_id == 1) reached_goal_idx = current;
                        current = sindex[0];
                        while(o_parents[current] != current) {
                            parent = o_parents[current];
                            cost += device_utils::l2_dist((float *)&o_nodes[current * dim], (float *)&o_nodes[parent * dim], dim);
                            for (int i = 0; i < dim; i++) path[o_tree_id][o_path_size * dim + i] = o_nodes[current * dim + i];
                            o_path_size++;
                            current = parent;
                        }
                        if (t_tree_id == 0) reached_goal_idx = current;
                        path_size[t_tree_id] = t_path_size;
                        path_size[o_tree_id] = o_path_size;
                        solved_iters = iter;
                    }
                    __syncthreads();
                }
            }
            else if (d_settings.dynamic_domain && tid == 0) {      
                // printf("no config added\n");
                volatile float *radius_ptr = &radii[t_tree_id][sindex[0]];
                float old_radius, new_radius;
                int expected, desired;
                do {
                    old_radius = *radius_ptr;
                    if (old_radius == FLT_MAX) {
                        new_radius = d_settings.dd_radius;
                    } else {
                        new_radius = fmaxf(old_radius * (1.f - d_settings.dd_alpha), d_settings.dd_min_radius);
                    }
                    expected = __float_as_int(old_radius);
                    desired = __float_as_int(new_radius);
                } while (atomicCAS((int *)radius_ptr, expected, desired) != expected);
            }
            __syncthreads();
            if (solved != 0) return;
        }
    }




    template <typename Robot>
    PlannerResult<Robot> solve(
        typename Robot::Configuration &start,
        std::vector<typename Robot::Configuration> &goals,
        ppln::collision::Environment<float> &h_environment,
        pRRTC_settings &settings
    ) 
    {
        auto start_time = std::chrono::steady_clock::now();
        static constexpr auto dim = Robot::dimension;
        std::size_t start_index = 0;
        PlannerResult<Robot> res;

        // copy data to GPU
        cudaMemcpyToSymbol(d_settings, &settings, sizeof(settings));
        int num_goals = goals.size();
        float *nodes[2];
        int *parents[2];
        float *radii[2];
        float **d_nodes;
        int **d_parents;
        float **d_radii;
        cudaMalloc(&d_nodes, 2 * sizeof(float*));
        cudaMalloc(&d_parents, 2 * sizeof(int*));
        cudaMalloc(&d_radii, 2 * sizeof(float*));
        const std::size_t config_size = dim * sizeof(float);

        for (int i = 0; i < 2; i++) {
            cudaMalloc(&nodes[i], settings.max_samples * config_size);
            cudaMalloc(&parents[i], settings.max_samples * sizeof(int));
            cudaMalloc(&radii[i], settings.max_samples * sizeof(float));
        }
        cudaMemcpy(d_nodes, nodes, 2 * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_parents, parents, 2 * sizeof(int*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_radii, radii, 2 * sizeof(float*), cudaMemcpyHostToDevice);

        // set nodes to unitialized
        std::vector<float> nodes_init(settings.max_samples * dim, UNWRITTEN_VAL);
        cudaMemcpy((void *)nodes[0], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)nodes[1], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);
        
       

        // initialize radii
        std::vector<float> radii_init(num_goals, FLT_MAX);
        cudaMemcpy((void *)radii[0], radii_init.data(), sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)radii[1], radii_init.data(), sizeof(float) * num_goals, cudaMemcpyHostToDevice);
        
        // create a curandState for each thread
        curandState *rng_states;
        int num_rng_states = settings.num_new_configs * dim;
        cudaMalloc(&rng_states, num_rng_states * sizeof(curandState));
        int numBlocks = (num_rng_states + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_rng<<<numBlocks, BLOCK_SIZE>>>(rng_states, 1, num_rng_states);

        HaltonState<Robot> *halton_states;
        cudaMalloc(&halton_states, settings.num_new_configs * sizeof(HaltonState<Robot>));
        int numBlocks1 = (settings.num_new_configs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_halton<Robot><<<numBlocks1, BLOCK_SIZE>>>(halton_states, rng_states);

        // free index for next available position in tree_a and tree_b
        int h_free_index[2] = {1, num_goals};
        cudaMemcpyToSymbol(atomic_free_index, &h_free_index, sizeof(int) * 2);
        cudaMemcpyToSymbol(nodes_size, &h_free_index, sizeof(int) * 2);
        
        // initialize completed_nodes counter
        int h_completed_nodes[2] = {1, num_goals}; // start and goals are already written
        cudaMemcpyToSymbol(completed_nodes, &h_completed_nodes, sizeof(int) * 2);
        
        // allocate for obstacles
        ppln::collision::Environment<float> *env;
        setup_environment_on_device(env, h_environment);
        cudaCheckError(cudaGetLastError());
        
        // Setup pinned memory for signaling
        int *h_solved;
        int current_samples[2];
        int h_solved_iters = -1;
        cudaMallocHost(&h_solved, sizeof(int));
        *h_solved = -1;

        
        auto copy_start_time = std::chrono::steady_clock::now();
        // add start to tree_a and goals to tree_b
        cudaMemcpy((void *)nodes[0], start.data(), config_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)parents[0], &start_index, sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy((void *)nodes[1], goals.data(), config_size * num_goals, cudaMemcpyHostToDevice);
        std::vector<int> parents_b_init(num_goals);
        iota(parents_b_init.begin(), parents_b_init.end(), 0); // consecutive integers from 0 ... num_goals - 1
        cudaMemcpy((void *)parents[1], parents_b_init.data(), sizeof(int) * num_goals, cudaMemcpyHostToDevice);
        res.copy_ns = get_elapsed_nanoseconds(copy_start_time);

        auto kernel_start_time = std::chrono::steady_clock::now();
        rrtc<Robot><<<settings.num_new_configs, 4*settings.granularity>>> (
            d_nodes,
            d_parents,
            d_radii,
            halton_states,
            rng_states,
            env
        );
        cudaDeviceSynchronize();
        res.kernel_ns = get_elapsed_nanoseconds(kernel_start_time);

        // get data from device
        copy_start_time = std::chrono::steady_clock::now();
        cudaMemcpyFromSymbol(current_samples, atomic_free_index, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(h_solved, solved, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&h_solved_iters, solved_iters, sizeof(int), 0, cudaMemcpyDeviceToHost);
        res.copy_ns += get_elapsed_nanoseconds(copy_start_time);

        cudaCheckError(cudaGetLastError());

        // add data to result struct
        if (*h_solved!=1) *h_solved=0;
        res.start_tree_size = current_samples[0];
        res.goal_tree_size = current_samples[1];
        if (*h_solved) {
            int h_path_size[2];
            float h_paths[2][MAX_PATH_SIZE];
            float h_cost;
            int h_reached_goal_idx;
            cudaMemcpyFromSymbol(h_path_size, path_size, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(h_paths, path, sizeof(float) * 2 * MAX_PATH_SIZE, 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&h_cost, cost, sizeof(float), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&h_reached_goal_idx, reached_goal_idx, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaCheckError(cudaGetLastError());
            res.path.emplace_back(goals[h_reached_goal_idx]);
            typename Robot::Configuration config;
            for (int i = h_path_size[1] - 1; i >= 0; i--) {
                std::copy_n(h_paths[1] + i * dim, dim, config.begin());
                res.path.emplace_back(config);
            }
            for (int i = 0; i < h_path_size[0]; i++) {
                std::copy_n(h_paths[0] + i * dim, dim, config.begin());
                res.path.emplace_back(config);
            }
            res.path.emplace_back(start);
            res.cost = h_cost;
            res.path_length = (h_path_size[0] + h_path_size[1]);
        }
        res.solved = (*h_solved) != 0;
        res.iters = h_solved_iters;
        
        cleanup_environment_on_device(env, h_environment);
        reset_device_variables();
        cudaFree((void *)nodes[0]);
        cudaFree((void *)nodes[1]);
        cudaFree((void *)parents[0]);
        cudaFree((void *)parents[1]);
        cudaFree((void *)radii[0]);
        cudaFree((void *)radii[1]);
        cudaFree(rng_states);
        cudaFree(halton_states);
        cudaFree(d_nodes);
        cudaFree(d_parents);
        cudaFree(d_radii);
        cudaFreeHost(h_solved);
        cudaCheckError(cudaGetLastError());
        res.wall_ns = get_elapsed_nanoseconds(start_time);
        cudaDeviceReset();
        return res;
    }

    //template PlannerResult<typename ppln::robots::Sphere> solve<ppln::robots::Sphere>(std::array<float, 3>&, std::vector<std::array<float, 3>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    template PlannerResult<typename ppln::robots::Panda> solve<ppln::robots::Panda>(std::array<float, 7>&, std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    template PlannerResult<typename ppln::robots::Fetch> solve<ppln::robots::Fetch>(std::array<float, 8>&, std::vector<std::array<float, 8>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    template PlannerResult<typename ppln::robots::Baxter> solve<ppln::robots::Baxter>(std::array<float, 14>&, std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, pRRTC_settings&);

}
