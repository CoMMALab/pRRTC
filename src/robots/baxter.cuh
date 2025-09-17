namespace ppln::collision {





#define BAXTER_SPHERE_COUNT 75
#define BAXTER_JOINT_COUNT 15
#define BAXTER_SELF_CC_RANGE_COUNT 80
#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5
#define BATCH_SIZE 16

__device__ __constant__ float4 baxter_spheres_array[75] = {
    { -0.025f, -0.1f, 0.1f, 0.25f },
    { -0.025f, 0.1f, 0.1f, 0.25f },
    { -0.065f, 0.0f, 0.4f, 0.23f },
    { 0.04f, 0.0f, 0.686f, 0.2f },
    { 0.0f, 0.0f, 0.25f, 0.1f },
    { 0.0f, 0.0f, 0.1f, 0.1f },
    { 0.0f, 0.0f, 0.0f, 0.1f },
    { -0.02f, 0.0f, 0.22f, 0.08f },
    { -0.01f, 0.0f, 0.11f, 0.08f },
    { 0.0f, 0.0f, 0.0f, 0.08f },
    { 0.0f, 0.0f, 0.0f, 0.1f },
    { 0.0f, 0.0f, 0.0f, 0.08f },
    { 0.0f, 0.0f, 0.22f, 0.08f },
    { 0.0f, 0.0f, 0.11f, 0.08f },
    { 0.0f, 0.0f, 0.03f, 0.07f },
    { 0.0f, 0.0f, -0.03f, 0.07f },
    { 0.0f, 0.0f, 0.02f, 0.07f },
    { 0.0f, 0.0f, -0.04f, 0.08f },
    { 0.01f, 0.0f, 0.09355f, 0.05f },
    { 0.0f, 0.02f, 0.13855f, 0.04f },
    { 0.0f, -0.02f, 0.13855f, 0.04f },
    { -0.005f, 0.081333f, 0.16655f, 0.015f },
    { -0.005f, 0.057333f, 0.16655f, 0.015f },
    { 0.0f, 0.086583f, 0.18855f, 0.012f },
    { 0.0f, 0.086583f, 0.20855f, 0.012f },
    { 0.0f, 0.086583f, 0.22855f, 0.012f },
    { 0.01f, 0.082083f, 0.26625f, 0.014f },
    { -0.01f, 0.082083f, 0.26625f, 0.014f },
    { -0.01f, 0.082083f, 0.24625f, 0.014f },
    { 0.01f, 0.082083f, 0.24625f, 0.014f },
    { 0.005f, -0.059333f, 0.16655f, 0.015f },
    { 0.005f, -0.079333f, 0.16655f, 0.015f },
    { 0.0f, -0.086583f, 0.18855f, 0.012f },
    { 0.0f, -0.086583f, 0.20855f, 0.012f },
    { 0.0f, -0.086583f, 0.22855f, 0.012f },
    { 0.01f, -0.082083f, 0.26625f, 0.014f },
    { -0.01f, -0.082083f, 0.26625f, 0.014f },
    { -0.01f, -0.082083f, 0.24625f, 0.014f },
    { 0.01f, -0.082083f, 0.24625f, 0.014f },
    { 0.0f, 0.0f, -0.6f, 0.5f },
    { 0.0f, 0.0f, 0.25f, 0.1f },
    { 0.0f, 0.0f, 0.1f, 0.1f },
    { 0.0f, 0.0f, 0.0f, 0.1f },
    { -0.02f, 0.0f, 0.22f, 0.08f },
    { -0.01f, 0.0f, 0.11f, 0.08f },
    { 0.0f, 0.0f, 0.0f, 0.08f },
    { 0.0f, 0.0f, 0.0f, 0.1f },
    { 0.0f, 0.0f, 0.0f, 0.08f },
    { 0.0f, 0.0f, 0.22f, 0.08f },
    { 0.0f, 0.0f, 0.11f, 0.08f },
    { 0.0f, 0.0f, 0.03f, 0.07f },
    { 0.0f, 0.0f, -0.03f, 0.07f },
    { 0.0f, 0.0f, 0.02f, 0.07f },
    { 0.0f, 0.0f, -0.04f, 0.08f },
    { 0.01f, 0.0f, 0.09355f, 0.05f },
    { 0.0f, 0.02f, 0.13855f, 0.04f },
    { 0.0f, -0.02f, 0.13855f, 0.04f },
    { -0.005f, 0.081333f, 0.16655f, 0.015f },
    { -0.005f, 0.057333f, 0.16655f, 0.015f },
    { 0.0f, 0.086583f, 0.18855f, 0.012f },
    { 0.0f, 0.086583f, 0.20855f, 0.012f },
    { 0.0f, 0.086583f, 0.22855f, 0.012f },
    { 0.01f, 0.082083f, 0.26625f, 0.014f },
    { -0.01f, 0.082083f, 0.26625f, 0.014f },
    { -0.01f, 0.082083f, 0.24625f, 0.014f },
    { 0.01f, 0.082083f, 0.24625f, 0.014f },
    { 0.005f, -0.059333f, 0.16655f, 0.015f },
    { 0.005f, -0.079333f, 0.16655f, 0.015f },
    { 0.0f, -0.086583f, 0.18855f, 0.012f },
    { 0.0f, -0.086583f, 0.20855f, 0.012f },
    { 0.0f, -0.086583f, 0.22855f, 0.012f },
    { 0.01f, -0.082083f, 0.26625f, 0.014f },
    { -0.01f, -0.082083f, 0.26625f, 0.014f },
    { -0.01f, -0.082083f, 0.24625f, 0.014f },
    { 0.01f, -0.082083f, 0.24625f, 0.014f }
};

__device__ __constant__ float baxter_fixed_transforms[] = {
    // joint 0
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 1
    0.707105, -0.707108, 0.0, 0.064027,
    0.707108, 0.707105, 0.0, 0.259027,
    0.0, 0.0, 1.0, 0.129626,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 2
    1.0, 0.0, 0.0, 0.069,
    0.0, 0.0, 1.0, 0.0,
    0.0, -1.0, 0.0, 0.27035,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 3
    0.0, -0.0, 1.0, 0.102,
    1.0, 0.0, -0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 4
    0.0, 1.0, -0.0, 0.069,
    0.0, 0.0, 1.0, 0.0,
    1.0, -0.0, 0.0, 0.26242,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 5
    0.0, -0.0, 1.0, 0.10359,
    1.0, 0.0, -0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 6
    0.0, 1.0, -0.0, 0.01,
    0.0, 0.0, 1.0, 0.0,
    1.0, -0.0, 0.0, 0.2707,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 7
    0.0, -0.0, 1.0, 0.115975,
    1.0, 0.0, -0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 8
    0.707105, 0.707108, 0.0, 0.064027,
    -0.707108, 0.707105, 0.0, -0.259027,
    0.0, 0.0, 1.0, 0.129626,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 9
    1.0, 0.0, 0.0, 0.069,
    0.0, 0.0, 1.0, 0.0,
    0.0, -1.0, 0.0, 0.27035,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 10
    0.0, -0.0, 1.0, 0.102,
    1.0, 0.0, -0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 11
    0.0, 1.0, -0.0, 0.069,
    0.0, 0.0, 1.0, 0.0,
    1.0, -0.0, 0.0, 0.26242,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 12
    0.0, -0.0, 1.0, 0.10359,
    1.0, 0.0, -0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 13
    0.0, 1.0, -0.0, 0.01,
    0.0, 0.0, 1.0, 0.0,
    1.0, -0.0, 0.0, 0.2707,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 14
    0.0, -0.0, 1.0, 0.115975,
    1.0, 0.0, -0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    
};

__device__ __constant__ int baxter_sphere_to_joint[75] = {
    0,
    0,
    0,
    0,
    1,
    1,
    2,
    3,
    3,
    3,
    4,
    5,
    5,
    5,
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
    0,
    8,
    8,
    9,
    10,
    10,
    10,
    11,
    12,
    12,
    12,
    13,
    13,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14
};

__device__ __constant__ int baxter_flattened_joint_to_spheres[90] = {
    0,
    1,
    2,
    3,
    39,
    -1,
    4,
    5,
    -1,
    6,
    -1,
    7,
    8,
    9,
    -1,
    10,
    -1,
    11,
    12,
    13,
    -1,
    14,
    15,
    -1,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    -1,
    40,
    41,
    -1,
    42,
    -1,
    43,
    44,
    45,
    -1,
    46,
    -1,
    47,
    48,
    49,
    -1,
    50,
    51,
    -1,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    -1
};

__device__ __constant__ int baxter_joint_types[] = {
    3,
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
    5,
    5
};

__device__ __constant__ int baxter_self_cc_ranges[80][3] = {
    { 0, 10, 38 },
    { 0, 46, 74 },
    { 1, 10, 38 },
    { 1, 46, 74 },
    { 2, 10, 38 },
    { 2, 46, 74 },
    { 3, 10, 20 },
    { 3, 26, 38 },
    { 3, 46, 56 },
    { 3, 62, 74 },
    { 4, 11, 39 },
    { 4, 50, 74 },
    { 5, 11, 39 },
    { 5, 50, 74 },
    { 6, 16, 38 },
    { 6, 46, 74 },
    { 7, 16, 38 },
    { 7, 46, 74 },
    { 8, 16, 38 },
    { 8, 46, 74 },
    { 9, 16, 38 },
    { 9, 46, 74 },
    { 10, 26, 29 },
    { 10, 35, 38 },
    { 10, 42, 74 },
    { 11, 26, 29 },
    { 11, 35, 39 },
    { 11, 42, 74 },
    { 12, 26, 29 },
    { 12, 35, 39 },
    { 12, 42, 74 },
    { 13, 26, 29 },
    { 13, 35, 39 },
    { 13, 42, 74 },
    { 14, 26, 29 },
    { 14, 35, 74 },
    { 15, 26, 29 },
    { 15, 35, 74 },
    { 16, 39, 74 },
    { 17, 39, 74 },
    { 18, 39, 74 },
    { 19, 39, 74 },
    { 20, 39, 74 },
    { 21, 39, 74 },
    { 22, 39, 74 },
    { 23, 39, 74 },
    { 24, 39, 74 },
    { 25, 39, 74 },
    { 26, 39, 74 },
    { 27, 39, 74 },
    { 28, 39, 74 },
    { 29, 39, 74 },
    { 30, 39, 74 },
    { 31, 39, 74 },
    { 32, 39, 74 },
    { 33, 39, 74 },
    { 34, 39, 74 },
    { 35, 39, 74 },
    { 36, 39, 74 },
    { 37, 39, 74 },
    { 38, 39, 74 },
    { 39, 47, 74 },
    { 40, 47, 74 },
    { 41, 47, 74 },
    { 42, 52, 74 },
    { 43, 52, 74 },
    { 44, 52, 74 },
    { 45, 52, 74 },
    { 46, 62, 65 },
    { 46, 71, 74 },
    { 47, 62, 65 },
    { 47, 71, 74 },
    { 48, 62, 65 },
    { 48, 71, 74 },
    { 49, 62, 65 },
    { 49, 71, 74 },
    { 50, 62, 65 },
    { 50, 71, 74 },
    { 51, 62, 65 },
    { 51, 71, 74 }
};

__device__ __constant__ int baxter_joint_parents[15] = {
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    0,
    8,
    9,
    10,
    11,
    12,
    13
};

__device__ __constant__ int baxter_T_memory_idx[15] = {
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
};

__device__ __constant__ int baxter_dfs_order[15] = {
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14
};

__device__ __constant__ int baxter_joint_id_to_dof[15] = {
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13
};

template <>
__device__ void fk<ppln::robots::Baxter>(
    const float* q,
    volatile float* sphere_pos, // 75 spheres x 16 robots x 3 coordinates (each column is a robot)
    float *T, // 16 robots x 2 x 4x4 transform matrix , column major
    const int tid
)
{
    // every 4 threads are responsible for one column of the transform matrix T
    // make_transform will calculate the necessary column of T_step needed for the thread
    const int col_ind = tid % 4;
    const int batch_ind = tid / 4;

    int T_offset = batch_ind * 2 * 16;
    float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
    float *T_base = T + T_offset; // 4x4 transform matrix for the batch
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        float *T_col_i = T_base + i * 16 + col_ind * 4;
        for (int r=0; r<4; r++) {
            T_col_i[r] = 0.0f;
        }
        T_col_i[col_ind] = 1.0f;
    }
    __syncthreads();

    int joint_to_sphere_ind = 0;

    for (int j = 0; j < BAXTER_JOINT_COUNT; ++j) {
        int i = baxter_dfs_order[j];
        float T_col_tmp[4];
        int parent_idx = baxter_joint_parents[i];
        int T_memory_idx_parent = baxter_T_memory_idx[parent_idx];
        int T_memory_idx = baxter_T_memory_idx[i];
        int q_idx = baxter_joint_id_to_dof[i];
        if (j > 0) {
            int ft_addr_start = i * 16;
            int joint_type = baxter_joint_types[i];

            if (joint_type <= Z_PRISM) {
                prism_fn(&baxter_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col, joint_type);
            }
            else if (joint_type == X_ROT) {
                xrot_fn(&baxter_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            else if (joint_type == Y_ROT) {
                yrot_fn(&baxter_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            else if (joint_type == Z_ROT) {
                zrot_fn(&baxter_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            
            for (int r=0; r<4; r++){
                T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
            }
            for (int r=0; r<4; r++){
                T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
            }
        }
        __syncwarp();
        while (baxter_flattened_joint_to_spheres[joint_to_sphere_ind] != -1) {
            int sphere_ind = baxter_flattened_joint_to_spheres[joint_to_sphere_ind];
            if (col_ind < 3) {
                // sphere sphere_ind, robot batch_ind (BATCH_SIZE robots), coord col_ind
                sphere_pos[sphere_ind * BATCH_SIZE * 3 + batch_ind * 3 + col_ind] = 
                    T_base[T_memory_idx*16 + col_ind] * baxter_spheres_array[sphere_ind].x +
                    T_base[T_memory_idx*16 + col_ind + M] * baxter_spheres_array[sphere_ind].y +
                    T_base[T_memory_idx*16 + col_ind + M*2] * baxter_spheres_array[sphere_ind].z +
                    T_base[T_memory_idx*16 + col_ind + M*3];
            }
            joint_to_sphere_ind++;
        }
        joint_to_sphere_ind++;
        __syncthreads();
    }
}

// 4 threads per discretized motion for self-collision check
template <>
__device__ bool self_collision_check<ppln::robots::Baxter>(volatile float* sphere_pos, volatile int* joint_in_collision, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool has_collision = false;
    for (int i = thread_ind; i < BAXTER_SELF_CC_RANGE_COUNT; i += 4) {
        if (warp_any_active_mask(has_collision)) return false;
        int sphere_1_ind = baxter_self_cc_ranges[i][0];
        if (joint_in_collision[20*batch_ind + baxter_sphere_to_joint[sphere_1_ind]] == 0) continue;
        float sphere_1[3] = {
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
        };
        for (int j = baxter_self_cc_ranges[i][1]; j <= baxter_self_cc_ranges[i][2]; j++) {
            float sphere_2[3] = {
                sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            if (sphere_sphere_self_collision(
                sphere_1[0], sphere_1[1], sphere_1[2], baxter_spheres_array[sphere_1_ind].w,
                sphere_2[0], sphere_2[1], sphere_2[2], baxter_spheres_array[j].w
            )){
                has_collision=true;
            }
        }
    }
    return !has_collision;

}

// 4 threads per discretized motion for env collision check
template <>
__device__ bool env_collision_check<ppln::robots::Baxter>(volatile float* sphere_pos, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool has_collision=false;

    for (int i = BAXTER_SPHERE_COUNT-1-thread_ind; i >=BAXTER_SPHERE_COUNT%4; i -= 4){
        // sphere i, robot batch_ind (16 robots)

        if (joint_in_collision[20*batch_ind + baxter_sphere_to_joint[i]] > 0 && 
            sphere_environment_in_collision(
                env,
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                baxter_spheres_array[i].w
            )
        ) {
            //return false;
            has_collision=true;
        }
        if (warp_any_full_mask(has_collision)) return false; 
    }
    int i=thread_ind;
    
    if (joint_in_collision[20*batch_ind + baxter_sphere_to_joint[i]] > 0 &&
        sphere_environment_in_collision(
            env,
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
            baxter_spheres_array[i].w
        )
    ) {
        //return false;
        has_collision=true;
    }

    return !has_collision;
}






#define BAXTER_APPROX_SPHERE_COUNT 33
#define BAXTER_APPROX_JOINT_COUNT 15
#define BAXTER_APPROX_SELF_CC_RANGE_COUNT 39
#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5
#define BATCH_SIZE 16

__device__ __constant__ float4 baxter_approx_spheres_array[33] = {
    { -0.044f, 0.0f, 0.222f, 0.409f },
    { 0.04f, 0.0f, 0.686f, 0.2f },
    { 0.0f, 0.0f, 0.175f, 0.175f },
    { 0.0f, 0.0f, 0.0f, 0.1f },
    { -0.01f, 0.0f, 0.11f, 0.19f },
    { 0.0f, 0.0f, 0.0f, 0.1f },
    { 0.0f, 0.0f, 0.11f, 0.19f },
    { 0.0f, 0.0f, 0.0f, 0.1f },
    { 0.0f, 0.0f, -0.015f, 0.105f },
    { 0.01f, 0.0f, 0.09355f, 0.05f },
    { 0.0f, 0.0f, 0.13855f, 0.06f },
    { -0.005f, 0.069333f, 0.16655f, 0.027f },
    { 0.0f, 0.086333f, 0.20855f, 0.032f },
    { 0.0f, 0.082583f, 0.25625f, 0.028f },
    { 0.005f, -0.069333f, 0.16655f, 0.025f },
    { 0.0f, -0.086333f, 0.20855f, 0.032f },
    { 0.0f, -0.082583f, 0.25625f, 0.028f },
    { 0.0f, 0.0f, -0.6f, 0.5f },
    { 0.0f, 0.0f, 0.175f, 0.175f },
    { 0.0f, 0.0f, 0.0f, 0.1f },
    { -0.01f, 0.0f, 0.11f, 0.19f },
    { 0.0f, 0.0f, 0.0f, 0.1f },
    { 0.0f, 0.0f, 0.11f, 0.19f },
    { 0.0f, 0.0f, 0.0f, 0.1f },
    { 0.0f, 0.0f, -0.015f, 0.105f },
    { 0.01f, 0.0f, 0.09355f, 0.05f },
    { 0.0f, 0.0f, 0.13855f, 0.06f },
    { -0.005f, 0.069333f, 0.16655f, 0.027f },
    { 0.0f, 0.086333f, 0.20855f, 0.032f },
    { 0.0f, 0.082583f, 0.25625f, 0.028f },
    { 0.005f, -0.069333f, 0.16655f, 0.025f },
    { 0.0f, -0.086333f, 0.20855f, 0.032f },
    { 0.0f, -0.082583f, 0.25625f, 0.028f }
};

__device__ __constant__ float baxter_approx_fixed_transforms[] = {
    // joint 0
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 1
    0.707105, -0.707108, 0.0, 0.064027,
    0.707108, 0.707105, 0.0, 0.259027,
    0.0, 0.0, 1.0, 0.129626,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 2
    1.0, 0.0, 0.0, 0.069,
    0.0, 0.0, 1.0, 0.0,
    0.0, -1.0, 0.0, 0.27035,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 3
    0.0, -0.0, 1.0, 0.102,
    1.0, 0.0, -0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 4
    0.0, 1.0, -0.0, 0.069,
    0.0, 0.0, 1.0, 0.0,
    1.0, -0.0, 0.0, 0.26242,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 5
    0.0, -0.0, 1.0, 0.10359,
    1.0, 0.0, -0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 6
    0.0, 1.0, -0.0, 0.01,
    0.0, 0.0, 1.0, 0.0,
    1.0, -0.0, 0.0, 0.2707,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 7
    0.0, -0.0, 1.0, 0.115975,
    1.0, 0.0, -0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 8
    0.707105, 0.707108, 0.0, 0.064027,
    -0.707108, 0.707105, 0.0, -0.259027,
    0.0, 0.0, 1.0, 0.129626,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 9
    1.0, 0.0, 0.0, 0.069,
    0.0, 0.0, 1.0, 0.0,
    0.0, -1.0, 0.0, 0.27035,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 10
    0.0, -0.0, 1.0, 0.102,
    1.0, 0.0, -0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 11
    0.0, 1.0, -0.0, 0.069,
    0.0, 0.0, 1.0, 0.0,
    1.0, -0.0, 0.0, 0.26242,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 12
    0.0, -0.0, 1.0, 0.10359,
    1.0, 0.0, -0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 13
    0.0, 1.0, -0.0, 0.01,
    0.0, 0.0, 1.0, 0.0,
    1.0, -0.0, 0.0, 0.2707,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 14
    0.0, -0.0, 1.0, 0.115975,
    1.0, 0.0, -0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    
};

__device__ __constant__ int baxter_approx_sphere_to_joint[33] = {
    0,
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
    7,
    7,
    7,
    7,
    7,
    7,
    0,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14
};

__device__ __constant__ int baxter_approx_flattened_joint_to_spheres[48] = {
    0,
    1,
    17,
    -1,
    2,
    -1,
    3,
    -1,
    4,
    -1,
    5,
    -1,
    6,
    -1,
    7,
    -1,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    -1,
    18,
    -1,
    19,
    -1,
    20,
    -1,
    21,
    -1,
    22,
    -1,
    23,
    -1,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    -1
};

__device__ __constant__ int baxter_approx_joint_types[] = {
    3,
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
    5,
    5
};

__device__ __constant__ int baxter_approx_self_cc_ranges[39][3] = {
    { 0, 5, 16 },
    { 0, 21, 32 },
    { 1, 5, 10 },
    { 1, 13, 16 },
    { 1, 21, 26 },
    { 1, 29, 32 },
    { 2, 6, 17 },
    { 2, 23, 32 },
    { 3, 8, 16 },
    { 3, 21, 32 },
    { 4, 8, 16 },
    { 4, 21, 32 },
    { 5, 13, 13 },
    { 5, 16, 16 },
    { 5, 19, 32 },
    { 6, 13, 13 },
    { 6, 16, 17 },
    { 6, 19, 32 },
    { 7, 13, 13 },
    { 7, 16, 32 },
    { 8, 17, 32 },
    { 9, 17, 32 },
    { 10, 17, 32 },
    { 11, 17, 32 },
    { 12, 17, 32 },
    { 13, 17, 32 },
    { 14, 17, 32 },
    { 15, 17, 32 },
    { 16, 17, 32 },
    { 17, 22, 32 },
    { 18, 22, 32 },
    { 19, 24, 32 },
    { 20, 24, 32 },
    { 21, 29, 29 },
    { 21, 32, 32 },
    { 22, 29, 29 },
    { 22, 32, 32 },
    { 23, 29, 29 },
    { 23, 32, 32 }
};

__device__ __constant__ int baxter_approx_joint_parents[15] = {
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    0,
    8,
    9,
    10,
    11,
    12,
    13
};

__device__ __constant__ int baxter_approx_T_memory_idx[15] = {
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
};

__device__ __constant__ int baxter_approx_dfs_order[15] = {
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14
};

__device__ __constant__ int baxter_approx_joint_id_to_dof[15] = {
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13
};

template <>
__device__ void fk_approx<ppln::robots::Baxter>(
    const float* q,
    volatile float* sphere_pos_approx, // 33 spheres x 16 robots x 3 coordinates (each column is a robot)
    float *T, // 16 robots x 2 x 4x4 transform matrix , column major
    const int tid
)
{
    // every 4 threads are responsible for one column of the transform matrix T
    // make_transform will calculate the necessary column of T_step needed for the thread
    const int col_ind = tid % 4;
    const int batch_ind = tid / 4;

    int T_offset = batch_ind * 2 * 16;
    float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
    float *T_base = T + T_offset; // 4x4 transform matrix for the batch
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        float *T_col_i = T_base + i * 16 + col_ind * 4;
        for (int r=0; r<4; r++) {
            T_col_i[r] = 0.0f;
        }
        T_col_i[col_ind] = 1.0f;
    }
    __syncthreads();

    int joint_to_sphere_ind = 0;

    for (int j = 0; j < BAXTER_APPROX_JOINT_COUNT; ++j) {
        int i = baxter_approx_dfs_order[j];
        float T_col_tmp[4];
        int parent_idx = baxter_approx_joint_parents[i];
        int T_memory_idx_parent = baxter_approx_T_memory_idx[parent_idx];
        int T_memory_idx = baxter_approx_T_memory_idx[i];
        int q_idx = baxter_approx_joint_id_to_dof[i];
        if (j > 0) {
            int ft_addr_start = i * 16;
            int joint_type = baxter_approx_joint_types[i];

            if (joint_type <= Z_PRISM) {
                prism_fn(&baxter_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col, joint_type);
            }
            else if (joint_type == X_ROT) {
                xrot_fn(&baxter_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            else if (joint_type == Y_ROT) {
                yrot_fn(&baxter_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            else if (joint_type == Z_ROT) {
                zrot_fn(&baxter_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            
            for (int r=0; r<4; r++){
                T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
            }
            for (int r=0; r<4; r++){
                T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
            }
        }
        __syncwarp();
        while (baxter_approx_flattened_joint_to_spheres[joint_to_sphere_ind] != -1) {
            int sphere_ind = baxter_approx_flattened_joint_to_spheres[joint_to_sphere_ind];
            if (col_ind < 3) {
                // sphere sphere_ind, robot batch_ind (BATCH_SIZE robots), coord col_ind
                sphere_pos_approx[sphere_ind * BATCH_SIZE * 3 + batch_ind * 3 + col_ind] = 
                    T_base[T_memory_idx*16 + col_ind] * baxter_approx_spheres_array[sphere_ind].x +
                    T_base[T_memory_idx*16 + col_ind + M] * baxter_approx_spheres_array[sphere_ind].y +
                    T_base[T_memory_idx*16 + col_ind + M*2] * baxter_approx_spheres_array[sphere_ind].z +
                    T_base[T_memory_idx*16 + col_ind + M*3];
            }
            joint_to_sphere_ind++;
        }
        joint_to_sphere_ind++;
        __syncthreads();
    }
}

// 4 threads per discretized motion for self-collision check
template <>
__device__ bool self_collision_check_approx<ppln::robots::Baxter>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool out = true;
    for (int i = thread_ind; i < BAXTER_APPROX_SELF_CC_RANGE_COUNT; i+=4) {
        int sphere_1_ind = baxter_approx_self_cc_ranges[i][0];
        float sphere_1[3] = {
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
        };
        for (int j = baxter_approx_self_cc_ranges[i][1]; j <= baxter_approx_self_cc_ranges[i][2]; j++) {
            float sphere_2[3] = {
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            if (sphere_sphere_self_collision(
                sphere_1[0], sphere_1[1], sphere_1[2], baxter_approx_spheres_array[sphere_1_ind].w,
                sphere_2[0], sphere_2[1], sphere_2[2], baxter_approx_spheres_array[j].w
            )){
                atomicAdd((int*)&joint_in_collision[20*batch_ind + baxter_approx_sphere_to_joint[sphere_1_ind]], 1);
                out = false;
            }
        } 
    }
    return out;
}

// 4 threads per discretized motion for env collision check
template <>
__device__ bool env_collision_check_approx<ppln::robots::Baxter>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool out = true;

    for (int i = thread_ind; i < BAXTER_APPROX_SPHERE_COUNT; i += 4){
        // sphere i, robot batch_ind (32 robots)
        if ( 
            sphere_environment_in_collision(
                env,
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                baxter_approx_spheres_array[i].w
            )
        ) {
            atomicAdd((int*)&joint_in_collision[20*batch_ind + baxter_approx_sphere_to_joint[i]],1);
            out = false;
        } 
    }
    return out;
}
}

