namespace ppln::collision {





#define UR5_APPROX_SPHERE_COUNT 17
#define UR5_APPROX_JOINT_COUNT 7
#define UR5_APPROX_SELF_CC_RANGE_COUNT 5
#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5
#define BATCH_SIZE 16

__device__ __constant__ float4 ur5_approx_spheres_array[17] = {
    { 0.0f, 0.0f, 0.9144f, 0.08f },
    { 0.0f, 0.0f, 0.0f, 0.08f },
    { 0.0f, 0.0f, 0.21f, 0.29f },
    { 0.002f, 0.003f, 0.185f, 0.265f },
    { 0.0f, 0.09f, 0.0f, 0.07f },
    { 0.0f, 0.0f, 0.09f, 0.07f },
    { 0.0f, 0.06f, 0.0f, 0.04f },
    { 1.6e-05f, 0.0973f, 0.0f, 0.04f },
    { -3.2e-05f, 0.1573f, 0.000532f, 0.06f },
    { 0.032629f, 0.206633f, 0.000571f, 0.02f },
    { 0.047184f, 0.245142f, 0.000602f, 0.033f },
    { 0.030551f, 0.180116f, 0.00055f, 0.02f },
    { 0.062228f, 0.198208f, 0.000564f, 0.035f },
    { -0.032771f, 0.206581f, 0.000571f, 0.02f },
    { -0.046908f, 0.245428f, 0.000602f, 0.033f },
    { -0.030651f, 0.180068f, 0.00055f, 0.02f },
    { -0.062375f, 0.198441f, 0.000565f, 0.035f }
};

__device__ __constant__ float ur5_approx_fixed_transforms[] = {
    // joint 0
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 1
    0.000796, -1.0, 0.0, 0.0,
    1.0, 0.000796, 0.0, 0.0,
    0.0, 0.0, 1.0, 1.003559,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 2
    0.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 0.13585,
    -1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 3
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, -0.1197,
    0.0, 0.0, 1.0, 0.425,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 4
    0.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0, 0.39225,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 5
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.093,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 6
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.09465,
    0.0, 0.0, 0.0, 1.0,
    
    
};

__device__ __constant__ int ur5_approx_sphere_to_joint[17] = {
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6
};

__device__ __constant__ int ur5_approx_flattened_joint_to_spheres[24] = {
    0,
    -1,
    1,
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
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    -1
};

__device__ __constant__ int ur5_approx_joint_types[] = {
    3,
    5,
    4,
    4,
    4,
    5,
    4
};

__device__ __constant__ int ur5_approx_self_cc_ranges[5][3] = {
    { 0, 2, 16 },
    { 1, 3, 16 },
    { 2, 4, 16 },
    { 3, 5, 16 },
    { 4, 7, 7 }
};

__device__ __constant__ int ur5_approx_joint_parents[7] = {
    0,
    0,
    1,
    2,
    3,
    4,
    5
};

__device__ __constant__ int ur5_approx_T_memory_idx[7] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0
};

__device__ __constant__ int ur5_approx_dfs_order[7] = {
    0,
    1,
    2,
    3,
    4,
    5,
    6
};

__device__ __constant__ int ur5_approx_joint_id_to_dof[7] = {
    18446744073709551615,
    0,
    1,
    2,
    3,
    4,
    5
};

template <>
__device__ void fk_approx<ppln::robots::Ur5>(
    const float* q,
    volatile float* sphere_pos_approx, // 17 spheres x 16 robots x 3 coordinates (each column is a robot)
    float *T, // 16 robots x 1 x 4x4 transform matrix , column major
    const int tid
)
{
    // every 4 threads are responsible for one column of the transform matrix T
    // make_transform will calculate the necessary column of T_step needed for the thread
    const int col_ind = tid % 4;
    const int batch_ind = tid / 4;

    int T_offset = batch_ind * 1 * 16;
    float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
    float *T_base = T + T_offset; // 4x4 transform matrix for the batch
    
    #pragma unroll
    for (int i = 0; i < 1; ++i) {
        float *T_col_i = T_base + i * 16 + col_ind * 4;
        for (int r=0; r<4; r++) {
            T_col_i[r] = 0.0f;
        }
        T_col_i[col_ind] = 1.0f;
    }
    __syncthreads();

    int joint_to_sphere_ind = 0;

    for (int j = 0; j < UR5_APPROX_JOINT_COUNT; ++j) {
        int i = ur5_approx_dfs_order[j];
        float T_col_tmp[4];
        int parent_idx = ur5_approx_joint_parents[i];
        int T_memory_idx_parent = ur5_approx_T_memory_idx[parent_idx];
        int T_memory_idx = ur5_approx_T_memory_idx[i];
        int q_idx = ur5_approx_joint_id_to_dof[i];
        if (j > 0) {
            int ft_addr_start = i * 16;
            int joint_type = ur5_approx_joint_types[i];

            if (joint_type <= Z_PRISM) {
                prism_fn(&ur5_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col, joint_type);
            }
            else if (joint_type == X_ROT) {
                xrot_fn(&ur5_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            else if (joint_type == Y_ROT) {
                yrot_fn(&ur5_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            else if (joint_type == Z_ROT) {
                zrot_fn(&ur5_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            
            for (int r=0; r<4; r++){
                T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
            }
            for (int r=0; r<4; r++){
                T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
            }
        }
        __syncwarp();
        while (ur5_approx_flattened_joint_to_spheres[joint_to_sphere_ind] != -1) {
            int sphere_ind = ur5_approx_flattened_joint_to_spheres[joint_to_sphere_ind];
            if (col_ind < 3) {
                // sphere sphere_ind, robot batch_ind (BATCH_SIZE robots), coord col_ind
                sphere_pos_approx[sphere_ind * BATCH_SIZE * 3 + batch_ind * 3 + col_ind] = 
                    T_base[T_memory_idx*16 + col_ind] * ur5_approx_spheres_array[sphere_ind].x +
                    T_base[T_memory_idx*16 + col_ind + M] * ur5_approx_spheres_array[sphere_ind].y +
                    T_base[T_memory_idx*16 + col_ind + M*2] * ur5_approx_spheres_array[sphere_ind].z +
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
__device__ bool self_collision_check_approx<ppln::robots::Ur5>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool out = true;
    for (int i = thread_ind; i < UR5_APPROX_SELF_CC_RANGE_COUNT; i+=4) {
        int sphere_1_ind = ur5_approx_self_cc_ranges[i][0];
        float sphere_1[3] = {
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
        };
        for (int j = ur5_approx_self_cc_ranges[i][1]; j <= ur5_approx_self_cc_ranges[i][2]; j++) {
            float sphere_2[3] = {
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            if (sphere_sphere_self_collision(
                sphere_1[0], sphere_1[1], sphere_1[2], ur5_approx_spheres_array[sphere_1_ind].w,
                sphere_2[0], sphere_2[1], sphere_2[2], ur5_approx_spheres_array[j].w
            )){
                atomicAdd((int*)&joint_in_collision[20*batch_ind + ur5_approx_sphere_to_joint[sphere_1_ind]], 1);
                out = false;
            }
        } 
    }
    return out;
}

// 4 threads per discretized motion for env collision check
template <>
__device__ bool env_collision_check_approx<ppln::robots::Ur5>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool out = true;

    for (int i = thread_ind; i < UR5_APPROX_SPHERE_COUNT; i += 4){
        // sphere i, robot batch_ind (32 robots)
        if ( 
            sphere_environment_in_collision(
                env,
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                ur5_approx_spheres_array[i].w
            )
        ) {
            atomicAdd((int*)&joint_in_collision[20*batch_ind + ur5_approx_sphere_to_joint[i]],1);
            out = false;
        } 
    }
    return out;
}




#define UR5_SPHERE_COUNT 40
#define UR5_JOINT_COUNT 7
#define UR5_SELF_CC_RANGE_COUNT 19
#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5
#define BATCH_SIZE 16

__device__ __constant__ float4 ur5_spheres_array[40] = {
    { 0.0f, 0.0f, 0.9144f, 0.08f },
    { 0.0f, 0.0f, 0.0f, 0.08f },
    { 0.0f, 0.0f, 0.105f, 0.08f },
    { 0.0f, 0.0f, 0.21f, 0.08f },
    { 0.0f, 0.0f, 0.315f, 0.08f },
    { 0.0f, 0.0f, 0.42f, 0.08f },
    { 0.0f, 0.0f, 0.0f, 0.08f },
    { 0.0f, 0.0f, 0.0f, 0.08f },
    { 0.0f, 0.0f, 0.1f, 0.04f },
    { 0.0f, 0.0f, 0.14f, 0.04f },
    { 0.0f, 0.0f, 0.18f, 0.04f },
    { 0.0f, 0.0f, 0.22f, 0.04f },
    { 0.0f, 0.0f, 0.26f, 0.04f },
    { 0.0f, 0.0f, 0.3f, 0.04f },
    { 0.0f, 0.0f, 0.34f, 0.04f },
    { 0.0f, 0.0f, 0.38f, 0.04f },
    { 0.0f, 0.09f, 0.03f, 0.04f },
    { 0.0f, 0.09f, -0.03f, 0.04f },
    { 0.0f, 0.09f, 0.0f, 0.04f },
    { 0.0f, 0.03f, 0.09f, 0.04f },
    { 0.0f, -0.03f, 0.09f, 0.04f },
    { 0.0f, 0.0f, 0.09f, 0.04f },
    { 0.0f, 0.06f, 0.0f, 0.04f },
    { 1.6e-05f, 0.0973f, 0.0f, 0.04f },
    { -4.8e-05f, 0.1773f, 0.000548f, 0.04f },
    { -1.6e-05f, 0.1373f, 0.000516f, 0.04f },
    { 0.032629f, 0.206633f, 0.000571f, 0.02f },
    { 0.047174f, 0.257142f, 0.000611f, 0.015f },
    { 0.047194f, 0.232142f, 0.000591f, 0.015f },
    { 0.030551f, 0.180116f, 0.00055f, 0.02f },
    { 0.062212f, 0.218208f, 0.00058f, 0.015f },
    { 0.062244f, 0.178208f, 0.000548f, 0.015f },
    { 0.062228f, 0.198208f, 0.000564f, 0.015f },
    { -0.032771f, 0.206581f, 0.000571f, 0.02f },
    { -0.046918f, 0.257428f, 0.000612f, 0.015f },
    { -0.046898f, 0.232428f, 0.000592f, 0.015f },
    { -0.030651f, 0.180068f, 0.00055f, 0.02f },
    { -0.062391f, 0.218441f, 0.000581f, 0.015f },
    { -0.062359f, 0.178441f, 0.000549f, 0.015f },
    { -0.062375f, 0.198441f, 0.000565f, 0.015f }
};

__device__ __constant__ float ur5_fixed_transforms[] = {
    // joint 0
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 1
    0.000796, -1.0, 0.0, 0.0,
    1.0, 0.000796, 0.0, 0.0,
    0.0, 0.0, 1.0, 1.003559,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 2
    0.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 0.13585,
    -1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 3
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, -0.1197,
    0.0, 0.0, 1.0, 0.425,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 4
    0.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0, 0.39225,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 5
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.093,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 6
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.09465,
    0.0, 0.0, 0.0, 1.0,
    
    
};

__device__ __constant__ int ur5_sphere_to_joint[40] = {
    0,
    1,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6
};

__device__ __constant__ int ur5_flattened_joint_to_spheres[47] = {
    0,
    -1,
    1,
    -1,
    2,
    3,
    4,
    5,
    6,
    -1,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    -1,
    16,
    17,
    18,
    -1,
    19,
    20,
    21,
    -1,
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
    39,
    -1
};

__device__ __constant__ int ur5_joint_types[] = {
    3,
    5,
    4,
    4,
    4,
    5,
    4
};

__device__ __constant__ int ur5_self_cc_ranges[19][3] = {
    { 0, 2, 39 },
    { 1, 7, 39 },
    { 2, 16, 39 },
    { 3, 16, 39 },
    { 4, 16, 39 },
    { 5, 16, 39 },
    { 6, 16, 39 },
    { 7, 19, 39 },
    { 8, 19, 39 },
    { 9, 19, 39 },
    { 10, 19, 39 },
    { 11, 19, 39 },
    { 12, 19, 39 },
    { 13, 19, 39 },
    { 14, 19, 39 },
    { 15, 19, 39 },
    { 16, 23, 23 },
    { 17, 23, 23 },
    { 18, 23, 23 }
};

__device__ __constant__ int ur5_joint_parents[7] = {
    0,
    0,
    1,
    2,
    3,
    4,
    5
};

__device__ __constant__ int ur5_T_memory_idx[7] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0
};

__device__ __constant__ int ur5_dfs_order[7] = {
    0,
    1,
    2,
    3,
    4,
    5,
    6
};

__device__ __constant__ int ur5_joint_id_to_dof[7] = {
    18446744073709551615,
    0,
    1,
    2,
    3,
    4,
    5
};

template <>
__device__ void fk<ppln::robots::Ur5>(
    const float* q,
    volatile float* sphere_pos, // 40 spheres x 16 robots x 3 coordinates (each column is a robot)
    float *T, // 16 robots x 1 x 4x4 transform matrix , column major
    const int tid
)
{
    // every 4 threads are responsible for one column of the transform matrix T
    // make_transform will calculate the necessary column of T_step needed for the thread
    const int col_ind = tid % 4;
    const int batch_ind = tid / 4;

    int T_offset = batch_ind * 1 * 16;
    float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
    float *T_base = T + T_offset; // 4x4 transform matrix for the batch
    
    #pragma unroll
    for (int i = 0; i < 1; ++i) {
        float *T_col_i = T_base + i * 16 + col_ind * 4;
        for (int r=0; r<4; r++) {
            T_col_i[r] = 0.0f;
        }
        T_col_i[col_ind] = 1.0f;
    }
    __syncthreads();

    int joint_to_sphere_ind = 0;

    for (int j = 0; j < UR5_JOINT_COUNT; ++j) {
        int i = ur5_dfs_order[j];
        float T_col_tmp[4];
        int parent_idx = ur5_joint_parents[i];
        int T_memory_idx_parent = ur5_T_memory_idx[parent_idx];
        int T_memory_idx = ur5_T_memory_idx[i];
        int q_idx = ur5_joint_id_to_dof[i];
        if (j > 0) {
            int ft_addr_start = i * 16;
            int joint_type = ur5_joint_types[i];

            if (joint_type <= Z_PRISM) {
                prism_fn(&ur5_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col, joint_type);
            }
            else if (joint_type == X_ROT) {
                xrot_fn(&ur5_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            else if (joint_type == Y_ROT) {
                yrot_fn(&ur5_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            else if (joint_type == Z_ROT) {
                zrot_fn(&ur5_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            
            for (int r=0; r<4; r++){
                T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
            }
            for (int r=0; r<4; r++){
                T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
            }
        }
        __syncwarp();
        while (ur5_flattened_joint_to_spheres[joint_to_sphere_ind] != -1) {
            int sphere_ind = ur5_flattened_joint_to_spheres[joint_to_sphere_ind];
            if (col_ind < 3) {
                // sphere sphere_ind, robot batch_ind (BATCH_SIZE robots), coord col_ind
                sphere_pos[sphere_ind * BATCH_SIZE * 3 + batch_ind * 3 + col_ind] = 
                    T_base[T_memory_idx*16 + col_ind] * ur5_spheres_array[sphere_ind].x +
                    T_base[T_memory_idx*16 + col_ind + M] * ur5_spheres_array[sphere_ind].y +
                    T_base[T_memory_idx*16 + col_ind + M*2] * ur5_spheres_array[sphere_ind].z +
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
__device__ bool self_collision_check<ppln::robots::Ur5>(volatile float* sphere_pos, volatile int* joint_in_collision, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool has_collision = false;

    for (int i = thread_ind; i < UR5_SELF_CC_RANGE_COUNT; i += 4) {
        if (warp_any_active_mask(has_collision)) return false;
        int sphere_1_ind = ur5_self_cc_ranges[i][0];
        if (joint_in_collision[20*batch_ind + ur5_sphere_to_joint[sphere_1_ind]] == 0) continue;
        float sphere_1[3] = {
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
        };
        for (int j = ur5_self_cc_ranges[i][1]; j <= ur5_self_cc_ranges[i][2]; j++) {
            float sphere_2[3] = {
                sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            if (sphere_sphere_self_collision(
                sphere_1[0], sphere_1[1], sphere_1[2], ur5_spheres_array[sphere_1_ind].w,
                sphere_2[0], sphere_2[1], sphere_2[2], ur5_spheres_array[j].w
            )){
                //return false;
                has_collision=true;
            }
        }
    }
    return !has_collision;

}

// 4 threads per discretized motion for env collision check
template <>
__device__ bool env_collision_check<ppln::robots::Ur5>(volatile float* sphere_pos, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool has_collision=false;

    for (int i = thread_ind; i < UR5_SPHERE_COUNT-UR5_SPHERE_COUNT%4; i += 4){
        // sphere i, robot batch_ind (16 robots)
        if (joint_in_collision[20*batch_ind + ur5_sphere_to_joint[i]] > 0 && 
            sphere_environment_in_collision(
                env,
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                ur5_spheres_array[i].w
            )
        ) {
            //return false;
            has_collision=true;
        } 
        if (warp_any_full_mask(has_collision)) return false;
    }
    int i=UR5_SPHERE_COUNT-1-thread_ind;
    if (joint_in_collision[20*batch_ind + ur5_sphere_to_joint[i]] > 0 && 
        sphere_environment_in_collision(
            env,
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
            ur5_spheres_array[i].w
        )
    ) {
        return false;
    } 
    return true;
}
}
