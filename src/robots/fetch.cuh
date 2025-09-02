namespace ppln::collision {





    #define FETCH_APPROX_SPHERE_COUNT 15
    #define FETCH_APPROX_JOINT_COUNT 9
    #define FETCH_APPROX_SELF_CC_RANGE_COUNT 15
    #define FIXED -1
    #define X_PRISM 0
    #define Y_PRISM 1
    #define Z_PRISM 2
    #define X_ROT 3
    #define Y_ROT 4
    #define Z_ROT 5
    #define BATCH_SIZE 16
    
    __device__ __constant__ float4 fetch_approx_spheres_array[15] = {
        { -0.02f, 0.0f, 0.188f, 0.34f },
        { -0.186875f, 0.0f, 0.587425f, 0.277f },
        { -0.1f, 0.0f, 0.3f, 0.308f },
        { 0.100125f, 0.0f, 0.662001f, 0.197f },
        { 0.06f, -0.015f, 0.03f, 0.124f },
        { 0.063f, 0.019f, 0.0f, 0.134f },
        { 0.056f, -0.02f, 0.0f, 0.134f },
        { 0.071f, 0.021f, 0.0f, 0.127f },
        { 0.064f, -0.026f, 0.0f, 0.124f },
        { 0.029f, 0.017f, 0.0f, 0.09f },
        { -0.015f, 0.0f, 0.0f, 0.07f },
        { 0.08145f, 0.0f, 0.0f, 0.075f },
        { 0.16645f, -0.056425f, 0.0f, 0.03f },
        { 0.16645f, 0.056425f, 0.0f, 0.03f },
        { 0.1f, 0.0f, 0.24f, 0.07f }
    };
    
    __device__ __constant__ float fetch_approx_fixed_transforms[] = {
        // joint 0
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 1
        1.0, 0.0, 0.0, -0.086875,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.37743,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 2
        1.0, 0.0, 0.0, 0.119525,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.34858,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 3
        1.0, 0.0, 0.0, 0.117,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.06,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 4
        1.0, 0.0, 0.0, 0.219,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 5
        1.0, 0.0, 0.0, 0.133,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 6
        1.0, 0.0, 0.0, 0.197,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 7
        1.0, 0.0, 0.0, 0.1245,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 8
        1.0, 0.0, 0.0, 0.1385,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        
    };
    
    __device__ __constant__ int fetch_approx_sphere_to_joint[15] = {
        0,
        0,
        1,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        8,
        8,
        8,
        1
    };
    
    __device__ __constant__ int fetch_approx_flattened_joint_to_spheres[24] = {
        0,
        1,
        -1,
        2,
        3,
        14,
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
        -1,
        9,
        -1,
        10,
        11,
        12,
        13,
        -1
    };
    
    __device__ __constant__ int fetch_approx_joint_types[] = {
        3,
        2,
        5,
        4,
        3,
        4,
        3,
        4,
        3
    };
    
    __device__ __constant__ int fetch_approx_self_cc_ranges[15][3] = {
        { 0, 7, 13 },
        { 1, 6, 13 },
        { 2, 6, 13 },
        { 3, 6, 13 },
        { 4, 6, 6 },
        { 4, 10, 13 },
        { 5, 11, 14 },
        { 6, 14, 14 },
        { 7, 14, 14 },
        { 8, 14, 14 },
        { 9, 14, 14 },
        { 10, 14, 14 },
        { 11, 14, 14 },
        { 12, 14, 14 },
        { 13, 14, 14 }
    };
    
    __device__ __constant__ int fetch_approx_joint_parents[9] = {
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
    };
    
    __device__ __constant__ int fetch_approx_T_memory_idx[9] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    };
    
    __device__ __constant__ int fetch_approx_dfs_order[9] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8
    };
    
    __device__ __constant__ int fetch_approx_joint_id_to_dof[9] = {
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
    };
    
    template <>
    __device__ void fk_approx<ppln::robots::Fetch>(
        const float* q,
        volatile float* sphere_pos_approx, // 15 spheres x 16 robots x 3 coordinates (each column is a robot)
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
    
        for (int j = 0; j < FETCH_APPROX_JOINT_COUNT; ++j) {
            int i = fetch_approx_dfs_order[j];
            float T_col_tmp[4];
            int parent_idx = fetch_approx_joint_parents[i];
            int T_memory_idx_parent = fetch_approx_T_memory_idx[parent_idx];
            int T_memory_idx = fetch_approx_T_memory_idx[i];
            int q_idx = fetch_approx_joint_id_to_dof[i];
            if (j > 0) {
                int ft_addr_start = i * 16;
                int joint_type = fetch_approx_joint_types[i];
    
                if (joint_type <= Z_PRISM) {
                    prism_fn(&fetch_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col, joint_type);
                }
                else if (joint_type == X_ROT) {
                    xrot_fn(&fetch_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                }
                else if (joint_type == Y_ROT) {
                    yrot_fn(&fetch_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                }
                else if (joint_type == Z_ROT) {
                    zrot_fn(&fetch_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                }
                
                for (int r=0; r<4; r++){
                    T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
                }
                for (int r=0; r<4; r++){
                    T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
                }
            }
            __syncwarp();
            while (fetch_approx_flattened_joint_to_spheres[joint_to_sphere_ind] != -1) {
                int sphere_ind = fetch_approx_flattened_joint_to_spheres[joint_to_sphere_ind];
                if (col_ind < 3) {
                    // sphere sphere_ind, robot batch_ind (BATCH_SIZE robots), coord col_ind
                    sphere_pos_approx[sphere_ind * BATCH_SIZE * 3 + batch_ind * 3 + col_ind] = 
                        T_base[T_memory_idx*16 + col_ind] * fetch_approx_spheres_array[sphere_ind].x +
                        T_base[T_memory_idx*16 + col_ind + M] * fetch_approx_spheres_array[sphere_ind].y +
                        T_base[T_memory_idx*16 + col_ind + M*2] * fetch_approx_spheres_array[sphere_ind].z +
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
    __device__ bool self_collision_check_approx<ppln::robots::Fetch>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool out = true;
        for (int i = thread_ind; i < FETCH_APPROX_SELF_CC_RANGE_COUNT; i+=4) {
            int sphere_1_ind = fetch_approx_self_cc_ranges[i][0];
            float sphere_1[3] = {
                sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            for (int j = fetch_approx_self_cc_ranges[i][1]; j <= fetch_approx_self_cc_ranges[i][2]; j++) {
                float sphere_2[3] = {
                    sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
                };
                if (sphere_sphere_self_collision(
                    sphere_1[0], sphere_1[1], sphere_1[2], fetch_approx_spheres_array[sphere_1_ind].w,
                    sphere_2[0], sphere_2[1], sphere_2[2], fetch_approx_spheres_array[j].w
                )){
                    atomicAdd((int*)&joint_in_collision[20*batch_ind + fetch_approx_sphere_to_joint[sphere_1_ind]], 1);
                    out = false;
                }
            } 
        }
        return out;
    }
    
    // 4 threads per discretized motion for env collision check
    template <>
    __device__ bool env_collision_check_approx<ppln::robots::Fetch>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool out = true;
    
        for (int i = thread_ind; i < FETCH_APPROX_SPHERE_COUNT; i += 4){
            // sphere i, robot batch_ind (32 robots)
            if ( 
                sphere_environment_in_collision(
                    env,
                    sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                    fetch_approx_spheres_array[i].w
                )
            ) {
                atomicAdd((int*)&joint_in_collision[20*batch_ind + fetch_approx_sphere_to_joint[i]],1);
                out = false;
            } 
        }
        return out;
    }
    
    
    
    
    #define FETCH_SPHERE_COUNT 111
    #define FETCH_JOINT_COUNT 9
    #define FETCH_SELF_CC_RANGE_COUNT 114
    #define FIXED -1
    #define X_PRISM 0
    #define Y_PRISM 1
    #define Z_PRISM 2
    #define X_ROT 3
    #define Y_ROT 4
    #define Z_ROT 5
    #define BATCH_SIZE 16
    
    __device__ __constant__ float4 fetch_spheres_array[111] = {
        { -0.12f, 0.0f, 0.182f, 0.24f },
        { 0.225f, 0.0f, 0.31f, 0.066f },
        { 0.08f, -0.06f, 0.16f, 0.22f },
        { 0.215f, -0.07f, 0.31f, 0.066f },
        { 0.185f, -0.135f, 0.31f, 0.066f },
        { 0.13f, -0.185f, 0.31f, 0.066f },
        { 0.065f, -0.2f, 0.31f, 0.066f },
        { 0.01f, -0.2f, 0.31f, 0.066f },
        { 0.08f, 0.06f, 0.16f, 0.22f },
        { 0.215f, 0.07f, 0.31f, 0.066f },
        { 0.185f, 0.135f, 0.31f, 0.066f },
        { 0.13f, 0.185f, 0.31f, 0.066f },
        { 0.065f, 0.2f, 0.31f, 0.066f },
        { 0.01f, 0.2f, 0.31f, 0.066f },
        { -0.186875f, -0.07f, 0.727425f, 0.12f },
        { -0.186875f, 0.07f, 0.727425f, 0.12f },
        { -0.186875f, -0.07f, 0.577425f, 0.12f },
        { -0.186875f, 0.07f, 0.577425f, 0.12f },
        { -0.186875f, 0.07f, 0.447425f, 0.12f },
        { -0.186875f, -0.07f, 0.447425f, 0.12f },
        { -0.1f, -0.05f, 0.15f, 0.15f },
        { -0.1f, 0.05f, 0.15f, 0.15f },
        { -0.1f, 0.05f, 0.3f, 0.15f },
        { -0.1f, 0.05f, 0.45f, 0.15f },
        { -0.1f, -0.05f, 0.45f, 0.15f },
        { -0.1f, -0.05f, 0.3f, 0.15f },
        { 0.053125f, 0.0f, 0.663001f, 0.15f },
        { 0.198125f, 0.0f, 0.661001f, 0.05f },
        { 0.198125f, -0.0425f, 0.661001f, 0.05f },
        { 0.198125f, 0.0425f, 0.661001f, 0.05f },
        { 0.198125f, 0.085f, 0.661001f, 0.05f },
        { 0.198125f, -0.085f, 0.661001f, 0.05f },
        { 0.115625f, -0.115f, 0.633001f, 0.03f },
        { 0.141125f, -0.115f, 0.633001f, 0.03f },
        { 0.166625f, -0.115f, 0.633001f, 0.03f },
        { 0.192125f, -0.115f, 0.633001f, 0.03f },
        { 0.115625f, -0.115f, 0.688001f, 0.03f },
        { 0.141125f, -0.115f, 0.688001f, 0.03f },
        { 0.166625f, -0.115f, 0.688001f, 0.03f },
        { 0.192125f, -0.115f, 0.688001f, 0.03f },
        { 0.213125f, -0.115f, 0.678001f, 0.03f },
        { 0.221125f, -0.115f, 0.660501f, 0.03f },
        { 0.213125f, -0.115f, 0.643001f, 0.03f },
        { 0.115625f, 0.115f, 0.633001f, 0.03f },
        { 0.141125f, 0.115f, 0.633001f, 0.03f },
        { 0.166625f, 0.115f, 0.633001f, 0.03f },
        { 0.192125f, 0.115f, 0.633001f, 0.03f },
        { 0.115625f, 0.115f, 0.688001f, 0.03f },
        { 0.141125f, 0.115f, 0.688001f, 0.03f },
        { 0.166625f, 0.115f, 0.688001f, 0.03f },
        { 0.192125f, 0.115f, 0.688001f, 0.03f },
        { 0.213125f, 0.115f, 0.678001f, 0.03f },
        { 0.221125f, 0.115f, 0.660501f, 0.03f },
        { 0.213125f, 0.115f, 0.643001f, 0.03f },
        { 0.0f, 0.0f, 0.0f, 0.055f },
        { 0.025f, -0.015f, 0.035f, 0.055f },
        { 0.05f, -0.03f, 0.06f, 0.055f },
        { 0.12f, -0.03f, 0.06f, 0.055f },
        { 0.025f, 0.04f, 0.025f, 0.04f },
        { -0.025f, 0.04f, -0.025f, 0.04f },
        { 0.025f, 0.04f, -0.025f, 0.04f },
        { -0.025f, 0.04f, 0.025f, 0.04f },
        { 0.08f, 0.0f, 0.0f, 0.055f },
        { 0.11f, 0.0f, 0.0f, 0.055f },
        { 0.14f, 0.0f, 0.0f, 0.055f },
        { -0.02f, 0.0f, 0.0f, 0.055f },
        { 0.03f, 0.0f, 0.0f, 0.055f },
        { 0.08f, 0.0f, 0.0f, 0.055f },
        { 0.11f, -0.045f, 0.02f, 0.03f },
        { 0.11f, -0.045f, -0.02f, 0.03f },
        { 0.155f, -0.045f, 0.02f, 0.03f },
        { 0.155f, -0.045f, -0.02f, 0.03f },
        { 0.13f, 0.0f, 0.0f, 0.055f },
        { 0.02f, 0.045f, 0.02f, 0.03f },
        { 0.02f, 0.045f, -0.02f, 0.03f },
        { -0.02f, 0.045f, 0.02f, 0.03f },
        { -0.02f, 0.045f, -0.02f, 0.03f },
        { 0.08f, 0.0f, 0.0f, 0.055f },
        { 0.14f, 0.0f, 0.0f, 0.055f },
        { 0.0f, 0.0f, 0.0f, 0.055f },
        { 0.05f, -0.06f, 0.02f, 0.03f },
        { 0.05f, -0.06f, -0.02f, 0.03f },
        { 0.1f, -0.06f, 0.02f, 0.03f },
        { 0.1f, -0.06f, -0.02f, 0.03f },
        { 0.15f, -0.06f, 0.02f, 0.03f },
        { 0.15f, -0.06f, -0.02f, 0.03f },
        { 0.0f, 0.0f, 0.0f, 0.055f },
        { 0.06f, 0.0f, 0.0f, 0.055f },
        { 0.02f, 0.045f, 0.02f, 0.03f },
        { 0.02f, 0.045f, -0.02f, 0.03f },
        { -0.02f, 0.045f, 0.02f, 0.03f },
        { -0.02f, 0.045f, -0.02f, 0.03f },
        { -0.03f, 0.0f, 0.0f, 0.055f },
        { 0.0f, 0.0f, 0.0f, 0.055f },
        { 0.09645f, 0.02f, 0.0f, 0.05f },
        { 0.09645f, -0.02f, 0.0f, 0.05f },
        { 0.06645f, 0.02f, 0.0f, 0.05f },
        { 0.06645f, -0.02f, 0.0f, 0.05f },
        { 0.18345f, -0.056925f, -0.005f, 0.012f },
        { 0.18345f, -0.056925f, 0.005f, 0.012f },
        { 0.16645f, -0.056925f, -0.005f, 0.012f },
        { 0.16645f, -0.056925f, 0.005f, 0.012f },
        { 0.14945f, -0.056925f, -0.005f, 0.012f },
        { 0.14945f, -0.056925f, 0.005f, 0.012f },
        { 0.18345f, 0.056925f, -0.005f, 0.012f },
        { 0.18345f, 0.056925f, 0.005f, 0.012f },
        { 0.16645f, 0.056925f, -0.005f, 0.012f },
        { 0.16645f, 0.056925f, 0.005f, 0.012f },
        { 0.14945f, 0.056925f, -0.005f, 0.012f },
        { 0.14945f, 0.056925f, 0.005f, 0.012f },
        { 0.1f, 0.0f, 0.24f, 0.07f }
    };
    
    __device__ __constant__ float fetch_fixed_transforms[] = {
        // joint 0
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 1
        1.0, 0.0, 0.0, -0.086875,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.37743,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 2
        1.0, 0.0, 0.0, 0.119525,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.34858,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 3
        1.0, 0.0, 0.0, 0.117,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.06,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 4
        1.0, 0.0, 0.0, 0.219,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 5
        1.0, 0.0, 0.0, 0.133,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 6
        1.0, 0.0, 0.0, 0.197,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 7
        1.0, 0.0, 0.0, 0.1245,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 8
        1.0, 0.0, 0.0, 0.1385,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        
    };
    
    __device__ __constant__ int fetch_sphere_to_joint[111] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
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
        1,
        1,
        1,
        1,
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
        3,
        3,
        3,
        4,
        4,
        4,
        4,
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
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        1
    };
    
    __device__ __constant__ int fetch_flattened_joint_to_spheres[120] = {
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
        14,
        15,
        16,
        17,
        18,
        19,
        -1,
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
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        110,
        -1,
        54,
        55,
        56,
        57,
        -1,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        -1,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        -1,
        73,
        74,
        75,
        76,
        77,
        78,
        -1,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        -1,
        86,
        87,
        88,
        89,
        90,
        91,
        -1,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        -1
    };
    
    __device__ __constant__ int fetch_joint_types[] = {
        3,
        2,
        5,
        4,
        3,
        4,
        3,
        4,
        3
    };
    
    __device__ __constant__ int fetch_self_cc_ranges[114][3] = {
        { 0, 73, 109 },
        { 1, 73, 109 },
        { 2, 73, 109 },
        { 3, 73, 109 },
        { 4, 73, 109 },
        { 5, 73, 109 },
        { 6, 73, 109 },
        { 7, 73, 109 },
        { 8, 73, 109 },
        { 9, 73, 109 },
        { 10, 73, 109 },
        { 11, 73, 109 },
        { 12, 73, 109 },
        { 13, 73, 109 },
        { 14, 65, 109 },
        { 15, 65, 109 },
        { 16, 65, 109 },
        { 17, 65, 109 },
        { 18, 65, 109 },
        { 19, 65, 109 },
        { 20, 65, 109 },
        { 21, 65, 109 },
        { 22, 65, 109 },
        { 23, 65, 109 },
        { 24, 65, 109 },
        { 25, 65, 109 },
        { 26, 65, 109 },
        { 27, 65, 109 },
        { 28, 65, 109 },
        { 29, 65, 109 },
        { 30, 65, 109 },
        { 31, 65, 109 },
        { 32, 65, 109 },
        { 33, 65, 109 },
        { 34, 65, 109 },
        { 35, 65, 109 },
        { 36, 65, 109 },
        { 37, 65, 109 },
        { 38, 65, 109 },
        { 39, 65, 109 },
        { 40, 65, 109 },
        { 41, 65, 109 },
        { 42, 65, 109 },
        { 43, 65, 109 },
        { 44, 65, 109 },
        { 45, 65, 109 },
        { 46, 65, 109 },
        { 47, 65, 109 },
        { 48, 65, 109 },
        { 49, 65, 109 },
        { 50, 65, 109 },
        { 51, 65, 109 },
        { 52, 65, 109 },
        { 53, 65, 109 },
        { 54, 65, 72 },
        { 54, 92, 109 },
        { 55, 65, 72 },
        { 55, 92, 109 },
        { 56, 65, 72 },
        { 56, 92, 109 },
        { 57, 65, 72 },
        { 57, 92, 109 },
        { 58, 94, 110 },
        { 59, 94, 110 },
        { 60, 94, 110 },
        { 61, 94, 110 },
        { 62, 94, 110 },
        { 63, 94, 110 },
        { 64, 94, 110 },
        { 65, 110, 110 },
        { 66, 110, 110 },
        { 67, 110, 110 },
        { 68, 110, 110 },
        { 69, 110, 110 },
        { 70, 110, 110 },
        { 71, 110, 110 },
        { 72, 110, 110 },
        { 73, 110, 110 },
        { 74, 110, 110 },
        { 75, 110, 110 },
        { 76, 110, 110 },
        { 77, 110, 110 },
        { 78, 110, 110 },
        { 79, 110, 110 },
        { 80, 110, 110 },
        { 81, 110, 110 },
        { 82, 110, 110 },
        { 83, 110, 110 },
        { 84, 110, 110 },
        { 85, 110, 110 },
        { 86, 110, 110 },
        { 87, 110, 110 },
        { 88, 110, 110 },
        { 89, 110, 110 },
        { 90, 110, 110 },
        { 91, 110, 110 },
        { 92, 110, 110 },
        { 93, 110, 110 },
        { 94, 110, 110 },
        { 95, 110, 110 },
        { 96, 110, 110 },
        { 97, 110, 110 },
        { 98, 110, 110 },
        { 99, 110, 110 },
        { 100, 110, 110 },
        { 101, 110, 110 },
        { 102, 110, 110 },
        { 103, 110, 110 },
        { 104, 110, 110 },
        { 105, 110, 110 },
        { 106, 110, 110 },
        { 107, 110, 110 },
        { 108, 110, 110 },
        { 109, 110, 110 }
    };
    
    __device__ __constant__ int fetch_joint_parents[9] = {
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
    };
    
    __device__ __constant__ int fetch_T_memory_idx[9] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    };
    
    __device__ __constant__ int fetch_dfs_order[9] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8
    };
    
    __device__ __constant__ int fetch_joint_id_to_dof[9] = {
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
    };
    
    template <>
    __device__ void fk<ppln::robots::Fetch>(
        const float* q,
        volatile float* sphere_pos, // 111 spheres x 16 robots x 3 coordinates (each column is a robot)
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
    
        for (int j = 0; j < FETCH_JOINT_COUNT; ++j) {
            int i = fetch_dfs_order[j];
            float T_col_tmp[4];
            int parent_idx = fetch_joint_parents[i];
            int T_memory_idx_parent = fetch_T_memory_idx[parent_idx];
            int T_memory_idx = fetch_T_memory_idx[i];
            int q_idx = fetch_joint_id_to_dof[i];
            if (j > 0) {
                int ft_addr_start = i * 16;
                int joint_type = fetch_joint_types[i];
    
                if (joint_type <= Z_PRISM) {
                    prism_fn(&fetch_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col, joint_type);
                }
                else if (joint_type == X_ROT) {
                    xrot_fn(&fetch_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                }
                else if (joint_type == Y_ROT) {
                    yrot_fn(&fetch_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                }
                else if (joint_type == Z_ROT) {
                    zrot_fn(&fetch_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                }
                
                for (int r=0; r<4; r++){
                    T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
                }
                for (int r=0; r<4; r++){
                    T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
                }
            }
            __syncwarp();
            while (fetch_flattened_joint_to_spheres[joint_to_sphere_ind] != -1) {
                int sphere_ind = fetch_flattened_joint_to_spheres[joint_to_sphere_ind];
                if (col_ind < 3) {
                    // sphere sphere_ind, robot batch_ind (BATCH_SIZE robots), coord col_ind
                    // printf("index: %d\n", sphere_ind * BATCH_SIZE * 3 + batch_ind * 3 + col_ind);
                    sphere_pos[sphere_ind * BATCH_SIZE * 3 + batch_ind * 3 + col_ind] = 
                        T_base[T_memory_idx*16 + col_ind] * fetch_spheres_array[sphere_ind].x +
                        T_base[T_memory_idx*16 + col_ind + M] * fetch_spheres_array[sphere_ind].y +
                        T_base[T_memory_idx*16 + col_ind + M*2] * fetch_spheres_array[sphere_ind].z +
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
    __device__ bool self_collision_check<ppln::robots::Fetch>(volatile float* sphere_pos, volatile int* joint_in_collision, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
    
        for (int i = thread_ind; i < FETCH_SELF_CC_RANGE_COUNT; i += 4) {
            int sphere_1_ind = fetch_self_cc_ranges[i][0];
            if (joint_in_collision[20*batch_ind + fetch_sphere_to_joint[sphere_1_ind]] == 0) continue;
            float sphere_1[3] = {
                sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            for (int j = fetch_self_cc_ranges[i][1]; j <= fetch_self_cc_ranges[i][2]; j++) {
                float sphere_2[3] = {
                    sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
                };
                if (sphere_sphere_self_collision(
                    sphere_1[0], sphere_1[1], sphere_1[2], fetch_spheres_array[sphere_1_ind].w,
                    sphere_2[0], sphere_2[1], sphere_2[2], fetch_spheres_array[j].w
                )){
                    return false;
                }
            }
        }
        return true;
    
    }
    
    // 4 threads per discretized motion for env collision check
    template <>
    __device__ bool env_collision_check<ppln::robots::Fetch>(volatile float* sphere_pos, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
    
        for (int i = thread_ind; i < FETCH_SPHERE_COUNT; i += 4){
            // sphere i, robot batch_ind (16 robots)
            if (joint_in_collision[20*batch_ind + fetch_sphere_to_joint[i]] > 0 && 
                sphere_environment_in_collision(
                    env,
                    sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                    fetch_spheres_array[i].w
                )
            ) {
                return false;
            } 
        }
        return true;
    }
    }
    