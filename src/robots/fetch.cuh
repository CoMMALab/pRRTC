namespace ppln::collision {

    #define FETCH_SPHERE_COUNT 111
    #define FETCH_JOINT_COUNT 9
    #define FETCH_SELF_CC_RANGE_COUNT 113
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
    
    __device__ __constant__ int fetch_sphere_to_joint[] = {
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
    
    __device__ __constant__ int fetch_self_cc_ranges[113][3] = {
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
        { 108, 110, 110 }
    };
    
    template <>
    __device__ void fk<ppln::robots::Fetch>(
        const float* q,
        volatile float* sphere_pos, // 111 spheres x 16 robots x 3 coordinates
        float *T, // 16 robots x 4x4 transform matrix
        const int tid
    )
    {
        // every 4 threads are responsible for one column of the transform matrix T
        // make_transform will calculate the necessary column of T_step needed for the thread
        const int col_ind = tid % 4;
        const int batch_ind = tid / 4;
        int transformed_sphere_ind = 0;
    
        int T_offset = batch_ind * 16;
        float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
        float *T_base = T + T_offset; // 4x4 transform matrix for the batch
        float *T_col = T_base + col_ind*4; // 1x4 column (column major) of the transform matrix for this thread
    
        for (int r=0; r<4; r++){
            T_col[r] = 0;
        }
        T_col[col_ind] = 1;
    
        // loop through each joint, accumulate transformation matrix, and update sphere positions
        for (int i = 0; i < FETCH_JOINT_COUNT; ++i) {
            if (i > 0) {
                int ft_addr_start = i * 16;
                int joint_type = fetch_joint_types[i];
                if (joint_type <= Z_PRISM) {
                    prism_fn(&fetch_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col, joint_type);
                }
                else if (joint_type == X_ROT) {
                    xrot_fn(&fetch_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
                }
                else if (joint_type == Y_ROT) { 
                    yrot_fn(&fetch_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
                }
                else if (joint_type == Z_ROT) {
                    zrot_fn(&fetch_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
                }
    
                for (int r=0; r<4; r++){
                    T_col[r] = dot4_col(&T_base[r], T_step_col);
                }
            }
            
            while (fetch_sphere_to_joint[transformed_sphere_ind]==i) {
                if (col_ind < 3) {
                    // sphere transformed_sphere_ind, robot batch_ind (16 robots), coord col_ind
                    sphere_pos[transformed_sphere_ind * 16 * 3 + batch_ind * 3 + col_ind] = 
                        T_base[col_ind] * fetch_spheres_array[transformed_sphere_ind].x +
                        T_base[col_ind + M] * fetch_spheres_array[transformed_sphere_ind].y +
                        T_base[col_ind + M*2] * fetch_spheres_array[transformed_sphere_ind].z +
                        T_base[col_ind + M*3];
                }
                transformed_sphere_ind++;
            }
        }
    }
    
    // 4 threads per discretized motion for self-collision check
    template <>
    __device__ bool self_collision_check<ppln::robots::Fetch>(volatile float* sphere_pos, volatile int* joint_in_collision, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool has_collision = false;

        for (int i = thread_ind; i < FETCH_SELF_CC_RANGE_COUNT; i += 4) {
            if (warp_any_active_mask(has_collision)) return false;
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
                    //return false;
                    has_collision=true;
                }
            }
        }
        return !has_collision;

    }

    // 4 threads per discretized motion for env collision check
    template <>
    __device__ bool env_collision_check<ppln::robots::Fetch>(volatile float* sphere_pos, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool has_collision=false;

        for (int i = FETCH_SPHERE_COUNT-1-thread_ind; i >=FETCH_SPHERE_COUNT%4; i -= 4){
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
                has_collision=true;
                //return false;
            } 
            if (warp_any_full_mask(has_collision)) return false;
        }

        int i=thread_ind;
        if (joint_in_collision[20*batch_ind + fetch_sphere_to_joint[i]] > 0 && 
            sphere_environment_in_collision(
                env,
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                fetch_spheres_array[i].w
            )
        ) {
            has_collision=true;
            //return false;
        } 

        return !has_collision;
    }

#define FETCH_APPROX_SPHERE_COUNT 15
#define FETCH_APPROX_JOINT_COUNT 9
#define FETCH_APPROX_SELF_CC_RANGE_COUNT 14
#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5

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

__device__ __constant__ int fetch_approx_sphere_to_joint[] = {
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

__device__ __constant__ int fetch_approx_self_cc_ranges[14][3] = {
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
    { 12, 14, 14 }
};

template <>
__device__ void fk_approx<ppln::robots::Fetch>(
    const float* q,
    volatile float* sphere_pos_approx, // 15 spheres x 16 robots x 3 coordinates (each column is a robot)
    float *T, // 16 robots x 4x4 transform matrix , column major
    const int tid
)
{
    // every 4 threads are responsible for one column of the transform matrix T
    // make_transform will calculate the necessary column of T_step needed for the thread
    const int col_ind = tid % 4;
    const int batch_ind = tid / 4;
    int transformed_sphere_ind = 0;

    int T_offset = batch_ind * 16;
    float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
    float *T_base = T + T_offset; // 4x4 transform matrix for the batch
    float *T_col = T_base + col_ind*4; // 1x4 column (column major) of the transform matrix for this thread

    for (int r=0; r<4; r++){
        T_col[r] = 0;
    }
    T_col[col_ind] = 1;

    // loop through each joint, accumulate transformation matrix, and update sphere positions
    for (int i = 0; i < FETCH_APPROX_JOINT_COUNT; ++i) {
        if (i > 0) {
            int ft_addr_start = i * 16;
            int joint_type = fetch_approx_joint_types[i];

            if (joint_type <= Z_PRISM) {
                prism_fn(&fetch_approx_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col, joint_type);
            }
            else if (joint_type == X_ROT) {
                xrot_fn(&fetch_approx_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }
            else if (joint_type == Y_ROT) {
                yrot_fn(&fetch_approx_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }
            else if (joint_type == Z_ROT) {
                zrot_fn(&fetch_approx_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }

            for (int r=0; r<4; r++){
                T_col[r] = dot4_col(&T_base[r], T_step_col);
            }
        }

        while (fetch_approx_sphere_to_joint[transformed_sphere_ind] == i) {
            if (col_ind < 3) {
                // sphere transformed_sphere_ind, robot batch_ind (16 robots), coord col_ind
                sphere_pos_approx[transformed_sphere_ind * 16 * 3 + batch_ind * 3 + col_ind] = 
                    T_base[col_ind] * fetch_approx_spheres_array[transformed_sphere_ind].x +
                    T_base[col_ind + M] * fetch_approx_spheres_array[transformed_sphere_ind].y +
                    T_base[col_ind + M*2] * fetch_approx_spheres_array[transformed_sphere_ind].z +
                    T_base[col_ind + M*3];
            }
            transformed_sphere_ind++;
        }
    }
}

// 4 threads per discretized motion for self-collision check
template <>
__device__ bool self_collision_check_approx<ppln::robots::Fetch>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;

    for (int i = thread_ind; i < FETCH_APPROX_SELF_CC_RANGE_COUNT; i+=4) {
        int sphere_1_ind = fetch_approx_self_cc_ranges[i][0];
        float sphere_1[3] = {
            sphere_pos_approx[sphere_1_ind * 16 * 3 + batch_ind * 3 + 0],
            sphere_pos_approx[sphere_1_ind * 16 * 3 + batch_ind * 3 + 1],
            sphere_pos_approx[sphere_1_ind * 16 * 3 + batch_ind * 3 + 2]
        };
        for (int j = fetch_approx_self_cc_ranges[i][1]; j <= fetch_approx_self_cc_ranges[i][2]; j++) {
            float sphere_2[3] = {
                sphere_pos_approx[j * 16 * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[j * 16 * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[j * 16 * 3 + batch_ind * 3 + 2]
            };
            if (sphere_sphere_self_collision(
                sphere_1[0], sphere_1[1], sphere_1[2], fetch_approx_spheres_array[sphere_1_ind].w,
                sphere_2[0], sphere_2[1], sphere_2[2], fetch_approx_spheres_array[j].w
            )){
                atomicAdd((int*)&joint_in_collision[20*batch_ind + fetch_approx_sphere_to_joint[sphere_1_ind]], 1);
                return false;
            }
        } 
    }
    return true;
}

// 4 threads per discretized motion for env collision check
template <>
__device__ bool env_collision_check_approx<ppln::robots::Fetch>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool out = true;
    
    #pragma unroll
    for (int i=FETCH_APPROX_SPHERE_COUNT/4*thread_ind; i<FETCH_APPROX_SPHERE_COUNT/4*(thread_ind+1); i++){
        // sphere i, robot batch_ind (16 robots)
        if (sphere_environment_in_collision(
            env,
            sphere_pos_approx[i * 16 * 3 + batch_ind * 3 + 0],
            sphere_pos_approx[i * 16 * 3 + batch_ind * 3 + 1],
            sphere_pos_approx[i * 16 * 3 + batch_ind * 3 + 2],
            fetch_approx_spheres_array[i].w
        )) {
            atomicAdd((int*)&joint_in_collision[20*batch_ind + fetch_approx_sphere_to_joint[i]],1);
            out=false;
        } 
    }

    int i = FETCH_APPROX_SPHERE_COUNT-1-thread_ind;
    if (sphere_environment_in_collision(
        env,
        sphere_pos_approx[i * 16 * 3 + batch_ind * 3 + 0],
        sphere_pos_approx[i * 16 * 3 + batch_ind * 3 + 1],
        sphere_pos_approx[i * 16 * 3 + batch_ind * 3 + 2],
        fetch_approx_spheres_array[i].w
    )) {
        atomicAdd((int*)&joint_in_collision[20*batch_ind + fetch_approx_sphere_to_joint[i]],1);
        out=false;
    }
    return out;
}
}
