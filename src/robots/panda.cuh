
namespace ppln::collision {
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
}