# pRRTC: GPU-Parallel RRT-Connect

[![arXiv VAMP](https://img.shields.io/badge/arXiv-2503.06757-b31b1b.svg)](https://arxiv.org/abs/2503.06757)
[![Project Website](https://img.shields.io/badge/Project-Website-blue.svg)](https://commalab.org/papers/pRRTC/)

This repository holds the code for the ICRA 2026 paper [pRRTC: GPU-Parallel RRT-Connect for Fast, Consistent, and Low-Cost Motion Planning.](https://arxiv.org/abs/2503.06757)

We introduce pRRTC, a GPU-based, parallel RRT-Connect-based algorithm. Our approach has three key improvements: 
- Concurrent sampling, expansion and connection of start and goal trees via GPU multithreading
- SIMT-optimized collision checking to quickly validate edges, inspired by the SIMD-optimized validation of [Vector-Accelerated Motion Planning](https://github.com/KavrakiLab/vamp/tree/main)
- Efficient memory management between block- and thread-level parallelism, reducing expensive memory transfer overheads

Our empirical evaluations show that pRRTC achieves a 10x average speedup on constrained reaching tasks. pRRTC also demonstrates a 5.4x reduction in solution time standard deviation and 1.4x improvement in initial path costs compared to state-of-the-art motion planners in complex environments.

## Supported Robots
pRRTC currently supports 7-DoF Franka Emika Panda, 8-DoF Fetch, and 14-DoF Rethink Robotics Baxter. The functions for tracing forward kinematics and collision checking were generated using [Cricket](https://github.com/CoMMALab/cricket.git).

## Building Code
To build pRRTC, follow the instructions below
```
git clone git@github.com:CoMMALab/pRRTC.git
cmake -B build
cmake --build build
```

## Running Code
The repository comes with two benchmarking scripts: evaluate_mbm.cpp and single_mbm.cpp.

evaluate_mbm.cpp allows users to benchmark pRRTC's performance using Panda, Fetch, or Baxter on the entire [MotionBenchMaker](https://github.com/KavrakiLab/motion_bench_maker) dataset. To run evaluate_mbm.cpp:
```
build/evaluate_mbm <robot> <experiment name>
```

single_mbm.cpp allows users to benchmark pRRTC's performance using Panda, Fetch, or Baxter on a single problem of [MotionBenchMaker](https://github.com/KavrakiLab/motion_bench_maker). To run single_mbm.cpp:
```
build/single_mbm <robot> <MBM problem name> <MBM problem index>
```

The [MotionBenchMaker](https://github.com/KavrakiLab/motion_bench_maker) JSON files are generated using the script detailed [here](https://github.com/KavrakiLab/vamp/blob/35080be604aabd4373cc7db8608297afaa446878/resources/README.md#motionbenchmaker-problems).

## Planner Configuration
pRRTC has the following parameters which can be modified in the benchmarking scripts:
- <ins>**max_samples**</ins>: maximum number of samples in trees
- <ins>**max_iters**</ins>: maximum number of planning iterations
- <ins>**num_new_configs**</ins>: amount of new samples generated per iteration
- <ins>**range**</ins>: maximum RRT-Connect extension range
- <ins>**granularity**</ins>: number of discretized motions along an edge during collision checking. Note: this parameter must match the BATCH_SIZE parameter in robot's header file (ex. fetch.cuh) for correct results.
- <ins>**balance**</ins>: whether to enable tree balancing -- 0 for no balancing; 1 for distributed balancing where each iteration may generate samples for one or two trees; 2 for single-sided balancing where each iteration generate samples for one tree only
- <ins>**tree_ratio**</ins>: the threshold for distinguishing which tree is smaller in size -- if balance set to 1, then set tree_ratio to 0.5; if balance set to 2, then set tree_ratio to 1
- <ins>**dynamic_domain**</ins>: whether to enable [dynamic domain sampling](https://ieeexplore.ieee.org/abstract/document/1570709) -- 0 for false; 1 for true
- <ins>**dd_alpha**</ins>: extent to which each radius is enlarged or shrunk per modification
- <ins>**dd_radius**</ins>: starting radius for dynamic domain sampling
- <ins>**dd_min_radius**</ins>: minimum radius for dynamic domain sampling

## Adding a Robot
### Generating FKCC kernels
1. Use [Foam](https://github.com/CoMMALab/foam) to generate two spherized urdfs:
- one with approximate geometry, i.e. 1 sphere per link. Ex. [here](https://github.com/CoMMALab/cricket/blob/gpu-cc-early-exit/resources/panda/panda_spherized_1.urdf)
- one with fine geometry. Ex. [here](https://github.com/CoMMALab/cricket/blob/gpu-cc-early-exit/resources/panda/panda_spherized_1.urdf)

2. Clone [Cricket](https://github.com/CoMMALab/cricket.git) and switch to the `gpu-cc-early-exit` branch.

3. Create a folder under `resources/<robot name>`

4. Add the two spherized urdfs and the robot srdf file to this folder.

5. Create a json config file for approximate fkcc kernel generation. Ex. `resources/robot_approx.json`:
```
{
    "name": "Robot",
    "urdf": "robot/robot_spherized_approx.urdf",
    "srdf": "robot/robot.srdf",
    "end_effector": "robot_grasptarget",
    "batch_size": 16,
    "template": "templates/prrtc_approx_template.hh",
    "subtemplates": [],
    "output": "robot_prrtc_approx.hh"
}
```
Make sure to reference the approximate urdf. Batch size should be equal to the number of discretized collision checks on each extension of pRRTC.

6. Repeat step 5 and create a config file for the main fkcc generation. Ex. `resources/robot_main.json`. See the panda, fetch, and baxter config files for examples.

7. After building cricket run the script `gpu_fkcc_gen.sh robot`. This will put the generated code into a file `robot_fk.hh`.

8. Add this to pRRTC as `src/robot/robot.cuh`, and include it in `src/planning/pRRTC.cu`.

### Integrating the generated code
9. Add a template instantiation for your robot to the bottom of `src/planning/pRRTC.cu`.
10. Add your robot to `src/planning/Robots.hh`.

    a. Generate the robot struct from cricket with `build/fkcc_gen robot_struct.json`.
    Ex. config file `robot_struct.json`:
    ```
    {
        "name": "robot",
        "urdf": "robot/robot_spherized.urdf",
        "srdf": "robot/robot.srdf",
        "end_effector": "robot_grasptarget",
        "resolution": 32,
        "template": "templates/prrtc_robot_template.hh",
        "subtemplates": [],
        "output": "robot_struct.hh"
    }
    ```

    b. copy the generated struct into `src/planning/Robots.hh`.
11. Recompile pRRTC.

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{huang2026prrtc,
    title={pRRTC: GPU-Parallel RRT-Connect for Fast, Consistent, and Low-Cost Motion Planning},
    author={Huang, Chih H and Jadhav, Pranav and Plancher, Brian and Kingston, Zachary},
    booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
    year={2026},
    organization={IEEE}
}
```

## Funding Acknowledgement
This material is based upon work supported by the National Science Foundation (under Award [2411369](https://www.nsf.gov/awardsearch/show-award/?AWD_ID=2411369)). Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect those of the funding organizations.
