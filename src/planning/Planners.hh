#pragma once

#include <vector>
#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include <cmath>

#include "Robots.hh"
#include "src/collision/environment.hh"
#include "pRRTC_settings.hh"

template <typename Robot>
struct PlannerResult {
    bool solved = false;
    std::vector<typename Robot::Configuration> path; 
    int start_tree_size = 0;
    int goal_tree_size = 0;
    int path_length = 0;
    int iters = 0;
    float cost = 0.0;
    std::size_t wall_ns = 0; // wall time of the solve function
    std::size_t kernel_ns = 0; // just kernel runtime
    std::size_t copy_ns = 0; // time to copy start/goals to gpu and copy path and path size back
};

template <typename Robot>
inline float l2dist(typename Robot::Configuration &a, typename Robot::Configuration &b)
{
    float res = 0;
    float diff;
    for (int i = 0; i < a.size(); i++) {
        diff = a[i] - b[i];
        res += diff * diff;
    }
    return sqrt(res);
}

template <typename Robot>
inline void print_cfg_ptr(float *config) {
    for (int i = 0; i < Robot::dimension; i++) {
        std::cout << config[i] << " ";
    }
    std::cout << "\n";
}

template <typename Robot>
inline void print_cfg(typename Robot::Configuration &config) {
    for (int i = 0; i < Robot::dimension; i++) {
        std::cout << config[i] << " ";
    }
    std::cout << "\n";
}

template <typename Robot>
inline void print_cfg_to_ss(typename Robot::Configuration &config, std::stringstream &out) {
    for (int i = 0; i < Robot::dimension; i++) {
        out << config[i] << " ";
    }
    out << "\\n";
}



inline std::size_t get_elapsed_nanoseconds(const std::chrono::time_point<std::chrono::steady_clock> &start)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
}

/* This file handles the declarations of each solve function so that they may be called from .cpp files. Implementation is in .cu files.*/
namespace RRT {
    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment);
}

namespace pRRT {
    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment);
}


namespace nRRT {
    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment);
}

namespace pRRTC {
    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment, pRRTC_settings &settings);
}

namespace nRRTC {
    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment, pRRTC_settings &settings);
}

namespace pwRRTC {
    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment, pRRTC_settings &settings);
}
