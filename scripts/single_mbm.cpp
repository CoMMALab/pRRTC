#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/planning/Planners.hh"
#include "src/planning/pRRTC_settings.hh"

using json = nlohmann::json;
using namespace ppln::collision;


Environment<float> problem_dict_to_env(const json& problem, const std::string& name) {
    Environment<float> env{};
    
    std::vector<Sphere<float>> spheres;
    std::vector<Capsule<float>> capsules;
    std::vector<Cuboid<float>> cuboids;
    // Fill spheres
    for (const auto& obj : problem["sphere"]) {
        const json& position = obj["position"];
        Sphere<float> sphere(position[0], position[1], position[2], obj["radius"]);
        sphere.name = obj["name"];
        spheres.push_back(sphere);
    }
    // Handle cylinders based on name
    if (name == "box") {
        for (const auto& obj : problem["cylinder"]) {
            const json& position = obj["position"];
            const json& orientation = obj["orientation_euler_xyz"];
            const float radius = obj["radius"];
            const std::array<float, 3> dims = {radius, radius, radius/2.0f};
            auto cuboid = factory::cuboid::array(
                position, orientation,
                dims
            );
            cuboid.name = obj["name"];
            cuboids.push_back(cuboid);
        }
    } else {
        for (const auto& obj : problem["cylinder"]) {
            const json& position = obj["position"];
            const json& orientation = obj["orientation_euler_xyz"];
            const float radius = obj["radius"];
            const float length = obj["length"];
            auto cylinder = factory::cylinder::center::array(
                position, orientation,
                radius, length
            );
            cylinder.name = obj["name"];
            capsules.push_back(cylinder);
        }
    }

    // Fill boxes
    for (const auto& obj : problem["box"]) {
        const json& position = obj["position"];
        const json& orientation = obj["orientation_euler_xyz"];
        const json& half_extents = obj["half_extents"];
        auto cuboid = factory::cuboid::array(
            position, orientation, half_extents
        );
        cuboid.name = obj["name"];
        cuboids.push_back(cuboid);
    }

    // Allocate memory on the heap for the arrays
    if (!spheres.empty()) {
        env.spheres = new Sphere<float>[spheres.size()];
        std::copy(spheres.begin(), spheres.end(), env.spheres);
        env.num_spheres = spheres.size();
    }

    if (!capsules.empty()) {
        env.capsules = new Capsule<float>[capsules.size()];
        std::copy(capsules.begin(), capsules.end(), env.capsules);
        env.num_capsules = capsules.size();
    }

    if (!cuboids.empty()) {
        env.cuboids = new Cuboid<float>[cuboids.size()];
        std::copy(cuboids.begin(), cuboids.end(), env.cuboids);
        env.num_cuboids = cuboids.size();
    }

    return env;
}


template <typename Robot>
void run_planner(json &data, Environment<float> &env, struct pRRTC_settings &settings) {
    using Configuration = typename Robot::Configuration;
    Configuration start = data["start"];
    std::vector<Configuration> goals = data["goals"];
    auto result = pRRTC::solve<Robot>(start, goals, env, settings);
    for (auto& cfg: result.path) {
        print_cfg<Robot>(cfg);
    }
    if (not result.solved) {
        std::cout << "failed!" << std::endl;
    }
    std::cout << "cost: " << result.cost << "\n";
    std::cout << "time (us): " << result.kernel_ns/1000.0f << "\n";
}

int main(int argc, char* argv[]) {
    std::string robot_name = "panda";
    std::string name = "cage";
    int problem_idx = 1;
    if (argc == 4) {
        robot_name = argv[1];
        name = argv[2];
        problem_idx = std::stoi(argv[3]);
    }
    else {
        std::cout << "Usage: ./single_mbm <robot_name> <problem_name> <problem_idx>\n";
        return 1;
    }
    std::string path = "scripts/" + robot_name + "_problems.json";
    std::ifstream f(path);
    json all_data = json::parse(f);
    json problems = all_data["problems"];
    auto pset = problems[name];
    json data = pset[problem_idx - 1];
    if (not data["valid"]) {
        return -1;
    }
    auto env = problem_dict_to_env(data, name);
    struct pRRTC_settings settings;
    settings.num_new_configs = 512; //usually:512
    settings.max_iters = 100000000;
    settings.granularity = 16;
    settings.range = 0.5;
    settings.balance = 2;
    settings.tree_ratio = 1.0;
    settings.dynamic_domain = true;
    settings.dd_radius = 4.0;
    settings.dd_min_radius = 1.0;
    settings.dd_alpha = 0.0001;
    if (robot_name == "fetch") {
        run_planner<robots::Fetch>(data, env, settings);
    } else if (robot_name == "panda") {
        run_planner<robots::Panda>(data, env, settings);
    } else if (robot_name == "baxter") {
        run_planner<robots::Baxter>(data, env, settings);
    } else {
        std::cerr << "Unsupported robot type: " << robot_name << "\n";
        return 1;
    }
}