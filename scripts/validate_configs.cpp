#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <vamp/planning/validate.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/factory.hh>
#include <vamp/robots/baxter.hh>
#include <vamp/robots/fetch.hh>
#include <vamp/robots/panda.hh>

using json = nlohmann::json;

static constexpr const std::size_t rake = vamp::FloatVectorWidth;

vamp::collision::Environment<vamp::FloatVector<rake>> problem_dict_vamp(const json& problem, const std::string &name) {
    auto env = vamp::collision::Environment<float>();
    
    // Handle spheres
    for (const auto& obj : problem["sphere"]) {
        const json& position = obj["position"];
        auto sphere = vamp::collision::factory::sphere::array(
            {position[0], position[1], position[2]},
            obj["radius"]
        );
        env.spheres.emplace_back(sphere);
    }

    // Handle cylinders
    if (name == "box") {
        for (const auto& obj : problem["cylinder"]) {
            const json& position = obj["position"];
            const json& orientation = obj["orientation_euler_xyz"];
            const float radius = obj["radius"];
            const std::array<float, 3> dims = {radius, radius, radius/2.0f};
            auto cuboid = vamp::collision::factory::cuboid::array(
                {position[0], position[1], position[2]},
                {orientation[0], orientation[1], orientation[2]},
                dims
            );
            env.cuboids.emplace_back(cuboid);
        }
    } else {
        for (const auto& obj : problem["cylinder"]) {
            const json& position = obj["position"];
            const json& orientation = obj["orientation_euler_xyz"];
            const float radius = obj["radius"];
            const float length = obj["length"];
            auto capsule = vamp::collision::factory::capsule::center::array(
                {position[0], position[1], position[2]},
                {orientation[0], orientation[1], orientation[2]},
                radius, length
            );
            env.capsules.emplace_back(capsule);
        }
    }

    // Handle boxes
    for (const auto& obj : problem["box"]) {
        const json& position = obj["position"];
        const json& orientation = obj["orientation_euler_xyz"];
        const json& half_extents = obj["half_extents"];
        auto cuboid = vamp::collision::factory::cuboid::array(
            {position[0], position[1], position[2]},
            {orientation[0], orientation[1], orientation[2]},
            {half_extents[0], half_extents[1], half_extents[2]}
        );
        env.cuboids.emplace_back(cuboid);
    }
    env.sort();
    auto env_v = vamp::collision::Environment<vamp::FloatVector<rake>>(env);
    return env_v;
}

template <typename VampRobot>
static auto load_path_and_validate(const std::string &config_path,
                                   const vamp::collision::Environment<vamp::FloatVector<rake>> &env_v) -> int
{
    using Configuration = typename VampRobot::Configuration;
    static constexpr std::size_t dimension = VampRobot::dimension;

    std::ifstream in(config_path);
    if (!in.is_open()) {
        std::cerr << "Failed to open config file: " << config_path << "\n";
        return 2;
    }

    std::vector<std::array<float, dimension>> configs;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        std::array<float, dimension> cfg{};
        for (std::size_t i = 0; i < dimension; ++i) {
            if (!(iss >> cfg[i])) {
                std::cerr << "Line " << line_number
                          << ": expected " << dimension << " floats, but could not parse value "
                          << (i + 1) << "\n";
                return 3;
            }
        }
        // Allow trailing whitespace or values; warn if extra tokens
        float extra;
        if (iss >> extra) {
            std::cerr << "Warning: line " << line_number << " has more than " << dimension
                      << " values; extra values will be ignored.\n";
        }
        configs.emplace_back(cfg);
    }

    if (configs.size() < 2) {
        std::cerr << "Need at least 2 configurations to validate motion.\n";
        return 4;
    }

    // Check state validity for each configuration (state-only)
    bool all_states_valid = true;
    // for (std::size_t i = 0; i < configs.size(); ++i) {
    //     Configuration c(configs[i]);
    //     std::cout << "Configuration " << c << std::endl;
    //     if (!vamp::planning::validate_motion<VampRobot, rake, 1>(c, c, env_v)) {
    //         std::cout << "Collision at configuration index " << i << " (state invalid).\n";
    //         all_states_valid = false;
    //     }
    // }

    // Check motion validity between consecutive configurations
    bool all_motions_valid = true;
    for (std::size_t i = 1; i < configs.size(); ++i) {
        Configuration c1(configs[i - 1]);
        Configuration c2(configs[i]);
        if (!vamp::planning::validate_motion<VampRobot, rake, VampRobot::resolution>(c1, c2, env_v)) {
            std::cout << "Collision found between indices " << (i - 1) << " and " << i << "\n";
            all_motions_valid = false;
        }
    }

    if (all_states_valid && all_motions_valid) {
        std::cout << "No collisions detected.\n";
        return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    // Usage:
    //   validate_configs <config_file> <robot:{baxter|panda|fetch}> <problem_name> <problem_index>
    if (argc != 5) {
        std::cout << "Usage: validate_configs <config_file> <robot:{baxter|panda|fetch}> <problem_name> <problem_index>\n";
        return -1;
    }

    const std::string config_path = argv[1];
    const std::string robot = argv[2];
    const std::string problem_name = argv[3];
    const int problem_index = std::stoi(argv[4]);

    // Load environment from scripts/<robot>_problems.json
    const std::string problems_json = std::string("scripts/") + robot + "_problems.json";
    std::ifstream f(problems_json);
    if (!f.is_open()) {
        std::cerr << "Failed to open problems JSON: " << problems_json << "\n";
        return 2;
    }
    json all_data = json::parse(f);
    if (!all_data.contains("problems") || !all_data["problems"].contains(problem_name)) {
        std::cerr << "Problem name not found in JSON: " << problem_name << "\n";
        return 2;
    }
    const auto &pset = all_data["problems"][problem_name];
    if (problem_index < 0 || problem_index >= static_cast<int>(pset.size())) {
        std::cerr << "Problem index out of range.\n";
        return 2;
    }
    auto env_v = problem_dict_vamp(pset[problem_index - 1], problem_name);

    if (robot == "baxter") {
        return load_path_and_validate<vamp::robots::Baxter>(config_path, env_v);
    } else if (robot == "panda") {
        return load_path_and_validate<vamp::robots::Panda>(config_path, env_v);
    } else if (robot == "fetch") {
        return load_path_and_validate<vamp::robots::Fetch>(config_path, env_v);
    } else {
        std::cerr << "Unsupported robot: " << robot << " (expected baxter|panda|fetch)\n";
        return -1;
    }
}


