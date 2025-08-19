#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/planning/Planners.hh"
#include "src/planning/pRRTC_settings.hh"

#include <vamp/planning/validate.hh>
#include <vamp/collision/factory.hh>
#include <vamp/collision/environment.hh>
#include <vamp/robots/fetch.hh> 
#include <vamp/robots/panda.hh>
#include <vamp/robots/baxter.hh>



using json = nlohmann::json;

using namespace ppln::collision;

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

void print_csv_header(std::ofstream &outfile) {
    outfile << "problem_name,problem_idx,solved,vamp_valid,cost,path_length,start_tree_size,goal_tree_size,iters,wall_ns,kernel_ns,";
    outfile << "copy_ns,num_new_configs,granularity,range,balance,tree_ratio,dynamic_domain,dd_alpha,dd_radius,dd_min_radius,configs_str\n";
}
template<typename Robot>
void print_planner_result_to_file(PlannerResult<Robot> &result, pRRTC_settings &settings, std::string problem_name, int problem_idx, std::ofstream &outfile, bool vamp_valid, std::string configs_str) {
    outfile << problem_name << ", ";
    outfile << problem_idx << ", ";
    outfile << result.solved << ", ";
    outfile << vamp_valid << ", ";
    outfile << result.cost << ", ";
    outfile << result.path_length << ", ";
    outfile << result.start_tree_size << ", ";
    outfile << result.goal_tree_size << ", ";
    outfile << result.iters << ", ";
    outfile << result.wall_ns << ", ";
    outfile << result.kernel_ns << ", ";
    outfile << result.copy_ns << ", ";
    outfile << settings.num_new_configs << ", ";
    outfile << settings.granularity << ", ";
    outfile << settings.range << ", ";
    outfile << settings.balance << ", ";
    outfile << settings.tree_ratio << ", ";
    outfile << settings.dynamic_domain << ", ";
    outfile << settings.dd_alpha << ", ";
    outfile << settings.dd_radius << ", ";
    outfile << settings.dd_min_radius << ", ";
    outfile << "\"" << configs_str << "\"";
    outfile << "\n";
}


template <typename Robot, typename vampRobot>
void run_planning(const json &problems, pRRTC_settings &settings, std::string run_name, std::string robot_name) {
    using Configuration = typename Robot::Configuration;
    std::ofstream outfile("test_output/"+robot_name+"_"+run_name+".csv");
    print_csv_header(outfile);

    int failed = 0;
    std::map<std::string, std::vector<PlannerResult<Robot>>> results;
    // std::vector<std::string> prob_names = {
    //     "bookshelf_small",
    //     "bookshelf_tall",
    //     "bookshelf_thin",
    //     "box",
    //     "cage",
    //     "table_pick",
    //     "table_under_pick",
    // };
    
    for (auto& [name, pset] : problems.items()) {
        std::cout << name << "\n";
        // auto pset = problems[name];
        
        for (int i = 0; i < pset.size(); i++) {
            std::cout << "idx: " << i << "\n";
            json data = pset[i];
            if (not data["valid"]) {
                continue;
            }
            auto env = problem_dict_to_env(data, name);
            auto vamp_env = problem_dict_vamp(data, name);
            printf("num spheres, capsules, cuboids: %d, %d, %d\n", env.num_spheres, env.num_capsules, env.num_cuboids);
            printf("num vamp spheres, capsules, cuboids: %d, %d, %d\n", vamp_env.spheres.size(), vamp_env.capsules.size(), vamp_env.cuboids.size());
            Configuration start = data["start"];
            std::vector<Configuration> goals = data["goals"];
            auto result = pRRTC::solve<Robot>(start, goals, env, settings);
            for (auto& cfg: result.path) {
                print_cfg<Robot>(cfg);
            }
            std::stringstream configs_str;
            for (auto& cfg: result.path) {
                print_cfg_to_ss<Robot>(cfg, configs_str);
            }
            std::cout << "kernel_ns: " << result.kernel_ns << "\n";
            if (not result.solved) {
                failed ++;
                std::cout << "failed " << name << std::endl;
            }
            std::cout << "cost: " << result.cost << "\n";
            results[name].emplace_back(result);
            
            // Validate the result
            bool vamp_valid = true;
            for (auto i = 1ul; i < result.path.size(); i++) {
                auto cfg1 = result.path[i-1];
                auto cfg2 = result.path[i];
                typename vampRobot::Configuration vamp_cfg1(cfg1);
                typename vampRobot::Configuration vamp_cfg2(cfg2);
                if (not vamp::planning::validate_motion<vampRobot, rake, 1>(vamp_cfg1, vamp_cfg2, vamp_env)) {
                    int index1 = result.path.size() - i - 1;
                    int index2 = result.path.size() - (i-1) - 1;
                    std::cout << "Vamp found collision in solution path between " << index1 << " and " << index2 << std::endl;
                    vamp_valid = false;
                    break;
                }
            }


            print_planner_result_to_file(result, settings, name, i+1, outfile, vamp_valid, configs_str.str());
        }
    }
}

int main(int argc, char* argv[]) {
    std::string robot_name = "panda";
    std::string run_name;
    pRRTC_settings settings;
    settings.num_new_configs = 512; // usually 512
    settings.granularity = 32;
    settings.range = 0.5;
    settings.balance = 2;
    settings.tree_ratio = 1.0;
    settings.dynamic_domain = true;
    settings.dd_radius = 4.0;
    settings.dd_min_radius = 1.0;
    settings.dd_alpha = 0.0001;
    

    if (argc == 3) {
        robot_name = argv[1];
        run_name = argv[2];
    }
    else {
        std::cout << "Usage: evaluate_mbm <robot_name> <run_name>\n";
        return -1;
    }

    std::string path = "scripts/" + robot_name + "_problems.json";
    std::ifstream f(path);
    json all_data = json::parse(f);
    json problems = all_data["problems"];
    if (robot_name == "fetch") {
        // settings.granularity = 32;
        run_planning<robots::Fetch, vamp::robots::Fetch>(problems, settings, run_name, robot_name);
    } else if (robot_name == "panda") {
        // settings.granularity = 32;
        run_planning<robots::Panda, vamp::robots::Panda>(problems, settings, run_name, robot_name);
    } else if (robot_name == "baxter") {
        // settings.granularity = 64;
        run_planning<robots::Baxter, vamp::robots::Baxter>(problems, settings, run_name, robot_name);
    } else {
        std::cerr << "Unsupported robot type: " << robot_name << "\n";
        return 1;
    }
}