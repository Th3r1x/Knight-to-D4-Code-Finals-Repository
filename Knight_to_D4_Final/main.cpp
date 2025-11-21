#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Include the nlohmann JSON library
#include "json.hpp"
// Include your solver's header
#include "solver.h"

using json = nlohmann::json;

// --- JSON Serializers ---

// For VehicleState (Output)
void to_json(json& j, const VehicleState& v) {
    j = json{
        {"id", v.id},
        {"capacity", v.capacity},
        {"max_capacity", v.max_capacity},
        {"location", v.location},
        {"destination", v.destination},
        {"departure_time", v.departure_time}, // Added for interpolation
        {"arrival_time", v.arrival_time},
        {"is_moving", v.is_moving},
        {"travel_progress", 0.0}, 
        {"edge", v.edge.has_value() ? json{v.edge.value().first, v.edge.value().second} : nullptr}
    };
}

// For DynamicState (History Step Output)
void to_json(json& j, const DynamicState& s) {
    json demands_obj = json::object();
    for (const auto& [node_id, demand_value] : s.current_demands) {
        demands_obj[std::to_string(node_id)] = demand_value;
    }

    j = json{
        {"time", s.time},
        {"total_distance", s.total_distance},
        {"action_info", s.action_info},
        {"vehicles", s.vehicles},
        {"demands", demands_obj}
    };
}

// --- Heuristic Selection ---
HeuristicFuncType get_heuristic_by_name(const std::string& name) {
    if (name == "TBIH-1 (Random)") return TBIH_1;
    if (name == "TBIH-2 (DSIH)") return TBIH_2;
    if (name == "TBIH-3 (DCW)") return TBIH_3;
    if (name == "TBIH-4 (DLA-SIH)") return TBIH_4;
    if (name == "TBIH-5 (DLA-CW)") return TBIH_5;
    return TBIH_1;
}

int main() {
    try {
        // 1. Read Input
        std::string input_string;
        std::string line;
        while (std::getline(std::cin, line)) input_string += line;
        
        if (input_string.empty()) return 0;

        json input_data = json::parse(input_string);

        // 2. Parse Data
        std::map<int, NodeData> nodes_map;
        for (const auto& node : input_data["nodes"]) {
            nodes_map[node["id"]] = NodeData{
                node.value("name", "Node " + std::to_string((int)node["id"])), // Default to ID if name missing
                node["type"], 
                node["lat"], 
                node["lon"], 
                node["demand"]
            };
        }

        std::map<std::pair<int, int>, EdgeData> edges_map;
        for (const auto& edge : input_data["edges"]) {
            edges_map[_get_edge_key(edge["u"], edge["v"])] = EdgeData{
                edge["weight"], edge["time"], 
                (double)edge["max_capacity"], // Init capacity to max
                (double)edge["damage"]
            };
        }

        std::vector<std::pair<double, int>> vehicle_configs;
        for (const auto& v : input_data["vehicles"]) {
            vehicle_configs.push_back({v["capacity"], v["start"]});
        }

        // 3. Setup
        std::string h_name = input_data["heuristic"];
        int n_sim = input_data["num_simulations"];
        int l_hor = input_data["lookahead_horizon"];
        int max_steps = input_data.value("max_steps", 1000);

        MDDVRPSRC_Environment env(nodes_map, edges_map, vehicle_configs);
        PDSRASolver solver(get_heuristic_by_name(h_name), n_sim, l_hor);

        // 4. Simulation Loop
        for (int k = 0; k < max_steps; ++k) {
            if (env.all_demands_met()) {
                // One last step to record final state
                env.step({}); 
                break;
            }

            if (k % 25 == 0) {
                const auto& state = env.get_state();
                std::cerr << "Step " << k 
                          << " | Time: " << std::fixed << std::setprecision(1) << state.time 
                          << " | Pending Demand: " << state.total_pending_demand 
                          << std::endl;
            }
            
            auto actions = solver.decide_actions(env);
            env.step(actions);
        }

        // 5. Output
        json output = env.get_history();
        std::cout << output.dump() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CPP Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}