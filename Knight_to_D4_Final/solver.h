#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#pragma once

// OpenMP for parallelization
#ifdef _OPENMP
#include <omp.h>
#endif

// A large constant used as a reward or penalty (Equation 17).
const long long G2 = 1000000000;

// --- Data Structures ---

struct NodeData {
    std::string name;    
    std::string type;    // 'depot', 'shelter', or 'connecting'
    double latitude;
    double longitude;
    double demand = 0.0; // Initial demand
};

struct EdgeData {
    double weight = 0.0; // distance
    double time = 0.0;   // base travel time
    double max_capacity = 0.0;
    double damage = 0.0; // Initial damage
    
    // OPTIMIZATION: Memoized cost (weight + damage penalty)
    // Calculated once during init to avoid repeated math in hot loops
    double precomputed_cost = 0.0; 
};

// Dynamic properties of an edge that change during simulation
struct DynamicEdgeState {
    double current_capacity;
    std::vector<int> occupants; // Vehicle IDs currently on this edge
};

// Helper to create a consistent, sorted key for undirected edges.
inline std::pair<int, int> _get_edge_key(int u, int v) {
    return {std::min(u, v), std::max(u, v)};
}

// --- State Structures ---

struct StaticContext {
    std::map<int, NodeData> nodes;
    std::map<std::pair<int, int>, EdgeData> edges;
    std::map<int, std::set<int>> adj; // Adjacency list
    std::set<int> depots;
};

struct VehicleState {
    int id;
    double capacity;
    double max_capacity;
    int location;
    int destination;
    
    double departure_time;
    double arrival_time;
    bool is_moving;
    
    std::optional<std::pair<int, int>> edge;
};

struct DynamicState {
    double time;
    double total_distance;
    
    // Optimization: Track total remaining demand to avoid O(N) loops
    double total_pending_demand = 0.0; 

    std::vector<VehicleState> vehicles;
    std::map<int, double> current_demands;
    
    std::map<std::pair<int, int>, DynamicEdgeState> edge_states;
    
    std::string action_info;
};

using HeuristicFuncType = std::function<std::map<int, int>(const StaticContext&, const DynamicState&, std::mt19937&)>;

/*
 * Manages the simulation.
 */
class MDDVRPSRC_Environment {
private:
    std::shared_ptr<StaticContext> static_ctx;
    DynamicState current_state;
    
    // History is only needed for the main simulation, not lookaheads.
    std::vector<DynamicState> history;

    std::mt19937 gen;

    static std::map<int, std::set<int>> build_adj(const std::map<std::pair<int, int>, EdgeData>& edges);
    double _calculate_t_wait();
    void assign_destination(int vehicle_idx, int destination);
    void update_stochastic_road_capacities();

    friend class PDSRASolver;

public:
    MDDVRPSRC_Environment(
        const std::map<int, NodeData>& nodes_map, 
        const std::map<std::pair<int, int>, EdgeData>& edges_map,
        const std::vector<std::pair<double, int>>& vehicle_configs
    );

    // OPTIMIZED: Copy constructor now skips copying history
    MDDVRPSRC_Environment(const MDDVRPSRC_Environment& other);

    const StaticContext& get_static_context() const { return *static_ctx; }
    const DynamicState& get_state() const { return current_state; }
    const std::vector<DynamicState>& get_history() const { return history; }

    void step(std::map<int, int> actions);
    bool all_demands_met() const;
    void record_state(const std::string& action_info);
    void advance_time_to_next_arrival();
};

// --- Solver ---

class PDSRASolver {
private:
    HeuristicFuncType heuristic_func;
    int num_simulations;
    int lookahead_horizon;
    std::mt19937 gen;

    // Helper to get a reward without re-calculating demand sums
    double _calculate_reward(const VehicleState& v, int destination, const StaticContext& s_ctx, const DynamicState& d_state) const;
    
    // Thread-safe simulation run
    double _run_pds_ra_simulation(MDDVRPSRC_Environment pds_env, std::mt19937& thread_gen);

public:
    PDSRASolver(HeuristicFuncType func, int num_sim, int lookahead);
    std::map<int, int> decide_actions(MDDVRPSRC_Environment& current_env);
};

// --- Heuristics Declarations ---
std::map<int, int> TBIH_1(const StaticContext& s, const DynamicState& d, std::mt19937& gen);
std::map<int, int> TBIH_2(const StaticContext& s, const DynamicState& d, std::mt19937& gen);
std::map<int, int> TBIH_3(const StaticContext& s, const DynamicState& d, std::mt19937& gen);
std::map<int, int> TBIH_4(const StaticContext& s, const DynamicState& d, std::mt19937& gen);
std::map<int, int> TBIH_5(const StaticContext& s, const DynamicState& d, std::mt19937& gen);