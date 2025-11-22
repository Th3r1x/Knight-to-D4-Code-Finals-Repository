#include <algorithm>
#include <chrono> // For seeding
#include <cmath>
#include <queue>
#include <sstream>
#include "solver.h"

// --- Helpers ---

std::map<int, std::set<int>> MDDVRPSRC_Environment::build_adj(const std::map<std::pair<int, int>, EdgeData>& edges_map) {
    std::map<int, std::set<int>> adj_list;
    for (const auto& [edge_pair, data] : edges_map) {
        adj_list[edge_pair.first].insert(edge_pair.second);
        adj_list[edge_pair.second].insert(edge_pair.first);
    }
    return adj_list;
}

// Helper to get weight safely
double get_edge_weight(const StaticContext& s, int u, int v) {
    auto key = _get_edge_key(u, v);
    auto it = s.edges.find(key);
    return (it != s.edges.end()) ? it->second.weight : std::numeric_limits<double>::infinity();
}

// --- MDDVRPSRC_Environment ---

MDDVRPSRC_Environment::MDDVRPSRC_Environment(
    const std::map<int, NodeData>& nodes_map, 
    const std::map<std::pair<int, int>, EdgeData>& edges_map,
    const std::vector<std::pair<double, int>>& vehicle_configs) 
{
    std::random_device rd;
    gen.seed(rd());

    static_ctx = std::make_shared<StaticContext>();
    static_ctx->nodes = nodes_map;
    static_ctx->edges = edges_map;
    static_ctx->adj = build_adj(edges_map);
    
    // Precompute costs
    for (auto& [key, edge] : static_ctx->edges) {
        double time_penalty = edge.time * (1.0 + edge.damage / 10.0);
        edge.precomputed_cost = edge.weight + time_penalty;
    }
    
    for (const auto& [id, data] : nodes_map) {
        if (data.type == "depot") static_ctx->depots.insert(id);
    }

    if (static_ctx->depots.empty()) {
        throw std::runtime_error("Graph must contain at least one depot node.");
    }

    current_state.time = 0.0;
    current_state.total_distance = 0.0;
    current_state.total_pending_demand = 0.0;

    for (const auto& [edge_pair, data] : edges_map) {
        DynamicEdgeState es;
        es.current_capacity = (data.max_capacity > 0) ? data.max_capacity : 0.0; 
        current_state.edge_states[edge_pair] = es;
    }

    for (const auto& [id, data] : nodes_map) {
        if (data.type == "shelter") {
            current_state.current_demands[id] = data.demand;
            current_state.total_pending_demand += data.demand;
        }
    }

    for (size_t i = 0; i < vehicle_configs.size(); i++) {
        VehicleState v;
        v.id = static_cast<int>(i);
        v.max_capacity = vehicle_configs[i].first;
        v.location = vehicle_configs[i].second;

        if (nodes_map.find(v.location) == nodes_map.end()) {
            throw std::runtime_error("Vehicle start location does not exist.");
        }

        v.destination = v.location;
        v.capacity = (nodes_map.at(v.location).type == "depot") ? v.max_capacity : 0.0;
        v.departure_time = 0.0;
        v.arrival_time = 0.0;
        v.is_moving = false;
        v.edge = std::nullopt;
        current_state.vehicles.push_back(v);
    }

    record_state("Initial State");
}

MDDVRPSRC_Environment::MDDVRPSRC_Environment(const MDDVRPSRC_Environment& other)
    : static_ctx(other.static_ctx), 
      current_state(other.current_state), 
      gen(other.gen) 
{
    // History intentionally not copied for performance
}

void MDDVRPSRC_Environment::update_stochastic_road_capacities() {
    for (auto& [edge_key, dyn_state] : current_state.edge_states) {
        auto it = static_ctx->edges.find(edge_key);
        if (it == static_ctx->edges.end()) continue;

        const auto& static_data = it->second;
        
        double mean_cap = std::max(0.1, static_data.max_capacity * std::exp(-0.0001 * static_data.damage * current_state.time));
        std::poisson_distribution<int> p_dist(mean_cap);
        int hat_r = p_dist(gen); 

        int occupants = (int)dyn_state.occupants.size();
        double avail = std::max(std::min((double)hat_r, static_data.max_capacity) - occupants, 0.0);
        
        dyn_state.current_capacity = avail;
    }
}

void MDDVRPSRC_Environment::step(std::map<int, int> actions) {
    std::vector<int> ready_vehicle_indices;
    ready_vehicle_indices.reserve(current_state.vehicles.size());
    
    for (size_t i=0; i<current_state.vehicles.size(); ++i) {
        if (!current_state.vehicles[i].is_moving) {
            ready_vehicle_indices.push_back(i);
        }
    }

    for (int idx : ready_vehicle_indices) {
        VehicleState& v = current_state.vehicles[idx];
        int dest = (actions.count(v.id) ? actions.at(v.id) : v.location);
        
        if (dest == v.location) {
            v.arrival_time = _calculate_t_wait();
            v.is_moving = true; 
            v.destination = v.location;
        } else {
            assign_destination(idx, dest);
        }
    }

    if (!actions.empty()) {
        record_state("Actions Applied");
    }

    advance_time_to_next_arrival();

    bool any_arrival = false;
    for (auto& v : current_state.vehicles) {
        if (v.is_moving && current_state.time >= v.arrival_time) {
            v.location = v.destination;
            v.is_moving = false;
            any_arrival = true;

            if (v.edge.has_value()) {
                auto& occ = current_state.edge_states[v.edge.value()].occupants;
                auto it = std::remove(occ.begin(), occ.end(), v.id);
                if (it != occ.end()) occ.erase(it, occ.end());
                v.edge = std::nullopt;
            }

            if (static_ctx->nodes.find(v.location) != static_ctx->nodes.end()) {
                std::string type = static_ctx->nodes.at(v.location).type;
                if (type == "depot") {
                    v.capacity = v.max_capacity;
                } else if (type == "shelter") {
                    double& dem = current_state.current_demands[v.location];
                    if (dem > 0) {
                        double delivered = std::min(v.capacity, dem);
                        v.capacity -= delivered;
                        dem -= delivered;
                        current_state.total_pending_demand -= delivered;
                    }
                }
            }
        }
    }

    if (any_arrival || !actions.empty()) {
        update_stochastic_road_capacities();
    }

    record_state("Step Complete");
}

double MDDVRPSRC_Environment::_calculate_t_wait() {
    double next_arrival = std::numeric_limits<double>::infinity();
    bool has_moving = false;
    
    for(const auto& v : current_state.vehicles) {
        if(v.is_moving && v.arrival_time > current_state.time) {
            has_moving = true;
            if(v.arrival_time < next_arrival) next_arrival = v.arrival_time;
        }
    }
    
    if (has_moving) return next_arrival;
    
    double min_t = std::numeric_limits<double>::infinity();
    for(auto& [k, e] : static_ctx->edges) {
        if (e.time < min_t) min_t = e.time;
    }
    
    if (std::isinf(min_t)) min_t = 1.0;
    
    return current_state.time + min_t;
}

void MDDVRPSRC_Environment::assign_destination(int vehicle_idx, int destination) {
    VehicleState& v = current_state.vehicles[vehicle_idx];
    if (v.location == destination) return;

    v.departure_time = current_state.time;
    std::pair<int, int> edge_key = _get_edge_key(v.location, destination);
    
    auto it = static_ctx->edges.find(edge_key);
    if (it == static_ctx->edges.end()) return; 

    const auto& edge = it->second;
    double penalty = edge.time * (1.0 + edge.damage / 10.0);
    
    v.arrival_time = current_state.time + edge.time + penalty;
    v.destination = destination;
    v.is_moving = true;
    current_state.total_distance += edge.weight;

    v.edge = std::make_pair(v.location, destination);
    
    auto& occ = current_state.edge_states[edge_key].occupants;
    bool exists = false;
    for(int id : occ) if(id == v.id) { exists = true; break; }
    if(!exists) occ.push_back(v.id);
}

void MDDVRPSRC_Environment::advance_time_to_next_arrival() {
    double next_t = std::numeric_limits<double>::infinity();
    bool moving = false;
    for (const auto& v : current_state.vehicles) {
        if (v.is_moving) {
            moving = true;
            if (v.arrival_time < next_t) next_t = v.arrival_time;
        }
    }
    if (moving && next_t > current_state.time && !std::isinf(next_t)) {
        current_state.time = next_t;
    }
}

bool MDDVRPSRC_Environment::all_demands_met() const {
    return current_state.total_pending_demand <= 1e-9;
}

void MDDVRPSRC_Environment::record_state(const std::string& info) {
    current_state.action_info = info;
    history.push_back(current_state);
}

// --- Heuristics & Solver ---

std::vector<int> get_valid_moves(int loc, const StaticContext& s, const DynamicState& d) {
    std::vector<int> moves;
    auto it = s.adj.find(loc);
    if (it == s.adj.end()) return moves;
    
    moves.reserve(it->second.size());
    
    for (int n : it->second) {
        std::pair<int, int> k = _get_edge_key(loc, n);
        auto e_it = d.edge_states.find(k);
        if (e_it != d.edge_states.end() && e_it->second.current_capacity > 0) {
            moves.push_back(n);
        }
    }
    return moves;
}

int random_choice(const std::vector<int>& v, std::mt19937& g) {
    if (v.empty()) return -1;
    std::uniform_int_distribution<> dis(0, (int)v.size()-1);
    return v[dis(g)];
}

// --- TBIH Logic ---

std::map<int, int> TBIH_base(
    const StaticContext& s, 
    const DynamicState& d, 
    std::function<int(const VehicleState&, const std::vector<int>&, const StaticContext&, const DynamicState&, std::mt19937&)> sp,
    std::mt19937& gen) 
{
    std::map<int, int> decisions;
    double total_demand = d.total_pending_demand;

    for(const auto& v : d.vehicles) {
        if (v.is_moving) {
            decisions[v.id] = v.destination;
            continue;
        }

        std::vector<int> valid = get_valid_moves(v.location, s, d);
        if (valid.empty()) {
            decisions[v.id] = v.location;
            continue;
        }

        // Categorize Moves
        std::vector<int> depots, shelters, non_obj; 
        depots.reserve(valid.size());
        shelters.reserve(valid.size());
        non_obj.reserve(valid.size());

        for(int n : valid) {
            bool is_depot = s.depots.count(n);
            bool is_shelter = false;
            
            if (!is_depot) {
                auto d_it = d.current_demands.find(n);
                if(d_it != d.current_demands.end() && d_it->second > 0 && s.nodes.at(n).type == "shelter") {
                    is_shelter = true;
                }
            }

            if (is_depot) depots.push_back(n);
            else if (is_shelter) shelters.push_back(n);
            else non_obj.push_back(n);
        }

        // Obvious Decisions (Teaching Part)
        int decision = -1;
        if (v.capacity == 0 && total_demand > 0 && !depots.empty()) decision = random_choice(depots, gen);
        else if (v.capacity > 0 && !shelters.empty()) decision = random_choice(shelters, gen);
        else if (total_demand <= 1e-9 && !s.depots.count(v.location) && !depots.empty()) decision = random_choice(depots, gen);
        else if (total_demand <= 1e-9 && s.depots.count(v.location)) decision = v.location;
        
        // Seeking Part (Heuristic)
        if (decision == -1) {
            // If no non-objective nodes available, use all valid moves
            const std::vector<int>& candidates = non_obj.empty() ? valid : non_obj;
            decision = sp(v, candidates, s, d, gen);
        }
        decisions[v.id] = decision;
    }
    return decisions;
}

// --- Heuristic Implementations ---

// TBIH-1: Random
int _sp_random(const VehicleState& v, const std::vector<int>& moves, const StaticContext&, const DynamicState&, std::mt19937& gen) {
    return !moves.empty() ? random_choice(moves, gen) : v.location;
}

// TBIH-2: Dynamic Sequential Insertion Heuristic (DSIH)
// Algorithm 5 in Paper
int _sp_dsih(const VehicleState& v, const std::vector<int>& moves, const StaticContext& s, const DynamicState& d, std::mt19937& gen) {
    if (moves.empty()) return v.location;
    if (moves.size() == 1) return moves[0];

    // 1. Identify potential seeds (neighbors of moves)
    // Note: Paper says "neighbors of potential destinations v". 
    // We consider 'j' as a seed if it's a neighbor of 'move'.
    std::vector<int> potential_seeds;
    for (int move : moves) {
        if (s.adj.count(move)) {
            for (int neighbor : s.adj.at(move)) {
                // Don't go back to current location immediately ideally, but allow it as seed
                if (neighbor != v.location) {
                    potential_seeds.push_back(neighbor);
                }
            }
        }
    }

    // If no seeds (disconnected?), fallback to random
    if (potential_seeds.empty()) return random_choice(moves, gen);

    // 2. Select random seed
    int seed = random_choice(potential_seeds, gen);

    // 3. Evaluate insertion of 'move' between 'current' and 'seed'
    // Cost function: c1 = c(curr, move) + c(move, seed) - c(curr, seed)
    // Maximize (lambda * c(curr, seed) - c1). Equivalent to Minimizing c1.
    
    int best_move = moves[0];
    double min_c1 = std::numeric_limits<double>::infinity();

    for (int move : moves) {
        // Check if edge (move, seed) exists
        if (s.edges.count(_get_edge_key(move, seed))) {
            double c_curr_move = get_edge_weight(s, v.location, move);
            double c_move_seed = get_edge_weight(s, move, seed);
            double c_curr_seed = get_edge_weight(s, v.location, seed); // Might be inf if no direct edge

            // If direct edge doesn't exist, c_curr_seed is inf, making c1 -inf (if we handle inf math), 
            // but physically we just want to minimize detour.
            // Generalized insertion cost usually: d(i,u) + d(u,j) - d(i,j)
            // We just minimize d(current, move) + d(move, seed).
            
            double c1 = c_curr_move + c_move_seed;
            if (!std::isinf(c_curr_seed)) c1 -= c_curr_seed;

            if (c1 < min_c1) {
                min_c1 = c1;
                best_move = move;
            }
        }
    }
    
    return best_move;
}

// TBIH-3: Dynamic Clarke and Wright (DCW)
// Algorithm 6 in Paper
int _sp_dcw(const VehicleState& v, const std::vector<int>& moves, const StaticContext& s, const DynamicState& d, std::mt19937& gen) {
    if (moves.empty()) return v.location;
    if (moves.size() == 1) return moves[0];

    // 1. Find valid edges between any two candidate moves
    struct PairSavings {
        int a;
        int b;
        double savings;
    };
    std::vector<PairSavings> valid_pairs;

    for (size_t i = 0; i < moves.size(); ++i) {
        for (size_t j = i + 1; j < moves.size(); ++j) {
            int m1 = moves[i];
            int m2 = moves[j];
            
            if (s.edges.count(_get_edge_key(m1, m2))) {
                // Savings = c(curr, m1) + c(curr, m2) - c(m1, m2)
                double c_c_1 = get_edge_weight(s, v.location, m1);
                double c_c_2 = get_edge_weight(s, v.location, m2);
                double c_1_2 = get_edge_weight(s, m1, m2);
                
                double savings = c_c_1 + c_c_2 - c_1_2;
                valid_pairs.push_back({m1, m2, savings});
            }
        }
    }

    // 2. If no pairs, random
    if (valid_pairs.empty()) return random_choice(moves, gen);

    // 3. Sort decreasing savings
    std::sort(valid_pairs.begin(), valid_pairs.end(), [](const PairSavings& a, const PairSavings& b){
        return a.savings > b.savings;
    });

    // 4. Select best pair. 
    // In a dynamic step, we can only move to one.
    // The heuristic implies starting a route curr -> a -> b. So we move to 'a' (or 'b').
    // To break ties or direction, we pick the one closer to current? Or just the first in pair.
    // Let's pick 'a' from the best pair.
    return valid_pairs[0].a;
}

// TBIH-4: Dynamic Lookahead SIH (DLASIH)
// Algorithm 7 in Paper
// Modified to reach distant, unserviced nodes.
int _sp_dlasih(const VehicleState& v, const std::vector<int>& moves, const StaticContext& s, const DynamicState& d, std::mt19937& gen) {
    if (moves.empty()) return v.location;
    
    // 1. Determine Target Type
    // If capacity > 0, we want Shelters. If capacity == 0, we want Depots.
    bool seeking_shelter = (v.capacity > 0);
    
    // 2. Filter Seeds: Keep only seeds that are neighbors of the Target Type
    std::vector<int> filtered_seeds;
    
    // Pre-collect targets to check adjacency fast
    // Optimization: This loops over all nodes. In a large graph, might be slow? 
    // Better: Iterate over moves, check their neighbors.
    
    for (int move : moves) {
        if (!s.adj.count(move)) continue;
        for (int neighbor : s.adj.at(move)) {
            // Check if neighbor is of target type
            bool is_target = false;
            if (seeking_shelter) {
                // Check if neighbor is a shelter with demand
                // Note: d.current_demands only contains shelters with demand > 0 usually?
                // Or just check type.
                // Paper implies "Unserved Shelters" (US)
                auto dem_it = d.current_demands.find(neighbor);
                if (dem_it != d.current_demands.end() && dem_it->second > 0) {
                    is_target = true;
                }
            } else {
                // Check if neighbor is Depot
                if (s.depots.count(neighbor)) is_target = true;
            }

            if (is_target) {
                filtered_seeds.push_back(neighbor);
            }
        }
    }

    // 3. Fallback: If no targeted seeds found, behave like TBIH-2 (Standard DSIH)
    // Note: We recurse/call _sp_dsih logic here essentially
    if (filtered_seeds.empty()) {
        return _sp_dsih(v, moves, s, d, gen);
    }

    // 4. Select Random Seed from Filtered List
    int seed = random_choice(filtered_seeds, gen);

    // 5. Eval Insertion (Same as TBIH-2 but with specific seed)
    int best_move = moves[0];
    double min_biased = std::numeric_limits<double>::infinity();

    for (int move : moves) {
        if (s.edges.count(_get_edge_key(move, seed))) {
            double c_curr_move = get_edge_weight(s, v.location, move);
            double c_curr_seed = get_edge_weight(s, v.location, seed);
            double c_move_seed = get_edge_weight(s, move, seed);
            
            // Insertion Cost: c1 = c(curr, move) + c(move, seed) - c(curr, seed)
            double c1 = c_curr_move + c_move_seed;
            if (!std::isinf(c_curr_seed)) {
                c1 -= c_curr_seed;
            }

            double biased = c1;
            // If the current move is the first step towards a GOAL, reduce the perceived cost.
            if (seeking_shelter) {
                if (s.nodes.at(seed).type == "shelter") {
                    biased = 0.05; 
                }
            } else if (s.depots.count(seed)) {
                 // Bias slightly less if just going to depot, as it's not the final goal
                 biased = c1 * 0.5; 
            }

            if (biased < min_biased) {
                min_biased = biased;
                best_move = move;
            }
        }
    }

    return best_move;
}

// TBIH-5: Dynamic Lookahead CW (DLACW)
// Algorithm 8 in Paper
int _sp_dlacw(const VehicleState& v, const std::vector<int>& moves, const StaticContext& s, const DynamicState& d, std::mt19937& gen) {
    if (moves.empty()) return v.location;

    // 1. Filter Moves: Only keep moves that are directly connected to a Goal (Shelter/Depot)
    // Unlike TBIH-4 (which filters seeds), TBIH-5 filters the immediate candidates.
    
    bool seeking_shelter = (v.capacity > 0);
    std::vector<int> filtered_moves;

    for (int move : moves) {
        // Check if 'move' connects to ANY goal
        bool connects_to_goal = false;
        if (!s.adj.count(move)) continue;

        for (int neighbor : s.adj.at(move)) {
            if (seeking_shelter) {
                auto dem_it = d.current_demands.find(neighbor);
                if (dem_it != d.current_demands.end() && dem_it->second > 0) {
                    connects_to_goal = true; break;
                }
            } else {
                if (s.depots.count(neighbor)) {
                    connects_to_goal = true; break;
                }
            }
        }
        if (connects_to_goal) filtered_moves.push_back(move);
    }

    // 2. Fallback: If no moves lead to goal, use all moves (Standard DCW)
    const std::vector<int>& candidates = filtered_moves.empty() ? moves : filtered_moves;

    // 3. Run DCW logic on candidates
    return _sp_dcw(v, candidates, s, d, gen);
}

// Mapping
std::map<int, int> TBIH_1(const StaticContext& s, const DynamicState& d, std::mt19937& g) { return TBIH_base(s, d, _sp_random, g); }
std::map<int, int> TBIH_2(const StaticContext& s, const DynamicState& d, std::mt19937& g) { return TBIH_base(s, d, _sp_dsih, g); }
std::map<int, int> TBIH_3(const StaticContext& s, const DynamicState& d, std::mt19937& g) { return TBIH_base(s, d, _sp_dcw, g); }
std::map<int, int> TBIH_4(const StaticContext& s, const DynamicState& d, std::mt19937& g) { return TBIH_base(s, d, _sp_dlasih, g); }
std::map<int, int> TBIH_5(const StaticContext& s, const DynamicState& d, std::mt19937& g) { return TBIH_base(s, d, _sp_dlacw, g); }

PDSRASolver::PDSRASolver(HeuristicFuncType func, int num_sim, int lookahead)
    : heuristic_func(func), num_simulations(num_sim), lookahead_horizon(lookahead) {
    std::random_device rd;
    gen.seed(rd());
}

double PDSRASolver::_calculate_reward(const VehicleState& v, int dest, const StaticContext& s, const DynamicState& d) const {
    if (v.is_moving || v.location == dest) return 0.0;

    std::pair<int, int> key = _get_edge_key(v.location, dest);
    auto e_it = s.edges.find(key);
    if (e_it == s.edges.end()) return -G2;

    const auto& edge = e_it->second;
    double total_cost = edge.precomputed_cost;

    if (s.nodes.find(v.location) == s.nodes.end() || s.nodes.find(dest) == s.nodes.end()) {
        return -G2; 
    }

    if (s.nodes.at(v.location).type == "shelter") {
        auto d_it = d.current_demands.find(v.location);
        if (d_it != d.current_demands.end()) {
            double demand = d_it->second;
            if (demand > 0) {
                double match_reward = (v.capacity == demand) ? G2 : (G2 - std::abs(v.capacity - demand) * G2);
                return match_reward - total_cost;
            }
        }
    }
    
    double total_demand = d.total_pending_demand;

    if (s.nodes.at(dest).type == "depot" && (v.capacity == 0 || total_demand <= 1e-9)) {
        return G2 - total_cost;
    }

    return -total_cost;
}

double PDSRASolver::_run_pds_ra_simulation(MDDVRPSRC_Environment pds_env, std::mt19937& thread_gen) {
    double total_reward = 0;
    
    for (int h=0; h<lookahead_horizon; ++h) {
        if (pds_env.all_demands_met()) break;

        auto actions = heuristic_func(pds_env.get_static_context(), pds_env.get_state(), thread_gen);
        
        for (auto [vid, dest] : actions) {
            for (const auto& v : pds_env.get_state().vehicles) {
                if (v.id == vid) {
                    total_reward += _calculate_reward(v, dest, pds_env.get_static_context(), pds_env.get_state());
                    break;
                }
            }
        }
        
        pds_env.step(actions);
    }
    return total_reward;
}

std::map<int, int> PDSRASolver::decide_actions(MDDVRPSRC_Environment& env) {
    const auto& state = env.get_state();
    const auto& s_ctx = env.get_static_context();
    
    std::map<int, int> decisions;
    std::vector<const VehicleState*> ready_vehicles;
    for (const auto& v : state.vehicles) {
        if (!v.is_moving) ready_vehicles.push_back(&v);
    }

    for (const auto* v : ready_vehicles) {
        std::vector<int> valid = get_valid_moves(v->location, s_ctx, state);
        valid.push_back(v->location); 

        struct MoveResult {
            int move;
            double score;
        };
        
        std::vector<MoveResult> results(valid.size());

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)valid.size(); ++i) {
            int move = valid[i];
            
            unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count() + i;
            std::mt19937 thread_gen(seed);

            double imm_reward = _calculate_reward(*v, move, s_ctx, state);
            
            MDDVRPSRC_Environment pds_env = env; 
            
            int v_idx = -1;
            for(size_t k=0; k<pds_env.get_state().vehicles.size(); ++k) {
                if (pds_env.get_state().vehicles[k].id == v->id) { v_idx = k; break; }
            }
            
            if(v_idx != -1) pds_env.assign_destination(v_idx, move);

            double future_val = 0;
            if (num_simulations > 0) {
                double sim_sum = 0;
                for(int n=0; n<num_simulations; ++n) {
                    sim_sum += _run_pds_ra_simulation(pds_env, thread_gen); 
                }
                future_val = sim_sum / num_simulations;
            }

            results[i] = {move, imm_reward + future_val};
        }

        int best_move = v->location;
        double max_val = -std::numeric_limits<double>::infinity();

        for (const auto& res : results) {
            if (res.score > max_val) {
                max_val = res.score;
                best_move = res.move;
            }
        }
        
        decisions[v->id] = best_move;
    }
    return decisions;
}