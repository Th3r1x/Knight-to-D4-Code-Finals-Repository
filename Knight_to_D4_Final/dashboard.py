from io import StringIO
from plotly.subplots import make_subplots
import math
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import subprocess, json, os
import sys
import time
import warnings

# Suppress Plotly FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Page Configuration ---
st.set_page_config(
    page_title = "MDDVRPSRC Dashboard (Final)",
    page_icon = "🚑",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# --- Helper Functions ---

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates the bearing between two points."""
    if abs(lat1 - lat2) < 1e-9 and abs(lon1 - lon2) < 1e-9:
        return 0.0
    
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)

    d_lon = lon2_r - lon1_r

    y = math.sin(d_lon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon)
    bearing = math.atan2(y, x)
    return (math.degrees(bearing) + 360) % 360

@st.cache_data
def load_default_data():
    """Loads default sample data into pandas DataFrames."""
    # 30 Nodes (ID 0-29)
    nodes_csv = """id,name,type,lat,lon,demand
0,Makati Depot,depot,14.5547,121.0244,0
1,Mandaluyong Depot,depot,14.5790,121.0350,0
2,Pasig Depot,depot,14.5850,121.0600,0
3,Taguig Depot,depot,14.5200,121.0500,0
4,Evac Center A,shelter,14.5550,121.0340,120
5,Evac Center B,shelter,14.5600,121.0200,150
6,Evac Center C,shelter,14.5700,121.0250,100
7,Evac Center D,shelter,14.5400,121.0300,90
8,Evac Center E,shelter,14.5300,121.0450,110
9,Evac Center F,shelter,14.5900,121.0500,130
10,Evac Center G,shelter,14.5800,121.0700,140
11,Evac Center H,shelter,14.5100,121.0600,80
12,Evac Center I,shelter,14.5500,121.0600,95
13,Evac Center J,shelter,14.5750,121.0100,105
14,Junction 1,connecting,14.5620,121.0280,0
15,Junction 2,connecting,14.5650,121.0400,0
16,Junction 3,connecting,14.5500,121.0400,0
17,Junction 4,connecting,14.5450,121.0200,0
18,Junction 5,connecting,14.5350,121.0500,0
19,Junction 6,connecting,14.5250,121.0400,0
20,Junction 7,connecting,14.5950,121.0400,0
21,Junction 8,connecting,14.5850,121.0200,0
22,Junction 9,connecting,14.5750,121.0550,0
23,Junction 10,connecting,14.5650,121.0650,0
24,Junction 11,connecting,14.5550,121.0550,0
25,Junction 12,connecting,14.5450,121.0650,0
26,Junction 13,connecting,14.5350,121.0750,0
27,Junction 14,connecting,14.5150,121.0700,0
28,Junction 15,connecting,14.5050,121.0550,0
29,Junction 16,connecting,14.5950,121.0650,0
"""
    edges_csv = """u,v,weight,max_capacity,time,damage
0,14,0.8,10,4,0
0,17,1.2,10,5,0
1,21,0.9,10,4,0
1,22,1.1,10,5,0
2,20,1.0,10,4,0
2,23,1.3,10,5,0
3,19,0.8,10,3,0
3,27,1.5,10,6,0
4,14,0.5,8,3,0
4,16,0.6,8,3,0
5,14,0.7,8,4,0
5,21,1.0,8,5,0
6,21,0.6,8,3,0
6,22,0.8,8,4,0
7,17,0.5,8,3,0
7,19,0.9,8,5,0
8,18,0.4,8,2,0
8,19,0.7,8,4,0
9,20,0.6,8,3,0
9,22,0.8,8,4,0
10,23,0.7,8,4,0
10,29,0.9,8,5,0
11,27,0.6,8,3,0
11,28,0.8,8,4,0
12,16,0.5,8,3,0
12,24,0.7,8,4,0
13,21,1.2,8,5,0
14,15,1.0,12,4,0
14,16,0.9,12,4,0
15,16,0.8,12,3,0
15,22,1.1,12,5,0
15,24,1.3,12,6,0
16,17,1.0,12,4,0
16,24,0.9,12,4,0
17,18,1.2,12,5,0
18,19,0.8,12,3,0
18,25,1.1,12,5,0
19,28,1.4,12,6,0
20,21,1.0,12,4,0
20,29,1.2,12,5,0
21,22,0.9,12,4,0
22,23,1.1,12,5,0
22,29,1.5,12,6,0
23,24,1.0,12,4,0
23,26,1.3,12,5,0
24,25,0.9,12,4,0
25,26,1.1,12,5,0
26,27,1.2,12,5,0
27,28,1.0,12,4,0
"""
    nodes_df = pd.read_csv(StringIO(nodes_csv)).set_index('id')
    edges_df = pd.read_csv(StringIO(edges_csv))
    nodes_df['lat'] = nodes_df['lat'].astype(float)
    nodes_df['lon'] = nodes_df['lon'].astype(float)
    return nodes_df, edges_df

def create_graph_from_dfs(nodes_df, edges_df):
    G = nx.Graph()
    for idx, row in nodes_df.iterrows():
        name = row.get('name', f"Node {idx}")
        G.add_node(int(idx), name=name, type=row['type'], pos=(row['lon'], row['lat']), demand=int(row['demand']))
    for _, row in edges_df.iterrows():
        G.add_edge(int(row['u']), int(row['v']), weight=float(row['weight']), time=float(row['time']), max_capacity=int(row['max_capacity']), damage=int(row.get('damage', 0)))
    return G

def run_simulation(graph, vehicle_configs, heuristic_name, solver_params):
    nodes = [{"id": n, "name": d.get("name", str(n)), "type": d["type"], "lat": d["pos"][1], "lon": d["pos"][0], "demand": d["demand"]} for n, d in graph.nodes(data=True)]
    edges = [{"u": u, "v": v, "weight": d["weight"], "time": d["time"], "max_capacity": d["max_capacity"], "damage": d["damage"]} for u, v, d in graph.edges(data=True)]
    
    data = {
        "nodes": nodes,
        "edges": edges,
        "vehicles": [{"capacity": c, "start": s} for c, s in vehicle_configs],
        "heuristic": heuristic_name,
        "num_simulations": solver_params.get("num_simulations", 3),
        "lookahead_horizon": solver_params.get("lookahead_horizon", 7),
        "max_steps": 1000
    }

    exe_name = "solver_app.exe" if sys.platform.startswith("win") else "solver_app"
    exe_path = os.path.join(os.getcwd(), exe_name)

    if not os.path.exists(exe_path):
        files_here = os.listdir(os.getcwd())
        return [], f"Solver executable not found at '{exe_path}'. Files in working dir: {files_here}", 0.0

    try:
        start_time = time.perf_counter()
        result = subprocess.run(
            [exe_path],
            input=json.dumps(data).encode("utf-8"),
            capture_output=True,
            timeout=86400
        )
        runtime = time.perf_counter() - start_time

        if result.returncode != 0:
            return [], f"Solver Error:\n{result.stderr.decode()}", runtime

        output_str = result.stdout.decode()
        if not output_str.strip():
             return [], "No output from solver.", runtime

        history = json.loads(output_str)
        return history, None, runtime

    except Exception as e:
        return [], str(e), 0.0

def calculate_summary_statistics(history, nodes_df):
    if not history: return {}, [], []
    initial, final = history[0], history[-1]
    
    total_initial = sum(initial['demands'].values()) if initial.get('demands') else 0
    total_remaining = sum(final.get('demands', {}).values())
    total_served = total_initial - total_remaining
    pct_served = (total_served / total_initial * 100) if total_initial > 0 else 100.0

    # 1. Vehicle Stats
    vehicle_stats = []
    vehicle_paths = {v['id']: [v['location']] for v in initial['vehicles']}
    
    for step in history:
        for v in step['vehicles']:
            vid = v['id']
            loc = v['location']
            if vehicle_paths[vid][-1] != loc:
                vehicle_paths[vid].append(loc)

    for v_idx, v_init in enumerate(initial['vehicles']):
        v_final = final['vehicles'][v_idx]
        vid = v_init['id']
        move_steps = sum(1 for step in history if step['vehicles'][v_idx]['is_moving'])
        
        vehicle_stats.append({
            "Vehicle ID": vid,
            "Capacity": v_init['max_capacity'],
            "Time Moving (%)": f"{(move_steps / len(history)) * 100:.1f}%",
            "Final Location": v_final['location'],
            "Route History": " → ".join(map(str, vehicle_paths[vid]))
        })
    
    # 2. Service Logs (Delivery Events)
    service_logs = []
    prev_demands = {int(k): v for k, v in initial['demands'].items()}
    
    for i in range(1, len(history)):
        prev_step = history[i-1]
        curr_step = history[i]
        curr_time = curr_step['time']
        
        curr_demands = {int(k): v for k, v in curr_step['demands'].items()}
        
        # Check all nodes for demand drop
        for nid, d_curr in curr_demands.items():
            d_prev = prev_demands.get(nid, 0)
            if d_prev > d_curr:
                amount = d_prev - d_curr
                
                vehicle_id = -1
                for v in curr_step['vehicles']:
                    if v['location'] == nid and not v['is_moving']:
                        vehicle_id = v['id']
                        break
                
                node_type = nodes_df.loc[nid, 'type'] if nid in nodes_df.index else 'unknown'
                
                if vehicle_id != -1 and amount > 0.001 and node_type == 'shelter':
                    service_logs.append({
                        "Time": f"{curr_time:.2f}",
                        "Node ID": nid,
                        "Node Name": nodes_df.loc[nid, 'name'] if nid in nodes_df.index else f"Node {nid}",
                        "Vehicle ID": vehicle_id,
                        "Amount Delivered": f"{amount:.1f}"
                    })
        
        prev_demands = curr_demands

    # 3. Resupply Logs
    resupply_logs = []
    resupply_counts = {} # Key: (depot_id, vehicle_id) -> count

    for i in range(1, len(history)):
        prev_step = history[i-1]
        curr_step = history[i]
        
        # Check each vehicle for capacity increase
        for v_curr in curr_step['vehicles']:
            vid = v_curr['id']
            # Find corresponding vehicle in previous step
            v_prev = next((v for v in prev_step['vehicles'] if v['id'] == vid), None)
            
            if v_prev:
                # If capacity increased and vehicle is at a location (not moving or just arrived)
                if v_curr['capacity'] > v_prev['capacity'] and not v_curr['is_moving']:
                    location_id = v_curr['location']
                    node_type = nodes_df.loc[location_id, 'type'] if location_id in nodes_df.index else 'unknown'
                    
                    if node_type == 'depot':
                        key = (location_id, vid)
                        resupply_counts[key] = resupply_counts.get(key, 0) + 1

    for (depot_id, vid), count in resupply_counts.items():
        resupply_logs.append({
            "Depot ID": depot_id,
            "Depot Name": nodes_df.loc[depot_id, 'name'] if depot_id in nodes_df.index else f"Depot {depot_id}",
            "Vehicle ID": vid,
            "Resupply Count": count
        })
    
    # Sort by Depot ID then Vehicle ID
    resupply_df = pd.DataFrame(resupply_logs)
    if not resupply_df.empty:
        resupply_df = resupply_df.sort_values(by=["Depot ID", "Vehicle ID"])

    return {
        "total_served": total_served,
        "pct_served": pct_served,
        "total_time": final['time'],
        "total_dist": final['total_distance'],
        "vehicle_stats": pd.DataFrame(vehicle_stats)
    }, pd.DataFrame(service_logs), resupply_df

# --- Interpolation Logic ---

def preprocess_vehicle_movements(history):
    """
    Extracts discrete movement segments for every vehicle from the history.
    Returns: { vid: [ {t_start, t_end, start_loc, end_loc, capacity, max_capacity}, ... ] }
    """
    movements = {}
    
    # Init with vehicle IDs
    if not history: return {}
    for v in history[0]['vehicles']:
        movements[v['id']] = []

    for step in history:
        for v in step['vehicles']:
            vid = v['id']
            if v['is_moving']:
                # Unique signature for a specific trip leg
                sig = (v['departure_time'], v['arrival_time'])
                
                # Check if this leg is already recorded
                is_new = True
                if movements[vid]:
                    last = movements[vid][-1]
                    if (last['t_start'], last['t_end']) == sig:
                        is_new = False
                
                if is_new:
                    movements[vid].append({
                        't_start': v['departure_time'],
                        't_end': v['arrival_time'],
                        'start_loc': v['location'], 
                        'end_loc': v['destination'],
                        'capacity': v['capacity'],
                        'max_capacity': v['max_capacity']
                    })
    return movements

def get_interpolated_state_robust(history, movements, nodes_df, current_time):
    """
    Calculates exact position for all vehicles at current_time.
    Uses pre-calculated movements to ensure continuity and correct capacity.
    """
    
    # 1. Demand State (Discrete Step Logic)
    step_idx = 0
    for i, step in enumerate(history):
        if step['time'] <= current_time: step_idx = i
        else: break
    curr_step = history[step_idx]
    
    demands = curr_step.get('demands', {})
    
    # UPDATED: Show names in table if available
    pending_demands = []
    
    # Filter out zero demands to keep the table clean, but show active/blocked ones
    for k, v in demands.items():
        if v > 0:
            node_name = f"Node {k}"
            if int(k) in nodes_df.index and 'name' in nodes_df.columns:
                node_name = nodes_df.loc[int(k), 'name']
            
            # Check if it's the last step and still positive -> likely blocked/unreachable
            status = f"{v:.0f}"
            if current_time >= history[-1]['time'] and v > 0:
                 status += " (Blocked/Unreachable)"
                 
            pending_demands.append([node_name, status])

    if not pending_demands: pending_demands = [["-", "All Served"]]
    
    d_res = {
        "header": ["Node", "Pending Demand"], 
        "cells": list(map(list, zip(*pending_demands))) if pending_demands else [[], []],
        "raw_demands": demands # Return raw demands for color logic
    }

    # 2. Vehicle State (Continuous Interpolation)
    v_res = {'lats': [], 'lons': [], 'angles': [], 'texts': [], 'ids': [], 'hover': []}
    
    # Get all vehicle IDs from the initial state
    all_vids = [v['id'] for v in history[0]['vehicles']]
    
    for vid in all_vids:
        lat, lon = 0, 0
        angle = 0
        
        # Default: Assume stationary at the last known location from the discrete step
        # This acts as the "fallback" if we aren't in a movement window
        fallback_v = next((v for v in curr_step['vehicles'] if v['id'] == vid), None)
        capacity = fallback_v['capacity'] if fallback_v else 0
        max_cap = fallback_v['max_capacity'] if fallback_v else 0
        
        if fallback_v and fallback_v['location'] in nodes_df.index:
            node = nodes_df.loc[fallback_v['location']]
            lat, lon = node['lat'], node['lon']

        # Check for Active Movement
        active_move = None
        if vid in movements:
            for move in movements[vid]:
                # If current_time is within a movement window, use it
                if move['t_start'] <= current_time <= move['t_end']:
                    active_move = move
                    break
        
        if active_move:
            # Override capacity with the value during the move
            capacity = active_move['capacity']
            max_cap = active_move['max_capacity']
            
            # Interpolate Position
            if active_move['start_loc'] in nodes_df.index and active_move['end_loc'] in nodes_df.index:
                start_node = nodes_df.loc[active_move['start_loc']]
                end_node = nodes_df.loc[active_move['end_loc']]
                
                duration = active_move['t_end'] - active_move['t_start']
                if duration > 0:
                    frac = (current_time - active_move['t_start']) / duration
                else:
                    frac = 1.0
                
                frac = max(0.0, min(1.0, frac))
                
                lat = start_node['lat'] + (end_node['lat'] - start_node['lat']) * frac
                lon = start_node['lon'] + (end_node['lon'] - start_node['lon']) * frac
                
                if active_move['start_loc'] != active_move['end_loc']:
                    angle = calculate_bearing(start_node['lat'], start_node['lon'], end_node['lat'], end_node['lon'])

        v_res['lats'].append(lat)
        v_res['lons'].append(lon)
        v_res['angles'].append(angle)
        v_res['texts'].append(f"V{vid}")
        v_res['ids'].append(vid)
        v_res['hover'].append(f"Vehicle {vid}<br>Cap: {capacity:.0f} / {max_cap:.0f}")

    return v_res, d_res

def create_animated_dashboard_plot(history, nodes_df, edges_df):
    """Generates a combined Map + Table Plotly figure with interpolated frames."""
    
    movements = preprocess_vehicle_movements(history)
    
    # Using Scattermap (MapLibre) instead of Scattermapbox per request
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.8, 0.2], 
        specs=[[{"type": "scattermap"}], [{"type": "table"}]],
        vertical_spacing=0.05
    )

    # --- 1. Static Layer: Edges ---
    edge_x, edge_y = [], []
    for _, row in edges_df.iterrows():
        u_idx, v_idx = row['u'], row['v']
        if u_idx in nodes_df.index and v_idx in nodes_df.index:
            u, v = nodes_df.loc[u_idx], nodes_df.loc[v_idx]
            edge_x.extend([u['lon'], v['lon'], None])
            edge_y.extend([u['lat'], v['lat'], None])
    
    fig.add_trace(go.Scattermap(
        lon=edge_x, lat=edge_y, mode='lines', 
        line=dict(width=1.5, color='gray'), 
        hoverinfo='none', showlegend=False
    ), row=1, col=1)

    # --- 2. Static Layer: Nodes ---
    # Depots and connecting nodes are static. Shelters are dynamic.
    for ntype, color in [('depot', 'blue'), ('connecting', 'gray')]:
        subset = nodes_df[nodes_df['type'] == ntype]
        if not subset.empty:
            size = 8 if ntype == 'connecting' else 12
            
            # UPDATED: Custom Data for Hover
            names = subset['name'] if 'name' in subset.columns else subset.index.astype(str)
            
            fig.add_trace(go.Scattermap(
                lon=subset['lon'].tolist(), 
                lat=subset['lat'].tolist(), 
                mode='markers+text',
                marker=dict(size=size, color=color, opacity=1, allowoverlap=True),
                text=subset.index.astype(str), textposition="top center",
                textfont=dict(color="black"), 
                name=ntype.capitalize(),
                customdata=names,
                hovertemplate="<b>%{customdata}</b> (ID: %{text})<br>Type: " + ntype,
            ), row=1, col=1)

    # --- 3. Dynamic Layers Init (t=0) ---
    v_init, d_init = {'lats': [], 'lons': [], 'angles': [], 'texts': [], 'hover': [], 'ids': []}, {"header": ["Node", "Demand"], "cells": [[], []], "raw_demands": {}}
    max_t = 1.0
    if history:
        max_t = max(history[-1]['time'], 1.0)
        v_init, d_init = get_interpolated_state_robust(history, movements, nodes_df, 0.0)

    # Dynamic Shelter Trace (Init)
    shelter_nodes = nodes_df[nodes_df['type'] == 'shelter']
    shelter_lats = shelter_nodes['lat'].tolist()
    shelter_lons = shelter_nodes['lon'].tolist()
    shelter_ids = shelter_nodes.index.tolist()
    
    # Initial colors and hover text based on demand
    init_colors = []
    init_hover = []
    
    # Check if 'name' exists, otherwise use ID
    shelter_names = shelter_nodes['name'] if 'name' in shelter_nodes.columns else shelter_nodes.index.astype(str)
    
    for sid, sname in zip(shelter_ids, shelter_names):
        dem = d_init['raw_demands'].get(str(sid), nodes_df.loc[sid]['demand']) 
        if dem <= 0:
            init_colors.append('green')
            hover_txt = f"<b>{sname}</b> (ID: {sid})<br>Status: Served (0 Left)"
        else:
            init_colors.append('red')
            hover_txt = f"<b>{sname}</b> (ID: {sid})<br>Pending: {dem:.0f}"
        init_hover.append(hover_txt)

    fig.add_trace(go.Scattermap(
        lon=shelter_lons, lat=shelter_lats,
        mode='markers+text',
        marker=dict(size=12, color=init_colors, opacity=1, allowoverlap=True),
        text=[str(i) for i in shelter_ids], textposition="top center",
        textfont=dict(color="black"),
        name='Shelters (Dynamic)',
        hovertext=init_hover,
        hoverinfo="text"
    ), row=1, col=1)

    # Vehicle Trace
    fig.add_trace(go.Scattermap(
        lon=v_init['lons'], lat=v_init['lats'], 
        mode='markers+text',
        marker=dict(
            size=15, 
            color='#228B22', 
            symbol='triangle', 
            angle=v_init['angles'],
            opacity=1.0,
            allowoverlap=True 
        ),
        ids=v_init['ids'], 
        text=v_init['texts'], textposition="middle center",
        textfont=dict(color="white"),
        hovertext=v_init['hover'], hoverinfo='text',
        name='Vehicles'
    ), row=1, col=1)
    
    # Table Trace
    fig.add_trace(go.Table(
        header=dict(values=d_init['header'], fill_color='paleturquoise', align='left'),
        cells=dict(values=d_init['cells'], fill_color='lavender', align='left')
    ), row=2, col=1)

    shelter_trace_idx = len(fig.data) - 3
    vehicle_trace_idx = len(fig.data) - 2
    table_trace_idx = len(fig.data) - 1

    # --- 4. Frames Generation ---
    if history:
        num_frames = min(max(100, 2 * len(history)), 500)
        times = np.linspace(0, max_t, num_frames)
        frames = []
        
        for t in times:
            v_state, d_state = get_interpolated_state_robust(history, movements, nodes_df, t)
            
            # Calculate colors and hover for this frame
            frame_colors = []
            frame_hover = []
            
            for sid, sname in zip(shelter_ids, shelter_names):
                dem = d_state['raw_demands'].get(str(sid), 0)
                if dem <= 0:
                    frame_colors.append('green')
                    frame_hover.append(f"<b>{sname}</b> (ID: {sid})<br>Status: Served (0 Left)")
                else:
                    frame_colors.append('red')
                    frame_hover.append(f"<b>{sname}</b> (ID: {sid})<br>Pending: {dem:.0f}")

            frames.append(go.Frame(
                data=[
                    go.Scattermap(
                        marker=dict(size=12, color=frame_colors, opacity=1, allowoverlap=True),
                        hovertext=frame_hover
                    ),
                    go.Scattermap(
                        lon=v_state['lons'],
                        lat=v_state['lats'],
                        marker=dict(
                            angle=v_state['angles'], 
                            allowoverlap=True,
                            size=15,
                            color='#228B22',
                            symbol='triangle',
                            opacity=1.0
                        ), 
                        ids=v_state['ids'],
                        text=v_state['texts'],
                        textfont=dict(color="white"),
                        hovertext=v_state['hover']
                    ),
                    go.Table(
                        header=dict(values=d_state['header']),
                        cells=dict(values=d_state['cells'])
                    )
                ],
                name=f"{t:.2f}", 
                traces=[shelter_trace_idx, vehicle_trace_idx, table_trace_idx] 
            ))
        
        fig.frames = frames

    # --- 5. Layout & Controls ---
    if not nodes_df.empty:
        lat_center = nodes_df['lat'].mean()
        lon_center = nodes_df['lon'].mean()
    else:
        lat_center, lon_center = 0, 0

    fig.update_layout(
        height=850,
        margin={"r":10,"t":100,"l":10,"b":10}, 
        
        map=dict(
            style="carto-positron", 
            zoom=13,
            center=dict(lat=lat_center, lon=lon_center),
            domain={'y': [0.25, 1.0]} 
        ),
        
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),

        yaxis2=dict(domain=[0.0, 0.20]), 
        
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "y": 1.2, 
            "xanchor": "left", "yanchor": "top",
            "pad": {"t": 0, "r": 10},
            "direction": "left",
            "buttons": [{
                "label": "▶️ Play",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": 750, "redraw": True},
                    "fromcurrent": True, 
                    "transition": {"duration": 0},
                    "mode": "immediate"
                }]
            }, {
                "label": "⏸️ Pause",
                "method": "animate",
                "args": [[None], {
                    "frame": {"duration": 0, "redraw": False}, 
                    "mode": "immediate", 
                    "transition": {"duration": 0}
                }]
            }]
        }],
        
        sliders=[{
            "steps": [
                {
                    "args": [[f"{t:.2f}"], {
                        "frame": {"duration": 0, "redraw": True}, 
                        "mode": "immediate",
                        "transition": {"duration": 0} 
                    }],
                    "label": f"{t:.1f} m", 
                    "method": "animate"
                }
                for t in np.linspace(0, max_t, num_frames if history else 2)
            ],
            "currentvalue": {"prefix": "Mission Time: ", "visible": True, "xanchor": "center"},
            "len": 0.6, "x": 0.5, "y": 1.2, 
            "xanchor": "center",
            "pad": {"t": 0}
        }] if history else []
    )
    
    return fig

def create_path_summary_plot(history, nodes_df, edges_df):
    """Generates a static map showing the full path of each vehicle."""
    
    fig = go.Figure()

    # --- 1. Static Layer: Edges ---
    edge_x, edge_y = [], []
    for _, row in edges_df.iterrows():
        u_idx, v_idx = row['u'], row['v']
        if u_idx in nodes_df.index and v_idx in nodes_df.index:
            u, v = nodes_df.loc[u_idx], nodes_df.loc[v_idx]
            edge_x.extend([u['lon'], v['lon'], None])
            edge_y.extend([u['lat'], v['lat'], None])
    
    fig.add_trace(go.Scattermap(
        lon=edge_x, lat=edge_y, mode='lines', 
        line=dict(width=1.5, color='gray'), 
        hoverinfo='none', showlegend=False
    ))

    # --- 2. Static Layer: Nodes (Final State) ---
    # Calculate final state colors for shelters
    final_demands = history[-1]['demands'] if history else {}
    
    for ntype, color in [('depot', 'blue'), ('connecting', 'gray')]:
        subset = nodes_df[nodes_df['type'] == ntype]
        if not subset.empty:
            size = 8 if ntype == 'connecting' else 12
            names = subset['name'] if 'name' in subset.columns else subset.index.astype(str)
            
            fig.add_trace(go.Scattermap(
                lon=subset['lon'].tolist(), 
                lat=subset['lat'].tolist(), 
                mode='markers+text',
                marker=dict(size=size, color=color, opacity=1, allowoverlap=True),
                text=subset.index.astype(str), textposition="top center",
                textfont=dict(color="black"), 
                name=ntype.capitalize(),
                customdata=names,
                hovertemplate="<b>%{customdata}</b> (ID: %{text})<br>Type: " + ntype,
            ))

    # Shelters (Colored by final demand status)
    shelter_nodes = nodes_df[nodes_df['type'] == 'shelter']
    if not shelter_nodes.empty:
        shelter_lats = shelter_nodes['lat'].tolist()
        shelter_lons = shelter_nodes['lon'].tolist()
        shelter_ids = shelter_nodes.index.tolist()
        shelter_names = shelter_nodes['name'] if 'name' in shelter_nodes.columns else shelter_nodes.index.astype(str)
        
        final_colors = []
        final_hover = []
        for sid, sname in zip(shelter_ids, shelter_names):
            dem = final_demands.get(str(sid), nodes_df.loc[sid]['demand'])
            if dem <= 0:
                final_colors.append('green')
                final_hover.append(f"<b>{sname}</b> (ID: {sid})<br>Status: Served (0 Left)")
            else:
                final_colors.append('red')
                final_hover.append(f"<b>{sname}</b> (ID: {sid})<br>Pending: {dem:.0f}")

        fig.add_trace(go.Scattermap(
            lon=shelter_lons, lat=shelter_lats,
            mode='markers+text',
            marker=dict(size=12, color=final_colors, opacity=1, allowoverlap=True),
            text=[str(i) for i in shelter_ids], textposition="top center",
            textfont=dict(color="black"),
            name='Shelters (Final Status)',
            hovertext=final_hover,
            hoverinfo="text"
        ))

    # --- 3. Vehicle Paths ---
    if history:
        # Extract full paths for each vehicle
        initial_vehicles = history[0]['vehicles']
        vehicle_paths = {v['id']: [v['location']] for v in initial_vehicles}
        
        for step in history:
            for v in step['vehicles']:
                vid = v['id']
                loc = v['location']
                # Only add if location changed to avoid duplicates in path drawing
                if vehicle_paths[vid][-1] != loc:
                    vehicle_paths[vid].append(loc)
        
        # Distinct colors for vehicles (up to 20)
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
            '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194'
        ]

        for i, v_init in enumerate(initial_vehicles):
            vid = v_init['id']
            path_node_ids = vehicle_paths[vid]
            color = colors[i % len(colors)]
            
            path_lats = []
            path_lons = []
            
            for nid in path_node_ids:
                if nid in nodes_df.index:
                    node = nodes_df.loc[nid]
                    path_lats.append(node['lat'])
                    path_lons.append(node['lon'])
            
            # Draw Path Lines
            fig.add_trace(go.Scattermap(
                lon=path_lons, lat=path_lats,
                mode='lines',
                line=dict(width=4, color=color),
                opacity=0.8,
                name=f"Vehicle {vid} Path",
                hoverinfo='skip',
                legendgroup=f"V{vid}"  # Group path with end marker
            ))
            
            # Draw End Marker (Circle)
            if path_lats:
                fig.add_trace(go.Scattermap(
                    lon=[path_lons[-1]], lat=[path_lats[-1]],
                    mode='markers+text',
                    marker=dict(size=15, color=color, symbol='circle', opacity=1.0),
                    text=[f"V{vid}"], textposition="middle center",
                    textfont=dict(color="white", size=10),
                    name=f"Vehicle {vid} End",
                    showlegend=False,  # Hide from legend, part of group
                    hoverinfo='name',
                    legendgroup=f"V{vid}"  # Group end marker with path
                ))

    # --- 4. Layout ---
    if not nodes_df.empty:
        lat_center = nodes_df['lat'].mean()
        lon_center = nodes_df['lon'].mean()
    else:
        lat_center, lon_center = 0, 0

    fig.update_layout(
        height=600,
        margin={"r":10,"t":30,"l":10,"b":10}, 
        map=dict(
            style="carto-positron", 
            zoom=13,
            center=dict(lat=lat_center, lon=lon_center)
        ),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
    )
    
    return fig

# --- Main App Logic ---

if 'sim_data' not in st.session_state: st.session_state.sim_data = None

# Initialize default data
if 'node_data' not in st.session_state:
    def_nodes, def_edges = load_default_data()
    st.session_state.node_data = def_nodes
    st.session_state.edge_data = def_edges

# Data Loader Callbacks
def load_nodes_file():
    uploaded = st.session_state.u_nodes
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if 'id' in df.columns: df.set_index('id', inplace=True)
            if 'lat' in df.columns: df['lat'] = df['lat'].astype(float)
            if 'lon' in df.columns: df['lon'] = df['lon'].astype(float)
            
            if 'name' not in df.columns:
                df['name'] = "Node " + df.index.astype(str)
            
            st.session_state.node_data = df
        except Exception as e:
            st.error(f"Error reading nodes: {e}")

def load_edges_file():
    uploaded = st.session_state.u_edges
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.edge_data = df
        except Exception as e:
            st.error(f"Error reading edges: {e}")

def reset_data():
    def_nodes, def_edges = load_default_data()
    st.session_state.node_data = def_nodes
    st.session_state.edge_data = def_edges

with st.sidebar:
    st.title("🔧 Config")
    
    st.subheader("Load Saved Run")
    uploaded_run = st.file_uploader("run.json", type="json", key="load_run")
    if uploaded_run is not None:
        try:
            loaded_data = json.load(uploaded_run)
            st.session_state.node_data = pd.read_json(StringIO(loaded_data['nodes_csv']))
            st.session_state.edge_data = pd.read_json(StringIO(loaded_data['edges_csv']))
            st.session_state.sim_data = loaded_data['sim_data']
            
            # Restore parameters to session state to update widgets
            params = loaded_data.get('solver_params', {})
            
            if 'heuristic' in params: st.session_state['solver_heuristic'] = params['heuristic']
            if 'num_simulations' in params: st.session_state['solver_nsim'] = params['num_simulations']
            if 'lookahead_horizon' in params: st.session_state['solver_horizon'] = params['lookahead_horizon']
            if 'n_veh' in params: st.session_state['fleet_nveh'] = params['n_veh']
            
            # Restore vehicle configs
            if 'vehicle_configs' in params:
                for i, (cap, start) in enumerate(params['vehicle_configs']):
                    st.session_state[f"vcap_{i}"] = cap
                    st.session_state[f"vstart_{i}"] = start
            
            st.success(f"Loaded run: {loaded_data.get('label', 'Unnamed')}")
            
        except Exception as e:
            st.error(f"Error loading run: {e}")

    st.subheader("Data Management")
    c1, c2 = st.columns(2)
    with c1:
        st.file_uploader("nodes.csv", type='csv', key='u_nodes', on_change=load_nodes_file)
    with c2:
        st.file_uploader("edges.csv", type='csv', key='u_edges', on_change=load_edges_file)
        
    if st.button("Reset to Sample Data", width='stretch'):
        reset_data()
        st.rerun()

    st.divider()
    st.subheader("Network Topology Editor")
    with st.expander("Edit Nodes", expanded=False):
        nodes_df = st.data_editor(st.session_state.node_data, num_rows="dynamic", key="editor_nodes")
    
    with st.expander("Edit Edges", expanded=False):
        edges_df = st.data_editor(st.session_state.edge_data, num_rows="dynamic", key="editor_edges")

    st.divider()
    st.subheader("Solver Parameters")
    heuristic = st.selectbox("Heuristic", ["TBIH-1 (Random)", "TBIH-2 (DSIH)", "TBIH-3 (DCW)", "TBIH-4 (DLA-SIH)", "TBIH-5 (DLA-CW)"], key="solver_heuristic")
    n_sim = st.slider("Simulations", 1, 20, 3, key="solver_nsim")
    horizon = st.slider("Lookahead Horizon", 1, 50, 20, key="solver_horizon")
    
    st.divider()
    st.subheader("Fleet Config")
    n_veh = st.number_input("Vehicle Count", 1, 100, 2, key="fleet_nveh")
    
    veh_configs = []
    valid_start_nodes = nodes_df.index.tolist()
    
    for i in range(n_veh):
        c1, c2 = st.columns(2)
        with c1:
            # Ensure key exists in session state if we increased n_veh
            if f"vcap_{i}" not in st.session_state: st.session_state[f"vcap_{i}"] = 100
            cap = st.number_input(f"V{i} Cap", 10, 1000000000, key=f"vcap_{i}")
        with c2:
            default_idx = 0
            if f"vstart_{i}" in st.session_state:
                val = st.session_state[f"vstart_{i}"]
                if val in valid_start_nodes:
                    default_idx = valid_start_nodes.index(val)
            
            start_node = st.selectbox(f"V{i} Start", valid_start_nodes, index=default_idx, key=f"vstart_{i}")
        veh_configs.append((cap, start_node))

    if st.button("🚀 Run Simulation", type="primary", width='stretch'):
        if nodes_df.empty or edges_df.empty:
             st.error("Nodes or Edges list cannot be empty.")
        else:
            graph = create_graph_from_dfs(nodes_df, edges_df)
            hist, err, runtime = run_simulation(graph, veh_configs, heuristic, {"num_simulations": n_sim, "lookahead_horizon": horizon})
            
            if err:
                st.error(err)
            else:
                st.session_state.sim_data = {"history": hist, "runtime": runtime}
                st.success(f"Solved in {runtime:.4f}s")

# --- Main Display ---

st.title("🚑 AGOS: Aid & Goods Optimization System")

if st.session_state.sim_data:
    data = st.session_state.sim_data
    hist = data['history']
    
    if not hist:
        st.warning("Simulation returned no history.")
    else:
        stats, service_logs, resupply_df = calculate_summary_statistics(hist, st.session_state.node_data)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Solver Runtime", f"{data['runtime']:.3f}s")
        m2.metric("Demand Served", f"{stats['pct_served']:.1f}%", f"{stats['total_served']} units")
        m3.metric("Total Distance", f"{stats['total_dist']:.1f} km")
        m4.metric("Mission Time", f"{stats['total_time']:.1f} min")

        st.divider()

        st.subheader("Live Route Animation & Real-Time Status")
        
        fig = create_animated_dashboard_plot(hist, nodes_df, edges_df)
        st.plotly_chart(fig, width='stretch')

        st.divider()

        st.subheader("Vehicle Route History")
        st.dataframe(stats['vehicle_stats'], hide_index=True, width='stretch')
        
        st.divider()

        st.subheader("Path Summary")
        
        path_fig = create_path_summary_plot(hist, nodes_df, edges_df)
        st.plotly_chart(path_fig, width='stretch')

        st.divider()
        
        st.subheader("Service Summary")
        if not service_logs.empty:
            st.dataframe(service_logs, hide_index=True, width='stretch')
        else:
            st.info("No deliveries recorded yet.")

        st.divider()

        st.subheader("Resupply Summary")
        if not resupply_df.empty:
            st.dataframe(resupply_df, hide_index=True, width='stretch')
        else:
            st.info("No resupplies recorded.")

        st.divider()
        
        st.subheader("Save This Run")
        run_label = st.text_input("Label for This Run (Optional)", value="my_run")
        
        c1, c2, c3 = st.columns([0.16, 0.16, 0.68])
        
        with c1:
            generate_clicked = st.button("Generate Save File")
            
        if generate_clicked:
            # Capture current configs manually to ensure they match what's on screen
            current_configs = []
            for i in range(n_veh):
                c = st.session_state.get(f"vcap_{i}", 100)
                s = st.session_state.get(f"vstart_{i}", valid_start_nodes[0] if valid_start_nodes else None)
                current_configs.append((c, s))

            run_data = {
                "label": run_label,
                "nodes_csv": st.session_state.node_data.to_json(),
                "edges_csv": st.session_state.edge_data.to_json(),
                "solver_params": {
                    "heuristic": heuristic,
                    "num_simulations": n_sim,
                    "lookahead_horizon": horizon,
                    "n_veh": n_veh,
                    "vehicle_configs": current_configs
                },
                "sim_data": st.session_state.sim_data
            }
            st.session_state.run_data_json = json.dumps(run_data)
            st.session_state.run_label = run_label
        
        with c2:
            if 'run_data_json' in st.session_state:
                st.download_button(
                    label="Download JSON",
                    data=st.session_state.run_data_json,
                    file_name=f"{st.session_state.get('run_label', 'run')}.json",
                    mime="application/json",
                    type="primary"
                )

else:
    st.info("👈 Upload CSVs, edit data manually, or just click Run to start.")
    
    fig = create_animated_dashboard_plot([], st.session_state.node_data, st.session_state.edge_data)
    st.plotly_chart(fig, width='stretch')

    st.markdown("""
    ### About
    This dashboard visualizes a **Multi-Depot Dynamic Vehicle Routing Problem with Stochastic Road Capacity (MDDVRPSRC)**.
    
    **Latest Features:**
    - **Named Nodes:** CSV now supports a `name` column.
    - **High Contrast:** Black text for nodes, white text for vehicles.
    - **Larger Vehicles:** Vehicle markers increased to 18px.
    - **Improved UI:** Clearer file upload labels.
    """)