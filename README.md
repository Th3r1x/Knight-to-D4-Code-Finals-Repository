# Knight-to-D4-Code-Finals-Repository
This code repsitory was used for the submission as finalist of the PJDSC2025 contest


# AGOS: Aid & Goods Optimization System

For the application to work, all the files within the finals folder must all be in the same working directory. 
In terminal, firt compile the c++ code

    g++ -std=c++20 main.cpp solver.cpp -o solver_app

Then, run the streamlit command in terminal to run the program

    streamlit run dashboard.py 
Note: streamlit extension must be installed by the uesr

Dashboard Walkthrough

WARNING: USING THE CURRENT MAP AND NODES PROVIDED, IT TAKES A SIGNIFICANT AMOUNT OF TIME TO RUN THE OUTPUT (>4 HOURS)
Therefore it is advisable to use the sample map built into the code to test an output

# Side Panel

On the side panel, 3 settings will be presented to the user

## Setting 1: Data input

The user may choose to use a JSON save file from a previously ran solution or input csv files. Otherwise, a default map will be presented which provides a number of depots, shelter , and connector nodes. From here, the user may immedately move on to the next settings should they choose to use this map. 

Otherwise, the user may upload two csv files
 - "nodes" csv file with columns: 
  
        id, type, lat, lon, demand 

   - "id" is a unique integer for every node;
   - "type" is either depot or shelter;
   - "lat" and "lon" are floats and coordinates of the node;
   - "demand" is the demand for a node (important, nodes with the "depot" type should have a value of 0 in the demand column)
   
 - "edges" csv file with columns:

        u, v, weight, max_capacity, time, damage

   - "u" and "v" are connected nodes and correspond to the id's in the nodes csv;
   - "weight" is the distance in km between u and v;
   - "max_capacity" is the maximum number of supported vehicles on that road(edge);
   -  "time" is the time it takes in min to travel between u and v;
   -  "damage" is an integer from 0-10 to indicate road damage

The user may then manually edit certain portions of the following csv files in the network topology area

## Setting 2: Heuristics

The user selects the type of heuristic based on the paper from 

Anuar, W. K., Lee, L. S., Seow, H.-V., & Pickl, S. (2022). A Multi-Depot Dynamic Vehicle Routing Problem with Stochastic Road Capacity: An MDP Model and Dynamic Policy for Post-Decision State Rollout Algorithm in Reinforcement Learning.

The user can set the number of simulations that the selected heuristic will run to check the most optimal solution amongst the number of simulations. The most optimal according to the paper is the fourth heuristic, however the user may test the other heuristics to according to their needs

WARNING THE HEURISTICS ARE VERY COMPUTATIONALLY HEAVY AND WILL TAKE A VERY LONG TIME TO LOAD ON A LARGE GRAPH

The user can also set the number of "look-ahead" that the algorithm runs per step for the algorithm to "look ahead" n number of steps before committing to a solution. 

The user is then able to run the simulation to output a solution

## Setting 2: Vehicles

The user manually assigns the number of vehicles, the capacity of each vehicle and the id of the node where each vehicle starts

# Map 

In the middle of the dashboard, a map of the nodes and the edges connecting them is presented. The user is presented with a map of the nodes as well as a visualization of the depot cars moving towards their respective destinations. Below this map is a table that displays the unserved which will update in real time according to the map.

The second map is used to display the full route of each vehicle as well as the routes of the vehicles as a collectiv. Each route can be toggled by double clicking on the legend.

# Statistics

Below the map are the following statistics presented:
- Route history of each vehicle
- Timestamp when each shelter has been served and by which vehicle
- The depots and the number of times each vehicle resupplied at each depot

These statistics are meant for user to gain insight and provide an adequate response based on the simulation

# Option to save to JSON file

An option to save the solution that was run is made to avoid having to rerun simulations from the start which is very time consuming.

