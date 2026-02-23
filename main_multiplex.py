import mesa
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random
from enum import Enum


# --- ENUMERATIONS FOR DIFFUSION MODEL TYPES AND AGENT STATES ---
class ModelType(Enum):
    SI = "SI"   # Susceptible-infected model; single parameter: prob. of infecting others
    SIS = "SIS" # Susceptible-infected-susceptible model; Two parameters: probs of infecting others and recovering
    THRESHOLD = "Threshold (#)"   # Threshold model; single parameter: # of infected neighbors that will trigger infection

class State(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1    # This state means that infection/information has reached an agent/node


# --- AGENT DEFINITION ---
class NetworkAgent(mesa.Agent):
    def __init__(self, model, initial_state):
        super().__init__(model)
        # Use a private variable during initialization to bypass property setter 
        # before the agent is officially placed on the model.
        self._state = initial_state 

    # State polling
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        # Automatically update the model's centralized active sets whenever an agent's state changes
        if self._state == State.SUSCEPTIBLE and value == State.INFECTED:
            self.model.active_infected_set.add(self.pos)
            self.model.newly_infected_this_step.add(self.pos)
        elif self._state == State.INFECTED and value == State.SUSCEPTIBLE:
            self.model.active_infected_set.discard(self.pos)
        self._state = value

    # Agent behavior in "Push" models of spread (e.g. SI, SIS)
    def try_to_infect_neighbors(self, G_layer, spread_prob):
        neighbors_ids = list(G_layer.neighbors(self.pos))
        if not neighbors_ids:
            return
            
        # Fetch all neighbor agents at once for efficiency and map them
        neighbors = self.model.grid.get_cell_list_contents(neighbors_ids)
        neighbor_dict = {n.pos: n for n in neighbors}

        # Push spread to susecptible neighbors in activated layer; success affected by spread_prob and weight 
        for n_id in neighbors_ids:
            neighbor = neighbor_dict[n_id]
            if neighbor.state == State.SUSCEPTIBLE:
                weight = G_layer[self.pos][n_id].get('weight', 1.0)
                if self.random.random() < (spread_prob * weight):
                    neighbor.state = State.INFECTED


    # Agent behavior in "Pull" models of spread (e.g., Threshold)
    def check_threshold_adoption(self, G_layer, threshold):
        neighbors_ids = list(G_layer.neighbors(self.pos))
        if not neighbors_ids:
            return
        
        # Fetch all neighbor agents at once for efficiency and map them
        neighbors = self.model.grid.get_cell_list_contents(neighbors_ids)
        neighbor_dict = {n.pos: n for n in neighbors}
        
        # Pull infection from infected neighbors in activated layer; success affected by threshold and weight 
        infected_weight_sum = 0
        for n_id in neighbors_ids:
            neighbor = neighbor_dict[n_id]
            if neighbor.state == State.INFECTED:
                weight = G_layer[self.pos][n_id].get('weight', 1.0)
                infected_weight_sum += weight
                
        if infected_weight_sum >= threshold:
            self.state = State.INFECTED

    # Route the agent's behavior based on the activated layer's rules
    def process_layer(self, G_layer, model_type, spread_prob, recovery_prob, threshold):
        if model_type == ModelType.SI:
            if self.state == State.INFECTED:
                self.try_to_infect_neighbors(G_layer, spread_prob)

        elif model_type == ModelType.SIS:
            if self.state == State.INFECTED:
                self.try_to_infect_neighbors(G_layer, spread_prob)
                if self.random.random() < recovery_prob:
                    self.state = State.SUSCEPTIBLE

        elif model_type == ModelType.THRESHOLD:
            if self.state == State.SUSCEPTIBLE:
                self.check_threshold_adoption(G_layer, threshold)

    # Agents actions when activated; includes layer timing control
    def step(self):
        # Layer 1 Processing
        if self.model.current_step % self.model.timing1 == 0:
            self.process_layer(
                self.model.G1, 
                self.model.model1_type, 
                self.model.spread_prob1, 
                self.model.recovery_prob1, 
                self.model.threshold1
            )
        # Layer 2 Processing
        if self.model.current_step % self.model.timing2 == 0:
            self.process_layer(
                self.model.G2, 
                self.model.model2_type, 
                self.model.spread_prob2, 
                self.model.recovery_prob2, 
                self.model.threshold2
            )


# --- MODEL DEFINITION ---
class NetworkModel(mesa.Model):
    def __init__(self, 
                 num_nodes=50, 
                 network1_type="watts_strogatz", avg_degree1=3, rewiring_prob1=0.2,
                 network2_type="sbm", avg_degree2=3, rewiring_prob2=0.2,
                 initial_outbreak=1, fixed_seeds=None, seeding_strategy="random",
                 model1_type=ModelType.SI, spread_prob1=0.1, recovery_prob1=0.0, threshold1=2, timing1=1,
                 model2_type=ModelType.THRESHOLD, spread_prob2=0.0, recovery_prob2=0.0, threshold2=2, timing2=5,
                 graph_seed=None, infected_node_seed=None, random_seed=None): 
        
        super().__init__(seed=random_seed)

        self.num_nodes = num_nodes
        self.current_step = 0  # Manual step tracker for modulo arithmetic needed for layer timing

        # Layer 1 parameters
        self.model1_type = model1_type
        self.spread_prob1 = spread_prob1
        self.recovery_prob1 = recovery_prob1
        self.threshold1 = threshold1
        self.timing1 = timing1

        # Layer 2 parameters
        self.model2_type = model2_type
        self.spread_prob2 = spread_prob2
        self.recovery_prob2 = recovery_prob2
        self.threshold2 = threshold2
        self.timing2 = timing2

        # Setup tracking sets for iIfected nodes and new infections
        self.active_infected_set = set()
        self.newly_infected_this_step = set()

        # Setup the multiplex network
        self.G1 = self._generate_graph(network1_type, num_nodes, avg_degree1, rewiring_prob1, graph_seed)
        self.G2 = self._generate_graph(network2_type, num_nodes, avg_degree2, rewiring_prob2, graph_seed)
        
        # Use layer 1 to initialize the Mesa grid for spatial placing/lookup of agents
        self.grid = mesa.space.NetworkGrid(self.G1)

        # Generate and place susceptible agents
        for node_id in self.G1.nodes():
            a = NetworkAgent(self, State.SUSCEPTIBLE)
            self.grid.place_agent(a, node_id)

        # Create an "flattend" graph for centrality-based seeding
        G_combined = nx.Graph()
        G_combined.add_nodes_from(self.G1.nodes())
        G_combined.add_edges_from(self.G1.edges())
        G_combined.add_edges_from(self.G2.edges())

        # Setup initially infected nodes
        initial_agents = []
        if fixed_seeds is not None and len(fixed_seeds) > 0:
            initial_agents = [a for a in self.agents if a.pos in fixed_seeds]
        else:
            actual_outbreak_size = min(initial_outbreak, self.num_nodes)
            
            if seeding_strategy == "degree":
                centrality = nx.degree_centrality(G_combined)
                sorted_nodes = sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)
                top_nodes = sorted_nodes[:actual_outbreak_size]
                initial_agents = [a for a in self.agents if a.pos in top_nodes]
                
            elif seeding_strategy == "betweenness":
                centrality = nx.betweenness_centrality(G_combined)
                sorted_nodes = sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)
                top_nodes = sorted_nodes[:actual_outbreak_size]
                initial_agents = [a for a in self.agents if a.pos in top_nodes]
                
            else: # "random"
                local_rng = random.Random(infected_node_seed)
                initial_agents = local_rng.sample(list(self.agents), actual_outbreak_size)

        # Update the state of initial infected seed set, and update active set    
        for a in initial_agents:
            a.state = State.INFECTED 
        self.initial_infected_ids = list(self.active_infected_set)

        # Data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Overall Width": compute_width,
                "Depth L1 (Hops)": lambda m: compute_depth_layer(m, m.G1),
                "Depth L2 (Hops)": lambda m: compute_depth_layer(m, m.G2),
                "Newly Infected Nodes": lambda m: list(m.newly_infected_this_step)
            }
        )
        self.datacollector.collect(self)

    def _generate_graph(self, network_type, num_nodes, avg_degree, rewiring_prob, seed):
        """Helper to generate distinct layers based on type."""
        if network_type == "watts_strogatz":
            G = nx.watts_strogatz_graph(n=num_nodes, k=avg_degree, p=rewiring_prob, seed=seed)
        elif network_type == "sbm":
            size1 = num_nodes // 2
            size2 = num_nodes - size1
            p_in = min(1.0, (avg_degree * 0.8) / size1) if size1 > 0 else 0
            p_out = min(1.0, (avg_degree * 0.2) / size2) if size2 > 0 else 0
            p_matrix = [[p_in, p_out], [p_out, p_in]]
            G = nx.Graph(nx.stochastic_block_model([size1, size2], p_matrix, seed=seed))
        else:
            raise ValueError(f"Unknown network_type: {network_type}")
            
        local_rng = random.Random(seed)
        for u, v in G.edges():
            G[u][v]['weight'] = local_rng.uniform(0.1, 1.0)
        return G

    def step(self):
        self.newly_infected_this_step.clear()
        
        self.agents.shuffle_do("step")
        
        self.datacollector.collect(self)
        self.current_step += 1


# --- PERFORMANCE EVALUATION FUNCTIONS ---

# Overall width (# of infected nodes) across *both* layers
def compute_width(model):
    return len(model.active_infected_set)

# Calculates maximum depth (furthest reached node) restricting traversal to a *specific* layer's edges
def compute_depth_layer(model, layer_graph):
    if not model.active_infected_set: 
        return 0
        
    try:
        # Calculate shortest paths from seeds using ONLY this layer's edges
        path_lengths = nx.multi_source_dijkstra_path_length(
            layer_graph, sources=model.initial_infected_ids, weight=None
        )
        
        # Keep only distances for nodes that are currently infected AND reachable in this layer
        infected_distances = [
            path_lengths[n_id] 
            for n_id in model.active_infected_set 
            if n_id in path_lengths
        ]
        
        return max(infected_distances) if infected_distances else 0
        
    except (nx.NetworkXNoPath, ValueError):
        return 0


# --- EXPERIMENT RUNNER AND DATA ANALYSIS ---
def run_experiment(
    max_steps=50, iterations=1, 
    save_csv=False, csv_filename="simulation_results.csv",
    plot_metrics=False, visualize_network_at_end=False,
    num_nodes=[50], initial_outbreak=[1], seeding_strategy=["random"],
    
    # Layer 1 Configurations
    n1_type=["watts_strogatz"], a_deg1=[3], m1_type=[ModelType.SI], s_prob1=[0.1], r_prob1=[0.0], thres1=[2], t1=[1],
    
    # Layer 2 Configurations
    n2_type=["sbm"], a_deg2=[3], m2_type=[ModelType.THRESHOLD], s_prob2=[0.0], r_prob2=[0.0], thres2=[2], t2=[5],
    
    graph_seed=None, infected_node_seed=None, random_seed=None
):
    def to_list(val): return val if isinstance(val, list) else [val]
    
    param_combinations = list(itertools.product(
        to_list(num_nodes), to_list(initial_outbreak), to_list(seeding_strategy),
        to_list(n1_type), to_list(a_deg1), to_list(m1_type), to_list(s_prob1), to_list(r_prob1), to_list(thres1), to_list(t1),
        to_list(n2_type), to_list(a_deg2), to_list(m2_type), to_list(s_prob2), to_list(r_prob2), to_list(thres2), to_list(t2)
    ))

    all_data = []
    print(f"Beginning Experiment: Testing {len(param_combinations)} Parameter Configurations...")

    for config_id, params in enumerate(param_combinations):
        (n_nodes, i_outbreak, s_strat, 
         n1, a1, m1, s_p1, r_p1, th1, t_1,
         n2, a2, m2, s_p2, r_p2, th2, t_2) = params
        
        for i in range(iterations):
            model = NetworkModel(
                num_nodes=n_nodes, initial_outbreak=i_outbreak, seeding_strategy=s_strat,
                network1_type=n1, avg_degree1=a1, model1_type=m1, spread_prob1=s_p1, recovery_prob1=r_p1, threshold1=th1, timing1=t_1,
                network2_type=n2, avg_degree2=a2, model2_type=m2, spread_prob2=s_p2, recovery_prob2=r_p2, threshold2=th2, timing2=t_2,
                graph_seed=graph_seed, infected_node_seed=infected_node_seed, random_seed=random_seed
            )
            
            absorption_step = -1
            
            for step in range(max_steps):
                model.step()
                
                # Check for Absorption (No new infections for SI/Threshold, or full absorption for SIS)
                if absorption_step == -1:
                    if m1 == ModelType.SIS or m2 == ModelType.SIS:
                        if len(model.active_infected_set) == 0 or len(model.active_infected_set) == n_nodes:
                            absorption_step = step + 1
                    else:
                        if len(model.newly_infected_this_step) == 0:
                            absorption_step = step + 1
                            
            df = model.datacollector.get_model_vars_dataframe()
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Step'}, inplace=True)
            
            df['Config ID'] = config_id
            df['Iteration'] = i
            df['L1 Type'] = n1
            df['L2 Type'] = n2
            df['Seeding'] = s_strat
            df['Absorption Step'] = absorption_step if absorption_step != -1 else "Did Not Absorb"
            
            all_data.append(df)
            
            if visualize_network_at_end and i == 0:
                plt.figure(figsize=(8, 6))
                # Visualize G1 structure
                pos = nx.spring_layout(model.G1, seed=42) 
                color_map = ["yellow" if a.pos in model.initial_infected_ids else "red" if a.state == State.INFECTED else "lightblue" for a in model.agents]
                weights = [model.G1[u][v].get('weight', 1.0) * 2 for u, v in model.G1.edges()]
                
                nx.draw(model.G1, pos, node_color=color_map, with_labels=True, node_size=300, 
                        font_color="black", font_weight='bold', width=weights, edge_color="gray")
                plt.title(f"Final State: Layer 1 Topology", fontweight='bold')
                plt.show()

    final_df = pd.concat(all_data, ignore_index=True)

    if save_csv:
        final_df.to_csv(csv_filename, index=False)
        print(f"Results successfully saved to {csv_filename}")

    if plot_metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for config_id in final_df['Config ID'].unique():
            config_data = final_df[final_df['Config ID'] == config_id]
            
            summary_df = config_data.groupby('Step').agg({
                'Overall Width': ['mean', 'std'],
                'Depth L1 (Hops)': ['mean', 'std'],
                'Depth L2 (Hops)': ['mean', 'std']
            }).reset_index()
            
            label = f"Config {config_id}"

            # Plot Overall Width (cumulative over both layers)
            ax1.plot(summary_df['Step'], summary_df['Overall Width']['mean'], label=label, linewidth=2)
            if iterations > 1:
                ax1.fill_between(summary_df['Step'], 
                                 summary_df['Overall Width']['mean'] - summary_df['Overall Width']['std'],
                                 summary_df['Overall Width']['mean'] + summary_df['Overall Width']['std'],
                                 alpha=0.15)

            # Plot Depth (separately for each layer)
            ax2.plot(summary_df['Step'], summary_df['Depth L1 (Hops)']['mean'], label=f"{label} (Layer 1)", linewidth=2, linestyle='-')
            ax2.plot(summary_df['Step'], summary_df['Depth L2 (Hops)']['mean'], label=f"{label} (Layer 2)", linewidth=2, linestyle='--')

        ax1.set_xlabel('Simulation Step', fontweight='bold')
        ax1.set_ylabel('Overall Width (Infected Count)', fontweight='bold')
        ax1.set_title("Spreading Progress: Width")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Simulation Step', fontweight='bold')
        ax2.set_ylabel('Depth (Hops)', fontweight='bold')
        ax2.set_title("Spreading Progress: Depth per Layer")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return final_df


# Example Test Execution
if __name__ == "__main__":
    df = run_experiment(
        max_steps=50, 
        iterations=3,
        save_csv=False, 
        plot_metrics=True, 
        visualize_network_at_end=False,
        num_nodes=50, 
        initial_outbreak=2, 
        seeding_strategy=["random"], 
        
        # Test 1: Fast Online SI spread vs Slow Offline Threshold spread
        n1_type="watts_strogatz", m1_type=ModelType.SI, s_prob1=0.15, t1=1,
        n2_type="sbm", m2_type=ModelType.THRESHOLD, thres2=1.5, t2=[3, 5, 7],
        
        graph_seed=42,
        infected_node_seed=42
    )