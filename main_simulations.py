import mesa
import networkx as nx
import numpy as np
from enum import Enum

# --- ENUMERATIONS FOR MODEL TYPES AND STATES ---
class ModelType(Enum):
    SI = "SI (Simple Contagion)"
    SIS = "SIS (Re-infection)"
    THRESHOLD = "Threshold (Complex Contagion)"

class State(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1

# --- AGENT DEFINITION ---
class NetworkAgent(mesa.Agent):
    def __init__(self, model, initial_state):
        super().__init__(model)
        self.state = initial_state

    # --- BEHAVIOR 1: VIRAL SPREAD (Push) ---
    def try_to_infect_neighbors(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        for neighbor in neighbors:
            if neighbor.state == State.SUSCEPTIBLE:
                if self.random.random() < self.model.spread_prob:
                    neighbor.state = State.INFECTED

    # --- BEHAVIOR 2: THRESHOLD ADOPTION (Pull) ---
    def check_threshold_adoption(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        # Count how many neighbors have received the information
        infected_neighbors = sum(1 for n in neighbors if n.state == State.INFECTED)
        
        # If enough neighbors act, I copy them
        if infected_neighbors >= self.model.threshold:
            self.state = State.INFECTED

    def step(self):
        """
        The Agent's Actions. Decisions depend on the selected Model Type.
        """
        # LOGIC FOR SI MODEL
        if self.model.model_type == ModelType.SI:
            if self.state == State.INFECTED:
                self.try_to_infect_neighbors()

        # LOGIC FOR SIS MODEL
        elif self.model.model_type == ModelType.SIS:
            if self.state == State.INFECTED:
                self.try_to_infect_neighbors()
                # Chance to recover (become Susceptible again)
                if self.random.random() < self.model.recovery_prob:
                    self.state = State.SUSCEPTIBLE

        # LOGIC FOR THRESHOLD MODEL
        elif self.model.model_type == ModelType.THRESHOLD:
            if self.state == State.SUSCEPTIBLE:
                self.check_threshold_adoption()

# --- MODEL (INCLUDING GRAPH) DEFINITION ---
class ViralNetworkModel(mesa.Model):
    def __init__(self, num_nodes=50, avg_degree=3, spread_prob=0.1, 
                 initial_outbreak=1, seed=None,
                 model_type=ModelType.SI,
                 recovery_prob=0.0,
                 threshold=2,
                 custom_graph=None,
                 fixed_seeds=None): # <--- Note: ability to set seeds
        
        super().__init__(seed=seed)
        self.model_type = model_type
        self.spread_prob = spread_prob
        self.recovery_prob = recovery_prob
        self.threshold = threshold

        # 1. Graph Setup
        if custom_graph is not None:
            # CASE A: Use User-Provided Graph
            self.G = custom_graph
            self.num_nodes = self.G.number_of_nodes()
        else:
            # CASE B: Generate a Synthetic Graph
            self.num_nodes = num_nodes
            self.G = nx.watts_strogatz_graph(n=num_nodes, k=avg_degree, p=0.1, seed=seed)

        self.grid = mesa.space.NetworkGrid(self.G)

        # 2. Agent Setup
        # Note: We assume the graph nodes are relabeled to 0, ..., N-1
        for node_id in self.G.nodes():
            a = NetworkAgent(self, State.SUSCEPTIBLE)
            self.grid.place_agent(a, node_id)

        # 3. Seed Setup
        initial_agents = []
        
        # CASE A: User manually selected specific nodes
        if fixed_seeds is not None and len(fixed_seeds) > 0:
            # Filter agents whose 'pos' matches the selected seeds
            initial_agents = [a for a in self.agents if a.pos in fixed_seeds]
            
        # CASE B: Random selection of seed set
        else:
            actual_outbreak_size = min(initial_outbreak, self.num_nodes)
            initial_agents = self.random.sample(list(self.agents), actual_outbreak_size)
        
        self.initial_infected_ids = [a.pos for a in initial_agents]
        for a in initial_agents:
            a.state = State.INFECTED

        # 4. Data Collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Width (Count)": compute_width,
                "Depth (Hops)": compute_depth
            }
        )
        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

# --- PERFORMANCE EVALUATION FUNCTIONS ---
def compute_width(model):
    return sum(1 for a in model.agents if a.state == State.INFECTED)

def compute_depth(model):
    infected_ids = [a.pos for a in model.agents if a.state == State.INFECTED]
    if not infected_ids: return 0
    try:
        path_lengths = nx.multi_source_dijkstra_path_length(
            model.G, sources=model.initial_infected_ids
        )
        infected_distances = [path_lengths.get(n_id, 0) for n_id in infected_ids]
        return max(infected_distances) if infected_distances else 0
    except nx.NetworkXNoPath:
        return 0