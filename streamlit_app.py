import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from main_multiplex import NetworkModel, ModelType, State

st.set_page_config(page_title="Multiplex Diffusion Sim", layout="wide")

# --- CSS FOR STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    div[data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("⚙️ Simulation Settings")

# 1. Global Network Settings
st.sidebar.header("1. Global Network Params")

num_nodes = st.sidebar.slider(
    "Number of Nodes", 10, 200, 50, 10, 
    help="The total population size (number of agents) in the network."
)

max_steps = st.sidebar.slider(
    "Max Steps", 10, 200, 50, 10,
    help="How long the simulation runs. One step includes potential updates for both layers depending on timing."
)

graph_seed = st.sidebar.number_input(
    "Graph Topology Seed", value=42,
    help="Fixes the random layout of the network connections. Same seed = same network structure every time."
)

st.sidebar.markdown("---")

# 2. Seeding Strategy
st.sidebar.header("2. Seeding Strategy")
seeding_strategy = st.sidebar.selectbox(
    "Infection Strategy", 
    ["random", "degree", "betweenness"],
    help="""
    How 'Patient Zero' is chosen:
    - **Random**: Picks any node arbitrarily.
    - **Degree**: Picks the most connected nodes (hubs).
    - **Betweenness**: Picks nodes that bridge different communities (bridges).
    """
)

initial_outbreak = st.sidebar.slider(
    "Initial Outbreak Size", 1, 20, 1,
    help="The number of agents infected at Step 0."
)

infected_node_seed = None
if seeding_strategy == "random":
    infected_node_seed = st.sidebar.number_input(
        "Infection RNG Seed", value=42,
        help="Controls the randomness of WHICH specific nodes get infected when using 'Random' strategy."
    )

st.sidebar.markdown("---")

# 3. Layer Configuration Helper
def layer_ui(layer_num, default_timing, default_model_index):
    st.sidebar.header(f"3.{layer_num} Layer {layer_num} Configuration")
    
    # --- A. Diffusion Model Choice ---
    model_str = st.sidebar.selectbox(
        f"L{layer_num} Diffusion Model", 
        ["SI", "SIS", "Threshold"], 
        index=default_model_index, 
        key=f"mod_l{layer_num}",
        help="""
        **SI**: Simple Contagion. Once infected, stays infected. Good for information/rumors.
        **SIS**: Flu-like. Agents can recover and become susceptible again.
        **Threshold**: Complex Contagion. Requires 'k' infected neighbors to adopt. Good for behaviors/social norms.
        """
    )
    
    # Map string to Enum
    model_enum = {
        "SI": ModelType.SI,
        "SIS": ModelType.SIS,
        "Threshold": ModelType.THRESHOLD
    }[model_str]

    # --- B. Model Specific Parameters ---
    spread_prob = 0.0
    recovery_prob = 0.0
    threshold = 0
    
    if model_str in ["SI", "SIS"]:
        spread_prob = st.sidebar.slider(
            f"L{layer_num} Spread Prob", 0.0, 1.0, 0.1, 0.05, 
            key=f"sp_l{layer_num}",
            help="Probability (0-1) that an infected agent will successfully infect a susceptible neighbor in one step."
        )
    
    if model_str == "SIS":
        recovery_prob = st.sidebar.slider(
            f"L{layer_num} Recovery Prob", 0.0, 1.0, 0.1, 0.05, 
            key=f"rp_l{layer_num}",
            help="Probability (0-1) that an infected agent recovers back to Susceptible state."
        )
        
    if model_str == "Threshold":
        threshold = st.sidebar.number_input(
            f"L{layer_num} Threshold (#)", 1, 10, 2, 
            key=f"th_l{layer_num}",
            help="The strict number of infected neighbors required to trigger adoption. (e.g., I need 2 friends to buy it before I do)."
        )

    # --- C. Network Topology & Timing ---
    st.sidebar.caption(f"Layer {layer_num} Topology & Timing")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        net_type = st.selectbox(
            f"Type", ["watts_strogatz", "sbm"], 
            key=f"net_l{layer_num}",
            help="**Watts-Strogatz**: Small-world ring (high clustering).\n**SBM**: Stochastic Block Model (distinct communities)."
        )
        avg_deg = st.number_input(
            f"Avg Degree", 1, 10, 3, 
            key=f"deg_l{layer_num}",
            help="Average number of connections per node."
        )
    with col2:
        timing = st.number_input(
            f"Timing Step", 1, 20, default_timing, 
            key=f"time_l{layer_num}",
            help="Timescale of this layer. 1 = Processes every step. 3 = Processes only every 3rd step (slower/offline layer)."
        )

    st.sidebar.markdown("---")

    return {
        "network_type": net_type, "avg_degree": avg_deg, "timing": timing,
        "model_type": model_enum, "spread_prob": spread_prob, 
        "recovery_prob": recovery_prob, "threshold": threshold
    }

# Configure Layer 1 (Default: SI)
l1_params = layer_ui(1, default_timing=1, default_model_index=0) 

# Configure Layer 2 (Default: Threshold)
l2_params = layer_ui(2, default_timing=3, default_model_index=2) 


# --- MAIN APP LOGIC ---

st.title("Multiplex Network Diffusion Simulator")
st.markdown(f"**Current Setup:** Layer 1 is **{l1_params['model_type'].value}** | Layer 2 is **{l2_params['model_type'].value}** | Seeding: **{seeding_strategy.capitalize()}**")

# Run Button
if st.button("▶️ Run Simulation", type="primary"):
    
    # Initialize Model with new Seeding Parameters
    model = NetworkModel(
        num_nodes=num_nodes,
        initial_outbreak=initial_outbreak,
        seeding_strategy=seeding_strategy,
        
        # Layer 1
        network1_type=l1_params["network_type"],
        avg_degree1=l1_params["avg_degree"],
        model1_type=l1_params["model_type"],
        spread_prob1=l1_params["spread_prob"],
        recovery_prob1=l1_params["recovery_prob"],
        threshold1=l1_params["threshold"],
        timing1=l1_params["timing"],

        # Layer 2
        network2_type=l2_params["network_type"],
        avg_degree2=l2_params["avg_degree"],
        model2_type=l2_params["model_type"],
        spread_prob2=l2_params["spread_prob"],
        recovery_prob2=l2_params["recovery_prob"],
        threshold2=l2_params["threshold"],
        timing2=l2_params["timing"],
        
        graph_seed=graph_seed,
        infected_node_seed=infected_node_seed
    )

    # Run Loop
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(max_steps):
        model.step()
        progress_bar.progress((i + 1) / max_steps)
        status_text.text(f"Simulating Step {i+1}/{max_steps}")
    
    status_text.empty()
    progress_bar.empty()
    
    # Collect Data
    df = model.datacollector.get_model_vars_dataframe()
    
    # --- METRICS ROW ---
    final_width = df["Overall Width"].iloc[-1]
    max_depth_l1 = df["Depth L1 (Hops)"].max()
    max_depth_l2 = df["Depth L2 (Hops)"].max()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Infected Nodes", f"{final_width} / {num_nodes}", help="Total unique nodes infected across both layers")
    col2.metric("Max Depth (Layer 1)", f"{max_depth_l1:.0f} hops", help="Furthest distance reached using only Layer 1 edges")
    col3.metric("Max Depth (Layer 2)", f"{max_depth_l2:.0f} hops", help="Furthest distance reached using only Layer 2 edges")

    # --- PLOTS ROW ---
    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "🕸️ Network Visualizer", "📄 Raw Data"])
    
    with tab1:
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.subheader("Total Infections")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df.index, df["Overall Width"], label="Total Infected", color="#e63946", linewidth=2.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Count")
            ax.set_title("Adoption Curve (Width)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        with col_p2:
            st.subheader("Layer Penetration")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(df.index, df["Depth L1 (Hops)"], label="Layer 1", linestyle="-", color="#1d3557", linewidth=2)
            ax2.plot(df.index, df["Depth L2 (Hops)"], label="Layer 2", linestyle="--", color="#457b9d", linewidth=2)
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Hops from Seed")
            ax2.set_title("Spreading Depth")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

    with tab2:
        st.subheader("Final Network State (Side-by-Side Layers)")
        st.markdown("Same nodes, different connections. **Red** = Infected, **Teal** = Susceptible, **Gold** = Patient Zero.")
        
        # Helper to generate common color map
        color_map = []
        node_sizes = []
        for agent in model.agents:
            if agent.pos in model.initial_infected_ids:
                color_map.append("#FFD700") # Gold for seeds
                node_sizes.append(250)
            elif agent.state == State.INFECTED:
                color_map.append("#FF6B6B") # Red for infected
                node_sizes.append(150)
            else:
                color_map.append("#4ECDC4") # Teal for susceptible
                node_sizes.append(150)

        # Create two columns for side-by-side graphs
        col_g1, col_g2 = st.columns(2)

        # --- LAYER 1 DRAWING ---
        with col_g1:
            st.caption(f"**Layer 1**: {l1_params['network_type']} ({l1_params['model_type'].value})")
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            pos1 = nx.spring_layout(model.G1, seed=42)
            nx.draw(model.G1, pos1, node_color=color_map, node_size=node_sizes, 
                    edge_color="#cfcfcf", width=1.5, alpha=0.8, ax=ax1)
            st.pyplot(fig1)

        # --- LAYER 2 DRAWING ---
        with col_g2:
            st.caption(f"**Layer 2**: {l2_params['network_type']} ({l2_params['model_type'].value})")
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            # Use a different layout calculation for G2 so its unique structure is visible
            pos2 = nx.spring_layout(model.G2, seed=42) 
            nx.draw(model.G2, pos2, node_color=color_map, node_size=node_sizes, 
                    edge_color="#cfcfcf", width=1.5, alpha=0.8, ax=ax2)
            st.pyplot(fig2)

    with tab3:
        st.dataframe(df)

else:
    st.info("👈 Configure settings in the sidebar and click **Run Simulation** to start.")