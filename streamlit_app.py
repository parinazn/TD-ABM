import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os
import random

from main_simulations import ViralNetworkModel, State, ModelType

st.set_page_config(page_title="Agent-Based Network Simulation", layout="wide")
st.title("🕸️ Spread of Information on Networks")

# =======================================================
# 1. NETWORK & SEED CONFIGURATION
# =======================================================
st.sidebar.header("1. Network Configuration")

# --- NETWORK SOURCE ---
source_type = st.sidebar.radio("Source:", ["Synthetic (Watts-Strogatz)", "Upload CSV"])

preview_graph = None

if source_type == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Edge List (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None, names=["source", "target"])
        G_raw = nx.from_pandas_edgelist(df, "source", "target")
        preview_graph = nx.convert_node_labels_to_integers(G_raw)
        st.sidebar.success(f"Loaded {preview_graph.number_of_nodes()} nodes.")
else:
    use_fixed_seed = st.sidebar.checkbox("Use Fixed Seed", value=True)
    # --- SEED CONTROL ---
    if use_fixed_seed:
        graph_seed = st.sidebar.number_input("Seed Value", min_value=0, value=42)
    else:
        # Note on "Random" Mode: We use session_state to hold the graph structure constant
        # until the user specifically asks for a new one.
        if "random_seed" not in st.session_state:
            st.session_state.random_seed = random.randint(0, 10000)
        
        if st.sidebar.button("🎲 Generate New Random Graph"):
            st.session_state.random_seed = random.randint(0, 10000)
            
        graph_seed = st.session_state.random_seed
        st.sidebar.caption(f"Graph Structure Seed: {graph_seed}")
    num_nodes = st.sidebar.slider("Nodes", 10, 200, 50)
    avg_degree = st.sidebar.slider("Avg Degree", 2, 10, 4)
    rewiring_prob = st.sidebar.slider("Rewiring Probability", 0.0, 1.0, 0.1)
    
    # Generate Preview Graph
    preview_graph = nx.watts_strogatz_graph(n=num_nodes, k=avg_degree, p=rewiring_prob, seed=graph_seed)

# =======================================================
# 2. PATIENT ZERO SELECTION
# =======================================================
st.sidebar.markdown("---")
st.sidebar.header("2. Seed Node Selection")

seed_method = st.sidebar.radio("Selection Method", ["Random", "Manual (Targeted)"])
fixed_seeds = None
initial_outbreak = 1

if seed_method == "Random":
    initial_outbreak = st.sidebar.slider("Random Initial Seeds", 1, 10, 1)
else:
    # Analyze the Preview Graph
    degrees = dict(preview_graph.degree())
    node_options = sorted(preview_graph.nodes(), key=lambda x: degrees[x], reverse=True)
    node_labels = {n: f"Node {n} (Conn: {degrees[n]})" for n in node_options}
    
    selected_indices = st.sidebar.multiselect(
        "Select Specific Nodes:",
        options=node_options,
        format_func=lambda x: node_labels[x],
        default=[node_options[0]] if node_options else None
    )
    fixed_seeds = selected_indices
    st.sidebar.info(f"Selected {len(fixed_seeds)} seeds.")

# =======================================================
# 3. MODEL CONFIGURATION
# =======================================================
st.sidebar.markdown("---")
st.sidebar.header("3. Spreading Rules")

model_choice = st.sidebar.selectbox("Model", [m.value for m in ModelType])
selected_model_enum = next(m for m in ModelType if m.value == model_choice)

recovery_prob = 0.0
threshold = 2
spread_prob = 0.3

if selected_model_enum == ModelType.SI:
    spread_prob = st.sidebar.slider("Spread Chance", 0.0, 1.0, 0.3)
elif selected_model_enum == ModelType.SIS:
    spread_prob = st.sidebar.slider("Spread Chance", 0.0, 1.0, 0.3)
    recovery_prob = st.sidebar.slider("Recovery Chance", 0.0, 1.0, 0.1)
elif selected_model_enum == ModelType.THRESHOLD:
    threshold = st.sidebar.slider("Threshold", 1, 5, 2)

steps = st.sidebar.slider("Steps", 5, 100, 30)

# =======================================================
# 4. EXECUTION MODE
# =======================================================
st.sidebar.markdown("---")
st.sidebar.header("4. Execution Mode")
mode = st.sidebar.radio("Mode", ["Single Run (Visual)", "Batch Run (Scientific)"])

if mode == "Batch Run (Scientific)":
    num_runs = st.sidebar.number_input("Number of Runs", min_value=5, max_value=100, value=30)

# =======================================================
# MAIN RUN LOGIC
# =======================================================
st.sidebar.markdown("---")
st.sidebar.header("5. Run Simulation")
if st.sidebar.button("Run Simulation"):
    
    # Check if a graph exists
    if preview_graph is None:
        st.error("Please configure a network first.")
    else:
        # ---------------------------------------------------
        # DETERMINING SIMULATION SEED
        # ---------------------------------------------------
        # If the user UNCHECKED "Fixed Seed", they expect the simulation 
        # to vary every time they click Run, even if the graph structure 
        # is held constant by the Session State.
        if use_fixed_seed:
            simulation_seed = graph_seed
        else:
            simulation_seed = None  # Randomize the spread dynamics

        # ---------------------------------------------------
        # MODE A: SINGLE RUN
        # ---------------------------------------------------
        if mode == "Single Run (Visual)":
            model = ViralNetworkModel(
                num_nodes=preview_graph.number_of_nodes(),
                avg_degree=0, # Ignored since we pass custom_graph
                spread_prob=spread_prob,
                initial_outbreak=initial_outbreak,
                seed=simulation_seed, 
                model_type=selected_model_enum,
                recovery_prob=recovery_prob,
                threshold=threshold,
                custom_graph=preview_graph.copy(), # Pass a copy to be safe
                fixed_seeds=fixed_seeds
            )

            progress_bar = st.progress(0)
            for i in range(steps):
                model.step()
                progress_bar.progress((i + 1) / steps)

            results_df = model.datacollector.get_model_vars_dataframe()
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("📊 Metrics")
                st.write("Width (Count)")
                st.line_chart(results_df["Width (Count)"])
                st.write("Depth (Hops)")
                st.line_chart(results_df["Depth (Hops)"])
            
            with col2:
                st.subheader("🕸️ Final State")
                net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
                pos = nx.spring_layout(model.G, seed=graph_seed) # Layout uses GRAPH seed
                agent_by_pos = {a.pos: a for a in model.agents}
                
                for node_id in model.G.nodes():
                    agent = agent_by_pos.get(node_id)
                    color = "#00C0F2"
                    size = 5
                    if agent and agent.state == State.INFECTED:
                        color = "#FF4B4B"
                        if fixed_seeds and node_id in fixed_seeds:
                            color = "#FFFF00"
                            size = 10
                    
                    x_pos = pos[node_id][0] * 1000
                    y_pos = pos[node_id][1] * 1000
                    net.add_node(node_id, label=str(node_id), x=x_pos, y=y_pos, color=color, size=size)
                
                for edge in model.G.edges():
                    net.add_edge(edge[0], edge[1], color="#555555")

                net.toggle_physics(False)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
                    path = tmpfile.name
                try:
                    net.save_graph(path)
                    with open(path, 'r', encoding='utf-8') as f:
                        html_data = f.read()
                    components.html(html_data, height=550)
                finally:
                    if os.path.exists(path):
                        os.remove(path)

        # ---------------------------------------------------
        # MODE B: BATCH RUN (SCIENTIFIC)
        # ---------------------------------------------------
        else:
            all_runs_width = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for r in range(num_runs):
                status_text.text(f"Simulating Run {r+1}/{num_runs}...")
                
                model = ViralNetworkModel(
                    num_nodes=preview_graph.number_of_nodes(),
                    avg_degree=0,
                    spread_prob=spread_prob,
                    initial_outbreak=initial_outbreak,
                    seed=None, # ALWAYS Random for Batch stats
                    model_type=selected_model_enum,
                    recovery_prob=recovery_prob,
                    threshold=threshold,
                    custom_graph=preview_graph.copy(), # Reuse fixed structure
                    fixed_seeds=fixed_seeds
                )
                
                for _ in range(steps):
                    model.step()
                
                run_data = model.datacollector.get_model_vars_dataframe()["Width (Count)"].values
                all_runs_width.append(run_data)
                progress_bar.progress((r + 1) / num_runs)
            
            # Plotting
            data_matrix = np.array(all_runs_width)
            avg_curve = np.mean(data_matrix, axis=0)
            std_curve = np.std(data_matrix, axis=0)
            
            st.subheader(f"📈 Aggregate Results ({num_runs} Runs)")
            fig, ax = plt.subplots(figsize=(10, 5))
            x = range(len(avg_curve))
            ax.plot(x, avg_curve, label="Average Infection", color="blue", linewidth=2)
            ax.fill_between(x, avg_curve - std_curve, avg_curve + std_curve, color="blue", alpha=0.2, label="Std Dev")
            
            ax.set_title(f"Spread Dynamics: {selected_model_enum.value}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Infected Count")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)