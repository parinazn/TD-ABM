# Multiplex Network Diffusion Simulator

A Streamlit-based web application for simulating and visualizing how information, viruses, or behaviors spread across multiplex networks (networks with multiple layers of connections).

## 🧪 About the Project

Real-world systems often involve multiple types of interactions. For example, a virus might spread via physical contact (Layer 1), while information about the virus spreads via social media (Layer 2). This simulator allows you to model these "coupled" spreading processes.

### Key Features
* **Multiplex Modeling:** Simulate two distinct network layers with different topologies:
    * **Watts-Strogatz:** Small-world networks (high clustering).
    * **Stochastic Block Model (SBM):** Community-based networks.
    * **Custom Uploads:** Support for uploading your own Edge Lists (`.csv`, `.txt`) or GEXF files.
* **Flexible Diffusion Rules:**
    * **SI (Simple Contagion):** Once infected, stays infected. Ideal for rumors or simple viruses.
    * **SIS (Re-infection):** Agents can recover and become susceptible again. Ideal for flu-like diseases.
    * **Threshold (Complex Contagion):** Agents only adopt if $k$ neighbors have already adopted. Ideal for social norms or expensive technologies.
* **Layer Timing:** Control the speed of each layer (e.g., fast "online" spread vs. slow "offline" interactions).
* **Visualizations:** Interactive time-series plots and side-by-side network graph rendering.

## 🛠 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

Run the web application locally using Streamlit:

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.

## 📂 File Structure

* `streamlit_app.py`: The frontend user interface. Handles user inputs, visualizations, and runs the simulation loop.
* `main_simulations.py`: The backend logic. Contains the `NetworkModel` and `NetworkAgent` classes (built on [Mesa](https://mesa.readthedocs.io/)).
* `requirements.txt`: List of external Python dependencies.

## ⚙️ Configuration Options

The sidebar provides the following controls:

### 1. Global Network Params

* **Number of Nodes:** Total population size (overridden if a file is uploaded).
* **Max Steps:** Duration of the simulation.
* **Topology Seed:** Ensures reproducible network structures.

### 2. Seeding Strategy

Determine how "Patient Zero" is selected:

* **Random:** Arbitrary selection.
* **Degree Centrality:** High-degree nodes (Hubs) are infected first.
* **Betweenness Centrality:** Bridge nodes (Connectors) are infected first.

### 3. Layer Configuration (Layer 1 & Layer 2)

For each layer, you can independently configure:

* **Diffusion Model:** SI, SIS, or Threshold.
* **Topology:** Mathematical generator or File Upload.
* **Timing:** How often this layer processes updates (e.g., every step vs. every 3rd step).

## 📦 Requirements

* Python 3.8+
* streamlit
* mesa
* networkx
* pandas
* matplotlib

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
