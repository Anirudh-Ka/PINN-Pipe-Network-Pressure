# PINN Pressure Prediction in Pipe Networks

A **Physics-Informed Neural Network (PINN)** that predicts pressure distribution across a pipe network by embedding fluid mechanics constraints directly into the training loss — no labeled pressure data required.

**Qualified for Smart India Hackathon 2024.**

---

## What it does

Traditional neural networks need labeled training data. This PINN learns by satisfying physics equations:

- **Continuity equation** — flow in = flow out at every interior junction
- **Darcy-Weisbach equation** — relates pressure drop to flow velocity across each pipe
- **Boundary conditions** — known inlet/outlet pressures are enforced as loss penalties

The network takes no meaningful input — it learns the pressure field as its weights, guided entirely by physics.

---

## Network

5 junctions, 6 pipes, randomized physical properties (length, diameter, friction factor).

```
Input (zeros) → FC(50) → Tanh → FC(50) → Tanh → FC(10) → Tanh → FC(5) → Pressures
```

| Parameter | Value |
|---|---|
| Junctions | 5 |
| Pipes | 6 |
| Epochs | 20,000 |
| Optimizer | Adam (lr=0.001) |
| Activation | Tanh |
| Inlet pressure | 100 Pa |
| Outlet pressure | 50 Pa |

---

## How to run

```bash
pip install torch numpy matplotlib networkx
jupyter notebook PINN_PipeNetwork.ipynb
```

Run all cells in order. The final cell saves `pinn_results.png` with the training loss curve and pressure distribution visualization.

---

## Results

The model converges to a pressure distribution that satisfies continuity at all interior junctions while respecting the inlet/outlet boundary conditions. Results are visualized as a color-mapped pipe network graph.

---

## Project structure

```
├── PINN_PipeNetwork.ipynb      # Clean notebook: model, training, visualization
├── pinn_results.png            # Output: loss curve + pressure distribution
├── pipe_network_pinn_model.pth # Saved model weights
└── README.md
```

---

## Physics background

**Darcy-Weisbach equation:**
$$\Delta P = f \cdot \frac{L}{D} \cdot \frac{\rho v^2}{2}$$

**Continuity (mass conservation):**
$$\sum Q_{in} = \sum Q_{out} \quad \text{at each junction}$$

These constraints replace the need for labeled training data — the loss function *is* the physics.

---

*Built as part of ML coursework, submitted to Smart India Hackathon 2024*
