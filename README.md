<p align="center">
  <img src="assets/logo.png" alt="Minab Logo" width="130" />
</p>

<h1 align="center">ምናብ | MINAB</h1>

<p align="center">
  <strong>Autonomous Aerial Exploration & Vision-Based 3D Spatial Reconstruction Platform</strong>
</p>

<p align="center">
  <em>See. Reconstruct. Understand. Explore.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Minab%20Research-16A34A?style=flat&labelColor=000000" alt="Minab Research" />
  <img src="https://img.shields.io/badge/Identity-Aerospace%20%26%20Vision-16A34A?style=flat&labelColor=000000" alt="Aerospace & Vision" />
  <img src="https://img.shields.io/badge/Accent-%2316A34A-16A34A?style=flat&labelColor=000000" alt="Emerald Green #16A34A" />
  <img src="https://img.shields.io/badge/Python-3.10-FFFFFF?style=flat&labelColor=000000&color=16A34A" alt="Python 3.10" />
  <img src="https://img.shields.io/badge/Engine-OpenCV%20%7C%20Open3D-FFFFFF?style=flat&labelColor=000000&color=16A34A" alt="OpenCV & Open3D" />
  <img src="https://img.shields.io/badge/Depth%20Model-Depth%20Anything%20V2-FFFFFF?style=flat&labelColor=000000&color=16A34A" alt="Depth Anything V2" />
  <img src="https://img.shields.io/badge/Branch-3d--module-16A34A?style=flat&labelColor=000000" alt="Branch 3d-module" />
</p>

<p align="center">
  <img src="assets/uav_exploration.png" alt="Minab Autonomous Aerial Exploration and Spatial Reconstruction" width="100%" />
</p>

---

## Core Identity

**ምናብ / Minab** is an aerospace and computer-vision platform designed to perceive physical environments from aerial imagery and reconstruct them into coherent, colorized 3D spatial models.

The system combines:
* Computer Vision
* Machine Learning
* 3D Reconstruction
* UAV Systems
* Autonomous Exploration
* Spatial Computing

### The Central Idea
> **A UAV explores an environment, Minab turns what it sees into a spatial model, and that model guides what should be explored next.**

This feedback loop makes Minab more than a 3D reconstruction tool: **it is an active aerial exploration system.**

---

## The Minab Loop

```text
                  DEFINE EXPLORATION AREA
                             │
                             ▼
                     PLAN FLIGHT ROUTE
                             │
                             ▼
                         UAV FLIES
                             │
                    ┌────────┴────────┐
                    │                 │
                 CAMERA           TELEMETRY
                    │                 │
                    └────────┬────────┘
                             ▼
                         DATA LINK
                             │
                             ▼
                      GROUND COMPUTER
                             │
                             ▼
                      MINAB PROCESSING
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
           POSE            DEPTH           FRAMES
             │               │               │
             └───────────────┼───────────────┘
                             ▼
                     3D RECONSTRUCTION
                             │
                             ▼
                     COLORED 3D MODEL
                             │
                             ▼
                   SPATIAL UNDERSTANDING
                             │
                             ▼
                  UNCERTAINTY / COVERAGE
                             │
                             ▼
                     NEXT FLIGHT PLAN
                             │
                             └──────────────────────►
```

That final feedback loop—evaluating coverage, identifying occluded areas, and planning the next-best flight route—is what makes Minab an autonomous exploration platform.

---

## What Minab Does Today (Current POC)

The current working proof-of-concept is the vision and 3D reconstruction core located on the **`3d-module`** branch.

The pipeline takes video footage and produces:

```text
Camera Video
     ↓
Frame Extraction
     ↓
Camera Calibration
     ↓
Feature Detection
     ↓
Feature Matching
     ↓
Camera Pose Estimation
     ↓
Monocular Depth Estimation
     ↓
Point Cloud Generation
     ↓
Multi-frame Fusion
     ↓
Surface Reconstruction
     ↓
3D Mesh
```

The implementation uses:
* **OpenCV / ORB** for feature detection and Lowe's ratio matching
* **recoverPose()** for relative camera motion and trajectory estimation
* **Depth Anything V2** for dense monocular relative depth estimation
* **Open3D** for multi-view point cloud fusion, statistical outlier filtering, and Poisson surface meshing

### Monocular Scale Notice
Monocular reconstruction has an inherent scale ambiguity: the baseline translation vector between views is normalized ($\|\mathbf{t}\| = 1.0$), and Depth Anything V2 estimates relative affine depth. All coordinates ($X, Y, Z$), point clouds, camera trajectories, and meshes are generated in arbitrary relative units unless external metric sensors or ground control points establish metric scale.

---

## The Six System Layers

The complete Minab system is organized across six major layers:

```mermaid
graph TD
    L1[01. Mission Layer<br>Coverage Planner • Waypoints • Route Objectives] --> L2[02. Aerial Data Layer<br>Video Frames • GPS • IMU • Telemetry Sync]
    L2 --> L3[03. Connectivity Layer<br>UAV Senses & Transmits • Ground PC Computes]
    L3 --> L4[04. Intelligence Layer<br>ORB Tracking • Depth Anything V2 • PyTorch]
    L4 --> L5[05. Spatial Layer<br>Multi-View Fusion • Poisson Mesh • Texture Projection]
    L5 --> L6[06. Autonomous Exploration<br>Coverage Analysis • Uncertainty Estimation • Next-Best-View]
    L6 -.->|Closed-Loop Replanning| L1
```

### 01. Mission Layer
The user defines:
* Exploration area
* Altitude
* Flight speed
* Camera configuration
* Desired overlap
* Route type
* Exploration objective

Minab converts this into an actionable flight mission:
```text
Exploration Area → Coverage Planner → Waypoints → Mission → UAV
```

### 02. Aerial Data Layer
The UAV collects:
* **Visual data:** Video, individual frames, timestamps
* **Flight data:** GPS, altitude, velocity, heading, attitude, battery, flight mode, mission state

Video and telemetry streams are timestamped and synchronized.

### 03. Connectivity Layer
The communication architecture maintains a clean separation of responsibilities:
```text
                    UAV
                     │
        ┌────────────┴────────────┐
        │                         │
     Camera                 Flight Controller
        │                         │
        ▼                       MAVLink
 Companion Computer               │
        │                         │
        ├──── Video ──────────────┤
        │                         │
        └──── Telemetry ──────────┘
                     │
                     ▼
                 Ground PC
                     │
                     ▼
                   Minab
```
The UAV primarily senses, flies, and transmits. The ground computer performs the computationally intensive machine learning and computer vision processing.

### 04. Intelligence Layer
The Python ML/CV core:
```text
Python
├── OpenCV
├── NumPy
├── SciPy
├── PyTorch
├── Depth Anything V2
├── Open3D
└── Computer Vision Models
```

Pipeline flow:
```text
ORB → BFMatcher → Essential Matrix → recoverPose() → Camera Trajectory → Depth Anything V2 → RGB-D Reconstruction → Open3D → Point Cloud → Mesh
```

### 05. Spatial Layer
Minab transforms raw observations into a spatial environment. The goal is not simply a point cloud, but a reconstructed spatial model:
```text
RGB Frames + Camera Poses + Depth → RGB-D Frames → Multi-view Fusion → Colored Point Cloud → Surface Reconstruction → Mesh → Texture / Color Projection → 3D Environment
```

Geometry and appearance remain conceptually decoupled:
1. Establish coherent spatial geometry.
2. Reconstruct and project visual appearance onto that geometry.

### 06. Autonomous Exploration
The closed-loop research component:
1. After reconstructing the environment, Minab evaluates:
   * What areas were observed
   * What areas were poorly observed
   * Where reconstruction confidence is low
   * Where geometry is incomplete
   * Where additional viewpoints are required
   * What route could improve the model
2. Decision cycle:
   ```text
   3D Model → Coverage Analysis → Uncertainty Estimation → Candidate Viewpoints → Next-Best-View Planning → New Mission → UAV
   ```
3. Cycle repeats: **Explore → Reconstruct → Understand → Decide → Explore Again.**

---

## Backend Architecture

Flask acts as the orchestration and API service, delegating heavy inference to worker processes:

```text
                    Flask
                      │
       ┌──────────────┼──────────────┐
       │              │              │
    Missions       Uploads        Results
       │              │              │
       └──────────────┼──────────────┘
                      │
                      ▼
                 ML Workers
                      │
             Python / PyTorch
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
      Pose          Depth      Reconstruction
```

The web interface serves as the mission, monitoring, and 3D visualization layer, while Python background workers execute the computational pipeline.

---

## Development Roadmap

```text
Phase 1 — Vision POC (Complete on branch 3d-module)
  Video → Pose → Depth → Point Cloud → Mesh

Phase 2 — Real Camera Validation
  Lens calibration, real-world motion, feature stability, color consistency

Phase 3 — UAV Integration
  UAV → Camera → Video → Ground PC → Minab → 3D Model

Phase 4 — Mission Planning
  User → Area → Altitude → Speed → Overlap → Route Planner → PX4 Mission → UAV

Phase 5 — Spatial Intelligence
  3D Model → Coverage → Uncertainty → Missing Regions

Phase 6 — Active Exploration
  Plan → Fly → Observe → Reconstruct → Analyze → Replan → Fly Again
```

---

## Visual Identity & Design Standards

The visual design system is locked to three primary values:

| Token | Hex | Application |
| :--- | :--- | :--- |
| **Black** | `#000000` | Primary canvas and viewports |
| **White** | `#FFFFFF` | Primary typography, borders, and structural marks |
| **Minab Green** | `#16A34A` | Identity accent, active states, scans, and trajectories |

No secondary blues, purples, cyans, decorative gradients, neon glow, or excessive shadows.

### Identity Symbol
The symbol-only mark encapsulates:
* **Mountain / M:** Physical environment and terrain
* **Contour Lines:** Spatial reconstruction and elevation
* **Flight Arc:** Aerial exploration
* **Node:** Autonomous UAV system
* **Viewfinder Brackets:** Computer vision and framing
* **Green:** Identity signal

### Interface Language
* Typography: Google Sans / Inter
* Thin, restrained borders
* Compact, structured cards
* Monochrome icons with green active indicators
* Aesthetic: Aerospace mission software meets scientific visualization

---

## Accessing the Working Code

The complete working 3D reconstruction codebase, Flask web application, Three.js studio viewer, and automated test suite are hosted on the **`3d-module`** branch.

### Switch to the Working Code

```powershell
git checkout 3d-module
```

### Quick Run

```powershell
# Setup virtual environment
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Start the Web Studio
python run.py
```

Open **`http://localhost:5000`** to access the 3D Reconstruction Studio.
