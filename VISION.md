# Saksham — AI Study Mentor
## Architectural & Product Vision Document (v2 Specification)

---

## 1. Executive Summary & Mission

**Saksham** (सक्षम — *capable, empowered*) is an AI study mentor engineered for educational equity. Most conventional adaptive learning systems rely on shallow heuristic rules or generic chatbot responses that cannot accurately infer a learner's latent cognitive state. 

Saksham bridges psychometrics, cognitive science, and deep learning to construct an authentic mental model of each student. By combining **Deep Knowledge Tracing (DKT)**, **Graph Neural Networks (GNN)** on prerequisite skill graphs, **Drift-Diffusion Models (DDM)** of decision dynamics, **Hidden Markov Models (HMM)** for fatigue tracking, and **POMDP-style lookahead planning**, Saksham acts as an empathetic, transparent, and mathematically rigorous study mentor.

### Core Philosophy
1. **Cognitive Fidelity:** Model knowledge mastery, evidence-accumulation speed, caution, and cognitive fatigue separately rather than collapsing everything into a single score.
2. **Root-Cause Remediation:** When a student struggles with an advanced concept, identify and remediate the fundamental prerequisite bottleneck upstream in the knowledge graph.
3. **Multi-Step Pedagogical Foresight:** Plan tutoring actions several steps ahead to maximize long-term retention and engagement, rather than responding greedily to the single latest answer.
4. **Radical Transparency ("Why this action?"):** Provide inspectable explanations for every pedagogical intervention so educators and learners understand why a specific question, hint, or rest break was recommended.

---

## 2. Theoretical & Cognitive Modeling Stack

Saksham's architecture replaces ad-hoc heuristics with established cognitive models and machine learning formulations:

```
+-----------------------------------------------------------------------------------+
|                                 Student Interaction                                |
|             (Question response, reaction time, hint requests, confidence)         |
+-----------------------------------------------------------------------------------+
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
      ┌──────────────────────┐┌──────────────────────┐┌──────────────────────┐
      │   Mastery Tracking   ││  Decision Dynamics   ││  Cognitive Fatigue   │
      │  (DKT LSTM + Bayes)  ││    (EZ-Diffusion)    ││    (2-State HMM)     │
      └──────────────────────┘└──────────────────────┘└──────────────────────┘
                 │                       │                       │
                 ▼                       │                       │
      ┌──────────────────────┐           │                       │
      │ Graph Propagation    │           │                       │
      │ (GNN Prereq DAG)     │           │                       │
      └──────────────────────┘           │                       │
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         ▼
                 +───────────────────────────────────────────+
                 |            Latent Belief State            |
                 |  b_t = [Mastery, Uncertainty, v, a,       |
                 |         Ter, P(fatigued), Memory]         |
                 +───────────────────────────────────────────+
                                         │
                                         ▼
                 +───────────────────────────────────────────+
                 |        POMDP Pedagogical Planner          |
                 |      (Depth-3 Monte Carlo Rollout)        |
                 +───────────────────────────────────────────+
                                         │
                                         ▼
                 +───────────────────────────────────────────+
                 |     Socratic LLM Mentor & UI Delivery     |
                 |   (Contextualized Hint / Question / Break)|
                 +───────────────────────────────────────────+
```

### 2.1. Knowledge Mastery: Deep Knowledge Tracing (DKT) & Cold-Start Bayesian Filtering
- **Deep Knowledge Tracing (`mastery_dkt.py`):** Single-layer LSTM (hidden dimension: 64). At interaction step $t$, the input is a one-hot encoding of $(k_t, r_t)$ where $k_t$ is the skill index and $r_t \in \{0, 1\}$ indicates correctness (input dimension: $2 \times N_{\text{skills}}$). The network outputs a sigmoid vector representing predicted $P(\text{correct}_{t+1})$ across all skills.
- **Cold-Start Bayesian Fallback (`mastery.py`):** For new learners with fewer than 5 recorded interactions, a Beta-Binomial conjugate Bayesian updater estimates mastery mean and variance without overfitting to short interaction sequences. Once $\text{len}(\text{history}) \ge 5$, inference hands off smoothly to the DKT LSTM.

### 2.2. Prerequisite Skill Propagation: Graph Neural Networks (GNN)
- **Skill DAG (`skill_graph.py` & `skill_gnn.py`):** Skills are nodes in a directed acyclic graph where edges represent prerequisite relationships ($\text{prereq} \to \text{target}$).
- **Propagation Layer:** 2-layer Graph Convolutional Network (`GCNConv`) or Graph Attention Network (`GATConv`) via PyTorch Geometric. Node feature vectors encode $[\mu_{\text{mastery}}, \sigma^2_{\text{mastery}}]$.
- **Root-Cause Discovery:** When performance falters on downstream skills, message passing propagates uncertainty-weighted belief upstream. The prerequisite ancestor with the lowest post-propagation belief is isolated as the genuine knowledge gap, eliminating naive static thresholding.

### 2.3. Latency & Caution: EZ-Diffusion Model (DDM)
- **Cognitive Dynamics (`ddm_ez.py`):** Instead of raw response times or crude $z$-scores, Saksham implements the closed-form EZ-diffusion model (Wagenmakers, van der Maas & Grasman, 2007) across rolling windows ($\ge 15$ trials):
  - **Drift Rate ($v$):** Rate of evidence accumulation, measuring true fluency and cognitive speed independent of caution.
  - **Boundary Separation ($a$):** Distance between decision thresholds, measuring learner caution versus impulsivity.
  - **Non-Decision Time ($T_{\text{er}}$):** Latency attributed to reading comprehension and motor execution (typing/clicking), separate from internal reasoning.
- **Cold-Start Fallback (`speed_caution.py`):** Standardized $z$-score latency heuristic operates during initial interactions ($< 15$ trials) until sufficient sample variance is accumulated.

### 2.4. Mental State: 2-State Hidden Markov Model (HMM)
- **Fatigue & Engagement Estimation (`attention_hmm.py`):** Discrete latent state $S_t \in \{\text{engaged}, \text{fatigued}\}$.
- **Observation Space:** Rolling vectors of $[\text{accuracy}, z_{\text{RT}}, \text{hint\_rate}]$ governed by multivariate Gaussian emission distributions (`hmmlearn.hmm.GaussianHMM`).
- **Belief Update:** Generates continuous posterior probability $P(S_t = \text{fatigued} \mid O_{1:t})$ fed directly into the pedagogical planner to modulate challenge difficulty and suggest micro-breaks.

### 2.5. Pedagogical Planning: POMDP Lookahead Rollout
- **Belief Formulation (`policy_pomdp.py`):** 
  $$\mathbf{b}_t = \left[ \boldsymbol{\mu}_{\text{mastery}}, \boldsymbol{\sigma}^2_{\text{mastery}}, v, a, T_{\text{er}}, P(\text{fatigued}), \text{calibration\_error}, \mathbf{m}_{\text{stability}} \right]$$
- **Planning Horizon:** Depth-3 Monte Carlo tree rollout using the synthetic simulator (`simulator.py`) as the generative world model.
- **Action Selection:** Rather than a greedy utility maximum, candidate primary actions ($a_1$) are evaluated by rolling forward simulated learner transitions across depth 3 and scoring resulting terminal belief states against balanced utility (learning gain, cognitive load reduction, engagement preservation). The top branching factor is constrained to ensure responsive sub-second execution.

### 2.6. Socratic Guidance: Belief-Grounded LLM Generation
- **Mentor Prompts (`llm_generator.py`):** The LLM receives structured JSON conditioning:
  - Identified root-cause concept and current confidence interval.
  - Cognitive caution profile (e.g., impulsive vs. overly hesitant).
  - Attention/fatigue state.
- **Output:** Socratic scaffolding questions, targeted conceptual analogies, or restorative check-ins without revealing full solutions prematurely.

---

## 3. System Architecture & Directory Layout

```
Saksham Edu4Good/
├── VISION.md                         # This architecture and vision document
├── DECISIONS.md                      # Key architectural & engineering choices
├── requirements.txt                  # Dependency specifications (PyTorch, PyG, hmmlearn, etc.)
├── questions.json                    # Structured question bank with prerequisite DAG annotations
├── app.py                            # Interactive UI (Streamlit / Modern Web App)
│
├── aile/                             # Core Cognitive Engine
│   ├── __init__.py
│   ├── db.py                         # SQLite schema for interactions, sessions, and telemetry
│   ├── skill_graph.py                # Prerequisite DAG constructor & PyG Data converter
│   ├── skill_gnn.py                  # PyTorch Geometric GNN for prerequisite belief propagation
│   ├── mastery.py                    # Dispatcher: Bayesian Beta-Binomial cold start -> DKT
│   ├── mastery_dkt.py                # Deep Knowledge Tracing LSTM architecture
│   ├── speed_caution.py              # Dispatcher: Z-score fallback -> EZ-diffusion
│   ├── ddm_ez.py                     # Closed-form EZ-Diffusion Model equations
│   ├── attention_hmm.py              # 2-state Gaussian HMM for fatigue inference
│   ├── memory.py                     # Spaced repetition & retention decay model
│   ├── calibration.py                # Metacognitive calibration (confidence vs. accuracy)
│   ├── policy.py                     # Base pedagogical utility functions
│   ├── policy_pomdp.py               # Depth-3 Monte Carlo tree rollout planner
│   ├── simulator.py                  # Synthetic student population & environment transition model
│   └── llm_generator.py              # Socratic prompt synthesizer (Claude/Gemini/OpenAI)
│
├── training/                         # Offline Training Pipeline
│   ├── generate_corpus.py            # Synthesizes thousands of student trajectories
│   ├── train_dkt.py                  # Trains DKT LSTM on synthetic sequence corpus
│   ├── train_gnn.py                  # Trains GNN on DAG propagation scenarios
│   └── train_hmm.py                  # Fits 2-state HMM on fatigue-injected sessions
│
├── models/                           # Serialized Pre-Trained Weights
│   ├── dkt.pt                        # Exported PyTorch DKT weights
│   ├── gnn.pt                        # Exported PyTorch Geometric GNN weights
│   └── hmm.pkl                       # Serialized hmmlearn model artifact
│
└── notebooks/                        # Verification, Calibration & Diagnostics
    ├── 01_calibration.ipynb          # Metacognitive calibration & psychometric evaluation
    └── 02_dkt_training_diagnostics.ipynb # Loss curves, ROC-AUC, and sequence validation
```

---

## 4. User Experience & Product Interfaces

Saksham delivers an intuitive, empowering interface with three core operational views:

### View 1: Active Study Session (The Student Workspace)
- **Clean Socratic Interface:** Distraction-free question presentation with adaptive difficulty.
- **Cognitive-Aware Scaffolding:** Context-sensitive hints, conceptual breakdowns, and pacing recommendations.
- **Metacognitive Reflection:** Low-friction confidence prompts before answering to calibrate self-efficacy.

### View 2: Cognitive Telemetry & Mastery Graph (The Knowledge Map)
- **Interactive Prerequisite DAG:** Visual representation of the concept graph. Nodes reflect real-time mastery beliefs; edges represent curriculum dependencies.
- **Bottleneck Highlighting:** GNN-identified prerequisite gaps highlighted with distinct visual indicators.
- **Decision Dynamics Readout:** Clear visualization of cognitive drift ($v$), decision caution ($a$), and sensory latency ($T_{\text{er}}$).

### View 3: Pedagogical Transparency ("Why this action?")
- **POMDP Decision Tree Viewer:** Interactive breakdown of candidate actions evaluated at depth 3.
- **Utility Breakdown:** Explicit display of estimated learning yield, cognitive load mitigation, and fatigue risk factors leading to the selected intervention.

---

## 5. Offline Training & Validation Strategy

To ensure deterministic reliability and sub-second latency during live study sessions, heavy machine learning modules are pre-trained offline:

1. **Synthetic Student Synthesis (`generate_corpus.py`):** Simulate heterogeneous student profiles (varying prior mastery, learning rates, guess/slip factors, and fatigue onset points).
2. **Supervised DKT Optimization (`train_dkt.py`):** Train next-step binary cross-entropy loss across synthetic trajectories until convergence; validate with AUC and calibration plots.
3. **Graph Attention Training (`train_gnn.py`):** Supervise prerequisite propagation against known ground-truth gap structures.
4. **Fatigue Emission Modeling (`train_hmm.py`):** Fit Gaussian emission parameters on sessions with programmatically injected fatigue windows.
5. **Runtime Pre-loading:** At application initialization, pre-trained weights in `models/` are loaded into memory, ensuring zero training overhead during interactive mentoring.

---

## 6. Roadmap & Implementation Phases

| Phase | Focus Areas | Deliverables |
|---|---|---|
| **Phase 1: Foundation** | Data schemas, simulator, question bank, and cold-start baselines | `db.py`, `questions.json`, `simulator.py`, Bayesian mastery, $z$-score speed |
| **Phase 2: Cognitive Algorithms** | Psychometric and machine learning core modules | `ddm_ez.py`, `mastery_dkt.py`, `skill_gnn.py`, `attention_hmm.py` |
| **Phase 3: Synthetic Pipeline** | High-volume simulation and offline model training | `generate_corpus.py`, `train_dkt.py`, `train_gnn.py`, `train_hmm.py`, weights in `models/` |
| **Phase 4: Planning & Socratic LLM** | Multi-step lookahead and natural language mentoring | `policy_pomdp.py`, `llm_generator.py` |
| **Phase 5: Interface & Diagnostics** | Student workspace, inspectable telemetry, and diagnostic notebooks | `app.py`, `01_calibration.ipynb`, `02_dkt_training_diagnostics.ipynb` |
| **Phase 6: Verification & Polish** | End-to-end benchmarking, latency profiling, and packaging | Unit tests, test student runs, `DECISIONS.md`, documentation |
