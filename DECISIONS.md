# Architecture Decision Records (ADR) — Saksham: AI Study Mentor

## ADR 001: Deep Knowledge Tracing (DKT) with Bayesian Cold-Start Fallback
- **Context:** Modeling knowledge mastery dynamics requires capturing sequential dependencies in learning trajectories.
- **Decision:** Use a single-layer LSTM (hidden dimension 64) trained on synthetic interaction sequences. For cold-start interactions ($N < 5$), fallback to a Beta-Binomial conjugate Bayesian updater to maintain calibration without overfitting short sequences.
- **Status:** Approved (v2 Specification).

## ADR 002: Graph Neural Networks for Prerequisite DAG Root-Cause Tracing
- **Context:** Static thresholds (e.g. `parent < 0.6`) fail when knowledge deficits are located multiple hops upstream in the curriculum graph.
- **Decision:** Construct a prerequisite DAG in PyTorch Geometric and apply graph convolutional/attention message passing to propagate uncertainty-weighted mastery beliefs. The lowest post-propagation belief ancestor is isolated as the root cause.
- **Status:** Approved (v2 Specification).

## ADR 003: EZ-Diffusion Closed-Form Estimation for Decision Dynamics
- **Context:** Full Drift-Diffusion Modeling (DDM) via MCMC is computationally prohibitive during real-time user sessions.
- **Decision:** Implement the Wagenmakers et al. (2007) EZ-diffusion closed-form estimator using mean accuracy ($P_c$), mean reaction time ($MRT$), and reaction time variance ($VRT$) over a rolling 15-trial window to estimate drift rate ($v$), boundary separation ($a$), and non-decision time ($T_{er}$). Fall back to $z$-score latency heuristics when $N < 15$.
- **Status:** Approved (v2 Specification).

## ADR 004: 2-State Hidden Markov Model for Cognitive Fatigue
- **Context:** Rule-based fatigue flags miss subtle probabilistic state transitions.
- **Decision:** Deploy a 2-state Gaussian HMM (`hmmlearn`) over rolling observation vectors `[accuracy, rt_z_score, hint_rate]`. Provide continuous $P(\text{fatigued})$ to the pedagogical planner.
- **Status:** Approved (v2 Specification).

## ADR 005: Depth-3 Monte Carlo Tree Rollout for POMDP Pedagogical Planning
- **Context:** Full POMDP point-based solvers introduce high implementation complexity and latency, while greedy 1-step heuristics fail to plan remediation sequences.
- **Decision:** Implement a depth-3 Monte Carlo rollout using the environment simulator (`simulator.py`) as the generative transition model, pruning expansion to top candidate actions to maintain sub-second inference.
- **Status:** Approved (v2 Specification).

## ADR 006: Offline Model Training Strategy
- **Context:** Training neural networks or fitting HMMs during live user requests would cause latency degradation.
- **Decision:** Execute synthetic corpus generation and model training (`train_*.py`) offline. Serialize weights to `models/` (`dkt.pt`, `gnn.pt`, `hmm.pkl`) and load them into memory at application boot.
- **Status:** Approved (v2 Specification).

## ADR 007: Continuous Latent State Estimation vs. Static Classification
- **Context:** Debunked "learning styles" (Pashler et al., 2009) or permanent clinical/archetype classifications are scientifically indefensible and vulnerable to criticism.
- **Decision:** Model the student as a continuous, time-varying probabilistic state vector $\mathbf{S}_t = [M_1..M_K, S_1..S_K, \tau, a, F_t, C, U]$ updated after every interaction, separating latent competence from observed performance.
- **Status:** Approved.

## ADR 008: Six-Category Learning-Science Error Taxonomy
- **Context:** Binary correctness flags discard critical diagnostic evidence needed for remediation.
- **Decision:** Classify errors into six psychometric categories: Knowledge Deficit, Retrieval Failure, Procedural Error, Careless Slip, Impulsive Error, and Cognitive Overload.
- **Status:** Approved.

## ADR 009: Item-Normalized Hierarchical Response-Time Modeling
- **Context:** Raw response times confound item complexity with student speed ("Slow $\neq$ Weak").
- **Decision:** Adopt van der Linden's hierarchical response time framework ($\log T_{ij} \sim \mathcal{N}(\lambda_j - \tau_i, \sigma^2)$) to normalize latency against question difficulty intensity.
- **Status:** Approved.
