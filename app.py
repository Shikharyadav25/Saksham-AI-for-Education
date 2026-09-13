import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from aile.attention_hmm import FatigueHMM
from aile.calibration import compute_metacognitive_calibration
from aile.db import (
    create_learner,
    delete_learner,
    ensure_learner,
    get_all_learners,
    get_learner,
    get_learner_interactions,
    init_db,
    record_interaction,
    save_cognitive_state,
    seed_sample_learners_if_empty,
    start_session,
)
from aile.ddm_ez import compute_ez_diffusion
from aile.error_taxonomy import classify_interaction_error
from aile.llm_generator import SocraticMentorGenerator
from aile.memory import MemoryModel
from aile.mastery import MasteryEngine
from aile.policy_pomdp import POMDPRolloutPlanner
from aile.simulator import StudentSimulator
from aile.skill_gnn import SkillGNNPredictor
from aile.skill_graph import SkillGraph
from aile.speed_caution import analyze_pacing_and_caution

# Initialize database and populate authentic benchmark profiles if empty
init_db()
seed_sample_learners_if_empty()

# Page configuration
st.set_page_config(
    page_title="Sarvagya — AI Study Mentor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom dark design system stylesheet
css_path = Path(__file__).resolve().parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



@st.cache_resource
def load_core_resources():
    skill_graph = SkillGraph()
    dkt_weights = Path(__file__).resolve().parent / "models" / "dkt.pt"
    gnn_weights = Path(__file__).resolve().parent / "models" / "gnn.pt"
    hmm_weights = Path(__file__).resolve().parent / "models" / "hmm.pkl"

    gnn_predictor = SkillGNNPredictor(weights_path=gnn_weights)
    fatigue_hmm = FatigueHMM(model_path=hmm_weights)
    simulator = StudentSimulator(skill_graph)
    planner = POMDPRolloutPlanner(skill_graph, simulator)
    llm = SocraticMentorGenerator()

    # Load questions
    with open(Path(__file__).resolve().parent / "questions.json", "r", encoding="utf-8") as f:
        q_data = json.load(f)

    return {
        "graph": skill_graph,
        "gnn": gnn_predictor,
        "hmm": fatigue_hmm,
        "simulator": simulator,
        "planner": planner,
        "llm": llm,
        "questions": q_data["questions"],
    }


res = load_core_resources()
graph: SkillGraph = res["graph"]
questions: List[Dict[str, Any]] = res["questions"]

# Fetch learners dynamically from SQLite (Alice demo profile + user-created profiles)
all_learners = get_all_learners()
learner_ids = [l["id"] for l in all_learners]

# Ensure active learner exists in DB list
if "learner_id" not in st.session_state or st.session_state["learner_id"] not in learner_ids:
    st.session_state["learner_id"] = learner_ids[0] if learner_ids else "alice_prereq_gap"

if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"sess_{st.session_state['learner_id']}_{int(time.time())}"

if "current_question_idx" not in st.session_state:
    st.session_state["current_question_idx"] = 0
if "question_start_time" not in st.session_state:
    st.session_state["question_start_time"] = time.time()
if "last_feedback" not in st.session_state:
    st.session_state["last_feedback"] = None
if "hint_requested" not in st.session_state:
    st.session_state["hint_requested"] = False
if "timer_running" not in st.session_state:
    st.session_state["timer_running"] = False
if "timer_start_time" not in st.session_state:
    st.session_state["timer_start_time"] = None
if "timer_last_recorded" not in st.session_state:
    st.session_state["timer_last_recorded"] = None

current_learner = get_learner(st.session_state["learner_id"])

# Sidebar Controls & Dynamic Learner Management
with st.sidebar:
    st.markdown("### 👤 Active Learner Profile")

    def format_learner(lid: str) -> str:
        for l in all_learners:
            if l["id"] == lid:
                age_part = f"{l['age']} yrs" if l.get("age") else "Age N/A"
                gender_part = l.get("gender") or "Gender N/A"
                return f"{l['name']} • {age_part}, {gender_part}"
        return lid

    curr_idx = learner_ids.index(st.session_state["learner_id"]) if st.session_state["learner_id"] in learner_ids else 0
    selected_learner = st.selectbox(
        "Select Active Learner Profile:",
        options=learner_ids,
        format_func=format_learner,
        index=curr_idx,
    )
    if selected_learner != st.session_state["learner_id"]:
        st.session_state["learner_id"] = selected_learner
        st.session_state["session_id"] = f"sess_{selected_learner}_{int(time.time())}"
        st.session_state["last_feedback"] = None
        st.session_state["hint_requested"] = False
        st.session_state["current_question_idx"] = 0
        st.session_state["question_start_time"] = time.time()
        st.rerun()

    # Active profile summary card
    if current_learner:
        history_cnt = len(get_learner_interactions(current_learner["id"]))
        is_demo = current_learner["id"] == "alice_prereq_gap"
        badge_label = "Demo Reference" if is_demo else "Custom Learner"
        badge_color = "#60A5FA" if is_demo else "#34D399"
        badge_bg = "rgba(59, 130, 246, 0.15)" if is_demo else "rgba(16, 185, 129, 0.15)"
        badge_border = "rgba(59, 130, 246, 0.3)" if is_demo else "rgba(16, 185, 129, 0.3)"
        st.markdown(
            f"""
            <div class="profile-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="profile-name">{current_learner['name']}</span>
                    <span class="profile-badge" style="background:{badge_bg}; color:{badge_color}; border:1px solid {badge_border};">{badge_label}</span>
                </div>
                <div class="profile-meta">
                    Age: <b>{current_learner.get('age', 'N/A')}</b> &nbsp;|&nbsp; 
                    Gender: <b>{current_learner.get('gender', 'N/A')}</b>
                </div>
                <div class="profile-meta">
                    Interactions Logged: <b>{history_cnt}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Form to add user profiles manually multiple times
    with st.expander("➕ Add New Learner Profile", expanded=(len(all_learners) <= 1)):
        st.caption("Create as many learners as needed. All records are saved to SQLite.")
        with st.form(key="create_learner_form", clear_on_submit=True):
            new_name = st.text_input("Learner Full Name*", placeholder="e.g. Arjun Patel")
            col_age, col_gen = st.columns(2)
            with col_age:
                new_age = st.number_input("Age*", min_value=5, max_value=99, value=16, step=1)
            with col_gen:
                new_gender = st.selectbox("Gender*", ["Female", "Male", "Non-Binary", "Other", "Prefer not to say"])

            submitted = st.form_submit_button("Save & Switch to Profile", type="primary", use_container_width=True)
            if submitted:
                if not new_name or not new_name.strip():
                    st.error("Please enter a valid student name.")
                else:
                    new_id = create_learner(name=new_name.strip(), age=int(new_age), gender=new_gender)
                    st.session_state["learner_id"] = new_id
                    st.session_state["session_id"] = f"sess_{new_id}_{int(time.time())}"
                    st.session_state["last_feedback"] = None
                    st.session_state["hint_requested"] = False
                    st.session_state["current_question_idx"] = 0
                    st.session_state["question_start_time"] = time.time()
                    st.success(f"Learner '{new_name.strip()}' created!")
                    st.rerun()

    # Optional deletion of custom learner
    if current_learner and current_learner["id"] != "alice_prereq_gap":
        if st.button("🗑️ Delete Selected Learner", help="Permanently deletes this learner and their interaction history", use_container_width=True):
            delete_learner(current_learner["id"])
            st.session_state["learner_id"] = "alice_prereq_gap"
            st.session_state["session_id"] = f"sess_alice_{int(time.time())}"
            st.session_state["last_feedback"] = None
            st.session_state["hint_requested"] = False
            st.session_state["current_question_idx"] = 0
            st.session_state["question_start_time"] = time.time()
            st.rerun()

    st.markdown("---")
    st.markdown("### Model Architecture Status")
    dkt_loaded = Path(__file__).resolve().parent / "models" / "dkt.pt"
    gnn_loaded = res["gnn"].is_loaded
    hmm_loaded = res["hmm"].is_loaded

    st.markdown(f"**DKT LSTM:** `{'✓ Trained & Loaded' if dkt_loaded.exists() else 'Bayesian Fallback'}`")
    st.markdown(f"**Prereq GNN:** `{'✓ Trained & Loaded' if gnn_loaded else 'Heuristic Fallback'}`")
    st.markdown(f"**Fatigue HMM:** `{'✓ Fitted (2-State)' if hmm_loaded else 'Prior Initialized'}`")
    st.markdown(f"**Question Bank:** `{len(questions)} Calibrated Items`")
    st.markdown(f"**Learners in DB:** `{len(all_learners)} Profiles`")

# Main Header
st.markdown(
    """
    <div class="brand-header">
        <h1 class="brand-title">🎓 Sarvagya — AI Study Mentor</h1>
        <p class="brand-subtitle">Empowering learners through Deep Knowledge Tracing, EZ-Diffusion, Prerequisite GNN, and POMDP multi-step lookahead.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Interaction History for Current Learner from SQLite
history = get_learner_interactions(st.session_state["learner_id"])
accuracies = [h["is_correct"] for h in history]
rts = [h["response_time"] for h in history]
confs = [h["confidence_rating"] for h in history if h["confidence_rating"] is not None]
conf_accs = [h["is_correct"] for h in history if h["confidence_rating"] is not None]

# Rebuild Mastery and Memory State dynamically from DB interactions (Zero Mock Data)
dkt_path = Path(__file__).resolve().parent / "models" / "dkt.pt"
mastery_engine = MasteryEngine(skill_graph=graph, dkt_weights_path=dkt_path)
memory_model = MemoryModel()
for h in history:
    mastery_engine.register_interaction(h["skill_id"], bool(h["is_correct"]))
    memory_model.update(h["skill_id"], bool(h["is_correct"]))

mastery_est = mastery_engine.get_mastery_estimate()
current_mastery_means = mastery_est.means
current_mastery_vars = mastery_est.variances

# Analyze Speed & Caution via EZ-Diffusion / z-score
pacing_profile = analyze_pacing_and_caution(accuracies, rts)

# Analyze Fatigue via 2-State Gaussian HMM
if len(history) >= 5:
    recent_window = history[-5:]
    w_acc = float(np.mean([x["is_correct"] for x in recent_window]))
    w_rt = float(np.mean([x["response_time"] for x in recent_window]))
    w_hints = float(np.mean([x["hint_requested"] for x in recent_window]))
    w_rt_z = (w_rt - 12.0) / 5.0
    p_fatigued = res["hmm"].predict_fatigue_probability(np.array([[w_acc, w_rt_z, w_hints]]))
else:
    p_fatigued = 0.10

# Analyze Memory & Decayed Skills
decayed_skills = [s for s in graph.skill_ids if memory_model.get_review_urgency(s) > 0.35]

# Metacognitive Calibration
calib_profile = compute_metacognitive_calibration(confs, conf_accs)

# Tabs
tab_study, tab_telemetry, tab_policy, tab_simulation = st.tabs([
    "📖 Active Study Session",
    "📊 Cognitive Telemetry",
    "🔍 Pedagogical Transparency",
    "🧪 Simulator & Benchmarking",
])

# ---------------------------------------------------------
# TAB 1: ACTIVE STUDY SESSION
# ---------------------------------------------------------
with tab_study:
    # ---------------------------------------------------------
    # Study Session Scope & Focus (Subject & Topic Selection)
    # ---------------------------------------------------------
    subjects_list = ["All Subjects (Adaptive Track)"] + graph.get_subjects()
    if "selected_subject" not in st.session_state or st.session_state["selected_subject"] not in subjects_list:
        st.session_state["selected_subject"] = subjects_list[0]

    col_subj_sel, col_top_sel = st.columns(2)
    with col_subj_sel:
        chosen_subject = st.selectbox(
            "📚 Choose Study Subject:",
            options=subjects_list,
            index=subjects_list.index(st.session_state["selected_subject"]),
            key="session_subject_picker",
            help="Select a core subject domain to focus this study session",
        )

    # Determine available topics for chosen subject
    if chosen_subject == "All Subjects (Adaptive Track)":
        subject_skills = [graph.skill_metadata[sid] for sid in graph.skill_ids]
    else:
        subject_skills = graph.get_skills_by_subject(chosen_subject)

    topic_keys = ["all"] + [s["id"] for s in subject_skills]

    def format_topic_option(tid: str) -> str:
        if tid == "all":
            return "🎯 All Topics (Comprehensive Practice)"
        return f"🎯 {graph.skill_metadata.get(tid, {}).get('name', tid)}"

    if "selected_topic" not in st.session_state or st.session_state["selected_topic"] not in topic_keys:
        st.session_state["selected_topic"] = topic_keys[0]

    with col_top_sel:
        chosen_topic = st.selectbox(
            "🎯 Choose Topic to Test Upon:",
            options=topic_keys,
            format_func=format_topic_option,
            index=topic_keys.index(st.session_state["selected_topic"]) if st.session_state["selected_topic"] in topic_keys else 0,
            key="session_topic_picker",
            help="Select a specific concept or topic to be tested on",
        )

    # React to user changes by resetting session item pointer and timers
    if chosen_subject != st.session_state["selected_subject"] or chosen_topic != st.session_state["selected_topic"]:
        st.session_state["selected_subject"] = chosen_subject
        st.session_state["selected_topic"] = chosen_topic
        st.session_state["current_question_idx"] = 0
        st.session_state["timer_running"] = False
        st.session_state["timer_start_time"] = None
        st.session_state["hint_requested"] = False
        st.session_state["last_feedback"] = None
        st.rerun()

    # Filter calibrated questions pool to match student selection
    if chosen_topic != "all":
        session_questions = [q for q in questions if q["skill_id"] == chosen_topic]
    elif chosen_subject != "All Subjects (Adaptive Track)":
        sub_sids = {s["id"] for s in subject_skills}
        session_questions = [q for q in questions if q["skill_id"] in sub_sids]
    else:
        session_questions = questions

    if not session_questions:
        session_questions = questions

    q_idx = st.session_state["current_question_idx"] % len(session_questions)
    curr_q = session_questions[q_idx]
    skill_id = curr_q["skill_id"]
    skill_name = graph.skill_metadata.get(skill_id, {}).get("name", skill_id)
    skill_subject = graph.skill_metadata.get(skill_id, {}).get("subject", "General Mathematics")

    col_q, col_sidebar_info = st.columns([7, 3])

    with col_q:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
                <span style="font-size:13px; color:var(--accent-blue); font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">
                    {skill_subject} &nbsp;•&nbsp; {skill_name}
                </span>
                <span style="font-size:12px; color:var(--text-muted); background:rgba(255,255,255,0.06); padding:2px 8px; border-radius:4px;">
                    Question #{q_idx + 1} of {len(session_questions)}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="question-card">
                <span class="question-number">Question #{q_idx + 1}</span>
                <p class="question-text">{curr_q['question_text']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Timer Control Bar
        timer_is_active = st.session_state.get("timer_running", False)

        t_col_status, t_col_btn = st.columns([7, 3])
        with t_col_status:
            if timer_is_active:
                st.markdown(
                    """
                    <div class="timer-card timer-card-active">
                        <span style="font-size: 20px;">⏱️</span>
                        <div>
                            <div class="timer-title-active">TIMER RUNNING • DELIBERATION ACTIVE</div>
                            <div class="timer-desc-active">Clock is actively counting. Solve the problem and submit below to stop the timer.</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="timer-card timer-card-idle">
                        <span style="font-size: 20px;">⏸️</span>
                        <div>
                            <div class="timer-title-idle">TIMER STOPPED / NOT STARTED</div>
                            <div class="timer-desc-idle">Press <b>Turn On Timer</b> when ready to begin solving for accurate latency tracking.</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with t_col_btn:
            if not timer_is_active:
                if st.button("▶️ Turn On Timer", type="primary", use_container_width=True, help="Starts measuring deliberation latency for this question"):
                    st.session_state["timer_running"] = True
                    st.session_state["timer_start_time"] = time.time()
                    st.rerun()
            else:
                if st.button("⏹️ Reset Timer", type="secondary", use_container_width=True, help="Reset timer if you were interrupted"):
                    st.session_state["timer_running"] = False
                    st.session_state["timer_start_time"] = None
                    st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # Options Form
        with st.form("study_answer_form"):
            user_choice = st.radio(
                "Select your solution:",
                options=["A", "B", "C", "D"],
                format_func=lambda opt: f"({opt})  {curr_q['options'][opt]}",
            )

            col_conf, col_btn = st.columns([6, 4])
            with col_conf:
                confidence = st.slider(
                    "Self-assessment: How confident are you in this answer?",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.75,
                    step=0.05,
                    format="%.0f%%",
                )

            with col_btn:
                st.write("")
                submit_label = "Submit Answer & Stop Timer" if timer_is_active else "Submit Answer (Timer Required)"
                submit_btn = st.form_submit_button(
                    submit_label,
                    type="primary",
                    disabled=not timer_is_active,
                    use_container_width=True,
                    help="Submits choice and stops the active timer" if timer_is_active else "Please turn on timer above to begin solving",
                )

        if not timer_is_active:
            st.caption("🔒 *Press **▶️ Turn On Timer** above to unlock submission. This ensures your response time is measured with high precision.*")

        if submit_btn and timer_is_active:
            raw_latency = time.time() - st.session_state["timer_start_time"]
            elapsed_rt = round(max(0.5, raw_latency), 2)

            # Stop timer immediately upon submission
            st.session_state["timer_running"] = False
            st.session_state["timer_start_time"] = None
            st.session_state["timer_last_recorded"] = elapsed_rt

            is_correct = (user_choice == curr_q["correct_answer"])

            # Record in SQLite DB with exact stopped response time
            record_interaction(
                session_id=st.session_state["session_id"],
                learner_id=st.session_state["learner_id"],
                question_id=curr_q["id"],
                skill_id=skill_id,
                is_correct=is_correct,
                response_time=elapsed_rt,
                hint_requested=st.session_state["hint_requested"],
                confidence_rating=confidence,
            )

            # Update engine state
            mastery_engine.register_interaction(skill_id, is_correct)
            memory_model.update(skill_id, is_correct)

            # Distractor feedback and scientific error classification
            rationale = curr_q["distractor_rationales"].get(user_choice, "") if not is_correct else ""
            err_diag = None
            if not is_correct:
                expected_int = 6.0 + 20.0 * float(curr_q.get("difficulty", 0.5))
                _, rc_temp, _ = res["gnn"].propagate_and_trace_root_cause(
                    graph, current_mastery_means, current_mastery_vars, struggling_skill=skill_id
                )
                err_diag = classify_interaction_error(
                    is_correct=False,
                    skill_id=skill_id,
                    prior_mastery=current_mastery_means.get(skill_id, 0.5),
                    retrievability=memory_model.compute_retrievability(skill_id),
                    response_time=elapsed_rt,
                    expected_intensity_sec=expected_int,
                    confidence_rating=confidence,
                    hint_requested=st.session_state["hint_requested"],
                    p_fatigued=p_fatigued,
                    distractor_rationale=rationale,
                    root_cause_prereq=rc_temp,
                )

            st.session_state["last_feedback"] = {
                "is_correct": is_correct,
                "rt": elapsed_rt,
                "user_choice": user_choice,
                "correct_answer": curr_q["correct_answer"],
                "rationale": rationale,
                "error_diagnosis": err_diag,
            }

            # Advance to next question with timer idle
            st.session_state["current_question_idx"] += 1
            st.session_state["hint_requested"] = False
            st.rerun()

        # Display last feedback if present
        if st.session_state["last_feedback"]:
            fb = st.session_state["last_feedback"]
            if fb["is_correct"]:
                st.success(f"✓ Correct! Accurately timed at {fb['rt']:.2f}s (Timer stopped and logged).")
            else:
                st.error(f"✗ Incorrect. Correct answer was ({fb['correct_answer']}). Latency: {fb['rt']:.2f}s (Timer stopped and logged).")
                if fb.get("error_diagnosis"):
                    diag = fb["error_diagnosis"]
                    st.warning(
                        f"🔍 **Error Diagnosis: {diag.title}** (`{diag.error_type}`)\n\n"
                        f"{diag.description}\n\n"
                        f"👉 **Pedagogical Prescription:** {diag.recommended_action}"
                    )
                if fb["rationale"]:
                    st.info(f"**Misconception Analysis:** {fb['rationale']}")

        # Socratic Hint Drawer
        with st.expander("💡 Need a Socratic Hint?", expanded=st.session_state["hint_requested"]):
            if not st.session_state["hint_requested"]:
                if st.button("Request Socratic Guidance"):
                    st.session_state["hint_requested"] = True
                    st.rerun()

            if st.session_state["hint_requested"]:
                # Check root cause with GNN
                _, rc_skill, _ = res["gnn"].propagate_and_trace_root_cause(
                    graph, current_mastery_means, current_mastery_vars, struggling_skill=skill_id
                )
                guidance = res["llm"].generate_guidance(
                    question_text=curr_q["question_text"],
                    skill_id=skill_id,
                    root_cause_skill=rc_skill,
                    caution_label=pacing_profile.caution_label,
                    fluency_label=pacing_profile.fluency_label,
                    p_fatigued=p_fatigued,
                    calibration_bias=calib_profile.bias_label,
                )
                st.markdown(f"**Mentor Hint:** {guidance.hint_text}")
                st.markdown(f"**Reflection:** *{guidance.reflection_question}*")
                if guidance.pacing_advice:
                    st.caption(f"⏱️ {guidance.pacing_advice}")

    with col_sidebar_info:
        st.markdown("#### Live Session Status")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Active Subject</div>
                <div class="metric-val" style="font-size:16px; line-height:1.3;">{skill_subject}</div>
                <div class="metric-sub">Focus: {skill_name}</div>
            </div>
            <div class="metric-card" style="margin-top: 12px;">
                <div class="metric-label">Mastery Estimate ({mastery_est.active_estimator})</div>
                <div class="metric-val">{current_mastery_means.get(skill_id, 0.5):.0%}</div>
                <div class="metric-sub">Prerequisites: {len(graph.get_prerequisites(skill_id))}</div>
            </div>
            <div class="metric-card" style="margin-top: 12px;">
                <div class="metric-label">Pacing Profile</div>
                <div class="metric-val" style="text-transform: capitalize;">{pacing_profile.caution_label}</div>
                <div class="metric-sub">Fluency: {pacing_profile.fluency_label.replace('_', ' ')}</div>
            </div>
            <div class="metric-card" style="margin-top: 12px;">
                <div class="metric-label">Fatigue Risk</div>
                <div class="metric-val">{p_fatigued:.0%}</div>
                <div class="metric-sub">HMM 2-State Posterior</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# TAB 2: COGNITIVE TELEMETRY
# ---------------------------------------------------------
with tab_telemetry:
    st.markdown("### Latent Cognitive State Telemetry")
    st.markdown(
        """
        <div class="telemetry-banner">
            <div class="telemetry-banner-title">
                Continuous Latent Learner State: S_t = [M_1..M_K, S_1..S_K, &tau;, a, F_t, C, U]
            </div>
            <p class="telemetry-banner-desc">
                <b>Core Principle:</b> <i>"A student's score is an observation; it is not the student's state."</i> Sarvagya does not assign students to fixed "learning styles" or permanent labels. It continuously updates a probabilistic mental model across mastery, memory decay, hierarchical processing speed (&tau;), decision caution (a), fatigue (F_t), and metacognitive calibration (C).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mastery Model ({mastery_est.active_estimator})</div>
                <div class="metric-val">{mastery_est.active_estimator.replace('_', ' ').title()}</div>
                <div class="metric-sub">{mastery_est.total_interactions} Recorded Trials in DB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        speed_val = f"&tau; = {pacing_profile.tau_speed:+.2f}"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Processing Speed (&tau;)</div>
                <div class="metric-val">{speed_val}</div>
                <div class="metric-sub">{pacing_profile.active_estimator.replace('_', ' ').title()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        caution_val = (
            f"a = {pacing_profile.ddm_result.boundary_separation:.3f}"
            if pacing_profile.ddm_result
            else f"z = {pacing_profile.z_score_rt:+.2f}"
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Decision Caution (a)</div>
                <div class="metric-val">{caution_val}</div>
                <div class="metric-sub">{pacing_profile.caution_label.title()} Threshold</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Calibration Error (C)</div>
                <div class="metric-val">{calib_profile.bias_label.replace('_', ' ').title()}</div>
                <div class="metric-sub">Brier Score: {calib_profile.brier_score:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_chart_m, col_chart_ddm = st.columns([6, 4])

    with col_chart_m:
        st.markdown("#### Skill Mastery Distribution")
        df_mastery = pd.DataFrame([
            {
                "Skill": graph.skill_metadata.get(s, {}).get("name", s),
                "Mastery": current_mastery_means.get(s, 0.5),
                "Decayed": "Yes" if s in decayed_skills else "No",
            }
            for s in graph.get_topological_order()
        ])
        st.bar_chart(df_mastery.set_index("Skill")["Mastery"], color="#3B82F6")

    with col_chart_ddm:
        st.markdown("#### EZ-Diffusion Parameters")
        if pacing_profile.ddm_result and pacing_profile.ddm_result.is_valid:
            ddm = pacing_profile.ddm_result
            st.json({
                "drift_rate_v (fluency)": round(ddm.drift_rate, 4),
                "boundary_separation_a (caution)": round(ddm.boundary_separation, 4),
                "non_decision_time_Ter (sec)": round(ddm.non_decision_time, 2),
                "mean_decision_time_MDT (sec)": round(ddm.mean_decision_time, 2),
                "sample_size": ddm.sample_size,
            })
        else:
            st.info(
                f"EZ-diffusion requires at least 15 trials (current: {len(rts)}). "
                "Active estimator is z-score heuristic."
            )

# ---------------------------------------------------------
# TAB 3: PEDAGOGICAL TRANSPARENCY ("Why this action?")
# ---------------------------------------------------------
with tab_policy:
    # 1. Run GNN prerequisite propagation and root cause isolation
    _, rc_diagnosed, attn_map = res["gnn"].propagate_and_trace_root_cause(
        graph, current_mastery_means, current_mastery_vars, struggling_skill=skill_id
    )

    # 2. Execute POMDP depth-3 lookahead rollout conditioned on belief state and diagnosed root cause
    planner: POMDPRolloutPlanner = res["planner"]
    plan_result = planner.plan(
        mastery_means=current_mastery_means,
        p_fatigued=p_fatigued,
        root_cause_skill=rc_diagnosed,
        decayed_skills=decayed_skills,
        struggling_skill=skill_id,
        max_depth=3,
        branching_limit=3,
    )

    st.success(f"**Planner Decision:** {plan_result.explanation}")
    st.caption(f"⚡ Depth-3 Monte Carlo tree completed in {plan_result.planning_time_ms:.1f}ms ({plan_result.total_rollouts} rollouts).")

    st.markdown("#### Explored Monte Carlo Rollout Branches")
    for node in plan_result.rollout_tree:
        is_chosen = (node["action"] == plan_result.best_action.action_type and node["target"] == plan_result.best_action.target_skill)
        badge = "🏆 CHOSEN" if is_chosen else "CANDIDATE"
        color = "#10B981" if is_chosen else "#64748B"

        st.markdown(
            f"""
            <div class="rollout-node" style="border-left-color: {color};">
                <span style="font-size: 11px; font-weight: 700; color: {color};">{badge}</span>
                <div class="rollout-node-title">
                    {node['description']}
                </div>
                <div class="rollout-node-sub">
                    Immediate Utility: <b>{node['immediate_utility']}</b> &nbsp;|&nbsp; 
                    3-Step Expected Rollout Utility: <b>{node.get('expected_rollout_score', 'N/A')}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### GNN Prerequisite Dependency Graph & Attention Weights")
    if rc_diagnosed:
        st.warning(f"⚠️ **Prerequisite Bottleneck Detected:** GNN propagation isolated `{rc_diagnosed}` as the upstream root cause.")
    else:
        st.info("No active upstream prerequisite bottleneck detected.")

    if attn_map:
        st.markdown("**GNN Message Passing Attention Weights along Prerequisite Edges:**")
        df_attn = pd.DataFrame([
            {"Prerequisite": k[0], "Target Skill": k[1], "Attention Weight": round(v, 4)}
            for k, v in attn_map.items()
        ])
        st.dataframe(df_attn, width="stretch")

# ---------------------------------------------------------
# TAB 4: SIMULATOR & BENCHMARKING
# ---------------------------------------------------------
with tab_simulation:
    st.markdown("### Synthetic Student Simulator Benchmarks")
    st.markdown("Verify mentor behavior across preset learner personas.")

    col_p, col_run = st.columns([7, 3])
    with col_p:
        persona = st.selectbox(
            "Select Benchmark Persona:",
            [
                "Persona A: Prerequisite Gap (Fails quadratic equations due to integer arithmetic)",
                "Persona B: Impulsive Guesser (Very fast RT, low caution boundary, random errors)",
                "Persona C: Fatigued Learner (High reaction time, dropping accuracy, high hint rate)",
                "Persona D: Memory Decay (High initial mastery, prolonged absence needing review)",
            ],
        )

    sim_student = None
    if st.button("Run 20-Trial Benchmark Simulation", type="primary"):
        sim: StudentSimulator = res["simulator"]
        if "Persona A" in persona:
            sim_student = sim.create_random_student("benchmark_persona_a", prerequisite_gap_skill="arithmetic_integers")
        elif "Persona B" in persona:
            sim_student = sim.create_random_student("benchmark_persona_b")
            sim_student.caution_a = 0.08
            sim_student.base_rt = 4.0
            sim_student.guess_rate = 0.40
        elif "Persona C" in persona:
            sim_student = sim.create_random_student("benchmark_persona_c")
            sim_student.fatigue_threshold = 3
            sim_student.is_fatigued = True
        else:
            sim_student = sim.create_random_student("benchmark_persona_d")

        # Simulate 20 steps
        benchmark_logs = []
        for step_i in range(20):
            test_skill = graph.get_topological_order()[step_i % len(graph.skill_ids)]
            corr, rt, hint = sim.step(sim_student, test_skill)
            benchmark_logs.append({
                "Trial": step_i + 1,
                "Skill": test_skill,
                "Correct": corr,
                "RT (s)": round(rt, 2),
                "Hint": hint,
            })

        df_bench = pd.DataFrame(benchmark_logs)
        st.dataframe(df_bench, width="stretch")

        # Run diagnostics on simulated student
        bench_accs = [1 if b["Correct"] else 0 for b in benchmark_logs]
        bench_rts = [b["RT (s)"] for b in benchmark_logs]
        bench_pacing = analyze_pacing_and_caution(bench_accs, bench_rts)

        st.markdown(f"**Pacing Diagnostic:** `{bench_pacing.caution_label}` (Estimator: {bench_pacing.active_estimator})")
        st.markdown(f"**Mentor Recommendation:** {bench_pacing.recommendation}")
