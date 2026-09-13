"""
Script to generate Sarvagya_AILE_10_Learner_Profiles_Guide.docx
Produces a comprehensive reference document with 10 distinct, psychometrically
grounded learner profiles, their exact inputs, engine calculations, and manual test instructions.
"""

from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "Sarvagya_AILE_10_Learner_Profiles_Guide.docx"


def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets inner padding for a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_callout(doc, title, text, bg_hex="F1F5F9", border_hex="3B82F6"):
    """Adds a stylish callout box with a left border."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    # Left border
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{border_hex}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run_t = p.add_run(f"{title}\n")
    run_t.bold = True
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(11)
    run_t.font.color.rgb = RGBColor(15, 23, 42)
    
    run_body = p.add_run(text)
    run_body.font.name = "Calibri"
    run_body.font.size = Pt(10)
    run_body.font.color.rgb = RGBColor(71, 85, 105)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def create_profile_section(doc, num, title, subtitle, context, trials, engine_calc, decision, guidance, how_to_test):
    """Creates a comprehensive profile section."""
    h1 = doc.add_heading(level=2)
    h1_run = h1.add_run(f"Profile {num:02d}: {title}")
    h1_run.font.name = "Calibri"
    h1_run.font.size = Pt(16)
    h1_run.font.color.rgb = RGBColor(30, 41, 59)
    h1.paragraph_format.space_before = Pt(16)
    h1.paragraph_format.space_after = Pt(2)
    
    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run(f"Behavioral Archetype / Diagnostic Label: {subtitle}")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(11)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(100, 116, 139)
    p_sub.paragraph_format.space_after = Pt(8)

    # 1. Context & Psychometric Rationale
    p_ctx = doc.add_paragraph()
    r_ctx_h = p_ctx.add_run("1. Clinical Context & Behavioral Phenotype:\n")
    r_ctx_h.bold = True
    r_ctx_h.font.color.rgb = RGBColor(30, 41, 59)
    r_ctx = p_ctx.add_run(context)
    r_ctx.font.color.rgb = RGBColor(51, 65, 85)
    p_ctx.paragraph_format.space_after = Pt(8)

    # 2. Exact Input Trials Table
    p_tbl_lbl = doc.add_paragraph()
    r_tbl_lbl = p_tbl_lbl.add_run("2. Exact Input Interaction Sequence (Telemetry):")
    r_tbl_lbl.bold = True
    r_tbl_lbl.font.color.rgb = RGBColor(30, 41, 59)
    p_tbl_lbl.paragraph_format.space_after = Pt(4)

    tbl = doc.add_table(rows=len(trials) + 1, cols=7)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    headers = ["Trial", "Question ID", "Skill ID", "Selected", "Outcome", "RT (sec)", "Confidence"]
    col_widths = [Inches(0.6), Inches(1.1), Inches(1.8), Inches(0.8), Inches(0.8), Inches(0.7), Inches(0.9)]

    # Header Row
    hdr_cells = tbl.rows[0].cells
    for idx, (h_text, w) in enumerate(zip(headers, col_widths)):
        hdr_cells[idx].text = h_text
        hdr_cells[idx].width = w
        set_cell_background(hdr_cells[idx], "0F172A")
        set_cell_margins(hdr_cells[idx], top=80, bottom=80, left=100, right=100)
        p = hdr_cells[idx].paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        p.runs[0].font.size = Pt(9)

    # Data Rows
    for r_idx, row_data in enumerate(trials, start=1):
        row_cells = tbl.rows[r_idx].cells
        bg = "FFFFFF" if r_idx % 2 != 0 else "F8FAFC"
        for c_idx, (val, w) in enumerate(zip(row_data, col_widths)):
            row_cells[c_idx].text = str(val)
            row_cells[c_idx].width = w
            set_cell_background(row_cells[c_idx], bg)
            set_cell_margins(row_cells[c_idx], top=60, bottom=60, left=100, right=100)
            p = row_cells[c_idx].paragraphs[0]
            p.runs[0].font.size = Pt(8.5)
            if c_idx == 4:
                if str(val) == "Correct":
                    p.runs[0].font.color.rgb = RGBColor(16, 185, 129)
                    p.runs[0].font.bold = True
                else:
                    p.runs[0].font.color.rgb = RGBColor(239, 68, 68)
                    p.runs[0].font.bold = True
            else:
                p.runs[0].font.color.rgb = RGBColor(51, 65, 85)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 3. What the Engine Computes & Understands
    add_callout(
        doc,
        title="3. What the AI Engine Computes & Infers (Continuous State Vector):",
        text=engine_calc,
        bg_hex="F8FAFC",
        border_hex="3B82F6",
    )

    # 4. Pedagogical Action & Decision
    add_callout(
        doc,
        title="4. Pedagogical Policy Decision (POMDP 3-Step Lookahead):",
        text=decision,
        bg_hex="F0FDF4",
        border_hex="10B981",
    )

    # 5. Socratic Mentor Response
    add_callout(
        doc,
        title="5. Generated Socratic Guidance & Feedback:",
        text=guidance,
        bg_hex="FFFBEB",
        border_hex="F59E0B",
    )

    # 6. How to Test Manually
    p_test = doc.add_paragraph()
    r_test_h = p_test.add_run("6. Manual App Replication Protocol:\n")
    r_test_h.bold = True
    r_test_h.font.color.rgb = RGBColor(30, 41, 59)
    r_test = p_test.add_run(how_to_test)
    r_test.font.color.rgb = RGBColor(71, 85, 105)
    r_test.font.size = Pt(9.5)
    p_test.paragraph_format.space_after = Pt(16)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)


def generate_document():
    doc = Document()

    # Page setup - 0.75 in margins
    for sec in doc.sections:
        sec.top_margin = Inches(0.75)
        sec.bottom_margin = Inches(0.75)
        sec.left_margin = Inches(0.75)
        sec.right_margin = Inches(0.75)

    # Title Block
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    run_title = title_p.add_run("Sarvagya — Adaptive Learning Intelligence Engine (AILE)")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(15, 23, 42)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(16)
    run_sub = sub_p.add_run("10 Calibrated Learner Profiles: Telemetry Inputs, Cognitive State Vectors & Pedagogical Decisions\nComprehensive Manual Testing & Demonstration Manual")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(12)
    run_sub.font.color.rgb = RGBColor(71, 85, 105)

    # Executive Overview Box
    add_callout(
        doc,
        title="EXECUTIVE DESIGN PRINCIPLE: STATE ESTIMATION OVER STATIC LABELS",
        text=(
            "Traditional adaptive learning engines make the fatal error of assigning students to fixed 'learning styles' "
            "or clinical archetypes. In contrast, Sarvagya models the learner as a dynamic, continuous probabilistic state vector:\n\n"
            "S_t = [M_1..M_K, S_1..S_K, τ, a, F_t, C, U]\n\n"
            "Where M_k is knowledge mastery (DKT LSTM/Bayesian), S_k is memory stability (Ebbinghaus decay), "
            "τ is processing speed normalized against question difficulty (Hierarchical RT model), a is response caution "
            "(Drift-Diffusion Model), F_t is attention/fatigue probability (2-State Gaussian HMM), C is metacognitive calibration error, "
            "and U is parameter uncertainty.\n\n"
            "This document defines 10 distinct, mathematically grounded learner profiles with their exact input telemetry, "
            "the engine's latent state calculations, error diagnoses, POMDP lookahead decisions, and step-by-step instructions "
            "to manually replicate them inside the Sarvagya Streamlit application."
        ),
        bg_hex="F8FAFC",
        border_hex="0F172A",
    )

    # -------------------------------------------------------------
    # PROFILE 1: Prerequisite-Blocked Learner
    # -------------------------------------------------------------
    create_profile_section(
        doc=doc,
        num=1,
        title="The Prerequisite-Blocked Learner (Prerequisite Fault Traversal)",
        subtitle="Foundational Prerequisite Bottleneck (Negative Integer Deficit)",
        context=(
            "The student is enrolled in quadratic equations. They understand the conceptual idea of finding roots and "
            "factoring, but consistently fail quadratic equation problems. Standard LMS systems mark quadratic equations as "
            "'unmastered' and deliver repeated quadratic practice. Sarvagya traces the error 3 hops upstream in the prerequisite DAG, "
            "discovering that the genuine root cause is double-negative integer arithmetic."
        ),
        trials=[
            ("1", "q_int_01", "arithmetic_integers", "B (-27)", "Incorrect", "11.2", "70%"),
            ("2", "q_int_02", "arithmetic_integers", "B (18)", "Incorrect", "10.5", "60%"),
            ("3", "q_lin_01", "linear_equations_1var", "B (1)", "Incorrect", "14.5", "55%"),
            ("4", "q_fact_01", "factoring_polynomials", "A (Correct)", "Correct", "12.0", "80%"),
            ("5", "q_quad_01", "quadratic_equations", "B (Roots swapped)", "Incorrect", "15.2", "50%"),
            ("6", "q_quad_02", "quadratic_equations", "B (Sign error)", "Incorrect", "16.0", "45%"),
        ],
        engine_calc=(
            "• Mastery Vector M_k: arithmetic_integers = 0.22, linear_equations_1var = 0.48, factoring_polynomials = 0.81, quadratic_equations = 0.32.\n"
            "• GNN Prerequisite Message Passing: PyTorch GNN propagates beliefs along prerequisite edges. Ancestors of quadratic_equations are evaluated. arithmetic_integers emerges with the lowest post-propagation belief (0.24).\n"
            "• Error Taxonomy Classification: 'Knowledge Deficit' (Foundational Prerequisite Gap in arithmetic_integers).\n"
            "• Speed & Caution: Processing speed τ = -0.05 (normal deliberation), Caution a = 0.142 (balanced boundary)."
        ),
        decision=(
            "• Action Selected: REMEDIATE_ROOT_CAUSE (Target: arithmetic_integers).\n"
            "• POMDP Depth-3 Monte Carlo Rollout: Evaluates branches. Remediating arithmetic_integers yields 3-step expected utility of +12.08, "
            "versus +4.20 for repeating quadratic equations. The planner redirects the student to fix the upstream arithmetic bottleneck first."
        ),
        guidance=(
            "Mentor Hint: 'Notice that this quadratic equation requires combining like signs across terms. "
            "Double-check your double-negative addition rule (-14 - (-8)) before solving for roots.'\n"
            "Pedagogical Prescription: Temporarily route learner to 3-minute prerequisite refresher on integer arithmetic."
        ),
        how_to_test=(
            "1. Select 'Alice (Prerequisite Gap Demo)' in the sidebar dropdown (or use 'Live Student' and input the 6 trials above).\n"
            "2. Navigate to Tab 2 (Cognitive Telemetry): Observe that arithmetic_integers has the lowest mastery bar.\n"
            "3. Navigate to Tab 3 (Pedagogical Transparency): Verify the alert: '⚠️ Prerequisite Bottleneck Detected: GNN propagation isolated arithmetic_integers'.\n"
            "4. Verify the top POMDP candidate is 'remediate_root_cause' on arithmetic_integers with highest rollout utility."
        ),
    )

    # -------------------------------------------------------------
    # PROFILE 2: Impulsive Fast Guesser
    # -------------------------------------------------------------
    create_profile_section(
        doc=doc,
        num=2,
        title="The Impulsive Fast Guesser (Low Caution Boundary)",
        subtitle="Sub-Threshold Deliberation & Overconfident Guessing",
        context=(
            "This student answers questions in 2.8 to 4.5 seconds—substantially faster than the cognitive reading and calculation "
            "time required for multi-step algebra. They rate confidence high (85–95%) on virtually every question despite scoring "
            "under 45%. The engine recognizes that the problem is not low IQ, but an abnormally low decision boundary (a)."
        ),
        trials=[
            ("1", "q_int_01", "arithmetic_integers", "A", "Correct", "3.8", "90%"),
            ("2", "q_int_02", "arithmetic_integers", "B", "Incorrect", "3.2", "95%"),
            ("3", "q_pemdas_01", "order_of_operations", "B", "Incorrect", "4.1", "85%"),
            ("4", "q_frac_01", "fraction_operations", "B", "Incorrect", "4.2", "95%"),
            ("5", "q_lin_01", "linear_equations_1var", "A", "Correct", "4.5", "85%"),
            ("6", "q_fact_01", "factoring_polynomials", "B", "Incorrect", "4.4", "85%"),
            ("7", "q_quad_01", "quadratic_equations", "B", "Incorrect", "3.5", "90%"),
        ],
        engine_calc=(
            "• EZ-Diffusion Parameters: Boundary separation a = 0.088 (Significantly below normal range 0.12–0.18; classified as 'Impulsive').\n"
            "• Drift Rate v = 0.062 (Weak evidence accumulation due to truncation of deliberation).\n"
            "• Non-Decision Time Ter = 2.40s (Sensory perception & motor execution consumes almost the entire reaction time).\n"
            "• Metacognitive Calibration C: Brier score = 0.42, Directional Bias = +0.48 ('Severely Overconfident').\n"
            "• Error Taxonomy Classification: 'Impulsive Error' (Sub-threshold decision time; answered before processing)."
        ),
        decision=(
            "• Action Selected: INTRODUCE_COGNITIVE_FRICTION / SCAFFOLD_STEP_VERIFICATION.\n"
            "• POMDP Rationale: Pushing new content or traditional lectures fails because the learner rushes past instructions. "
            "The policy mandates forced pause intervals and step-by-step intermediate input before choice submission."
        ),
        guidance=(
            "Mentor Hint: 'Notice: You answered in 3.4 seconds. Before selecting an option, write out step 1 and step 2 on scratch paper.'\n"
            "Pacing Advice: '⏱️ Low decision boundary detected. Slow down slightly on multi-step equations to convert guesses into confirmed mastery.'"
        ),
        how_to_test=(
            "1. Select 'Bob (Impulsive Fast Guesser Demo)' in the sidebar.\n"
            "2. Open Tab 2: Check Decision Caution gauge: displays 'a = 0.088' and badge 'Impulsive'.\n"
            "3. Observe Metacognition card: displays 'Overconfident' with high Brier calibration error.\n"
            "4. Go to Tab 1: Answer any question rapidly (<4.0s) with high confidence. Notice the feedback immediately diagnoses 'Impulsive Error'."
        ),
    )

    # -------------------------------------------------------------
    # PROFILE 3: Deliberate High-Accuracy Learner
    # -------------------------------------------------------------
    create_profile_section(
        doc=doc,
        num=3,
        title="The Deliberate High-Accuracy Learner ('Slow ≠ Weak')",
        subtitle="High Decision Threshold (a) & High Fluency (v)",
        context=(
            "A conventional EdTech application with a crude 'time bonus' or 'speed scoring' would penalize this learner as 'slow'. "
            "However, this student takes 20–35 seconds per question because they systematically write out intermediate proofs and verify "
            "roots. Their accuracy is 92%. Sarvagya's hierarchical response time model separates processing speed from intelligence."
        ),
        trials=[
            ("1", "q_int_01", "arithmetic_integers", "A", "Correct", "18.5", "85%"),
            ("2", "q_pemdas_01", "order_of_operations", "A", "Correct", "22.1", "90%"),
            ("3", "q_frac_01", "fraction_operations", "A", "Correct", "26.4", "85%"),
            ("4", "q_lin_01", "linear_equations_1var", "A", "Correct", "28.0", "90%"),
            ("5", "q_fact_01", "factoring_polynomials", "A", "Correct", "25.2", "95%"),
            ("6", "q_quad_01", "quadratic_equations", "A", "Correct", "32.0", "90%"),
            ("7", "q_qgraph_01", "quadratic_functions_graph", "A", "Correct", "34.5", "90%"),
        ],
        engine_calc=(
            "• Hierarchical RT Model: Latent speed parameter τ = -0.32 (Deliberate pacing relative to item intensity λ_j).\n"
            "• EZ-Diffusion Parameters: Boundary separation a = 0.215 (High caution / verification threshold), Drift rate v = 0.284 (High cognitive fluency).\n"
            "• Mastery Vector M_k: All active skills > 0.88.\n"
            "• Calibration C: Brier score = 0.04 ('Well-Calibrated').\n"
            "• Fatigue F_t: 0.08 (Steady, focused cognitive engagement)."
        ),
        decision=(
            "• Action Selected: ADVANCE_TO_SYNTHESIS (Target: harder transfer applications).\n"
            "• POMDP Rationale: DO NOT RETEACH. Removing artificial time pressure preserves high reasoning accuracy. "
            "Introduce open-ended synthesis and non-standard problem modeling."
        ),
        guidance=(
            "Mentor Guidance: 'Pacing is balanced and deliberate. Your verification steps are yielding high mathematical precision.'\n"
            "Pedagogical Prescription: Introduce challenging multi-step transfer problems without timer constraints."
        ),
        how_to_test=(
            "1. Select 'Live Student' in the sidebar.\n"
            "2. Submit 5 consecutive correct answers with response times between 22 and 30 seconds and confidence at 85%–90%.\n"
            "3. Navigate to Tab 2: Notice Cognitive Fluency displays 'High Fluency' while Decision Caution reflects high verification threshold a > 0.18.\n"
            "4. Notice POMDP planner recommends advancing to advanced curriculum rather than repeating foundational practice."
        ),
    )

    # -------------------------------------------------------------
    # PROFILE 4: Cognitive Fatigue Onset
    # -------------------------------------------------------------
    create_profile_section(
        doc=doc,
        num=4,
        title="The Cognitive Fatigue Onset Learner (Temporal State F_t)",
        subtitle="Within-Session Performance Collapse (HMM State Transition)",
        context=(
            "This learner starts the session performing with high competence (trials 1–14: 85% accuracy, 10s latency, zero hints). "
            "Around trial 15, cognitive fatigue sets in. Latency spikes to 20s+, hint requests multiply, and accuracy crashes to 25%. "
            "Rather than misdiagnosing this as a permanent learning disability (e.g. 'ADHD'), the 2-state HMM detects a temporary fatigue state."
        ),
        trials=[
            ("12", "q_fact_01", "factoring_polynomials", "A", "Correct", "11.2", "85%"),
            ("13", "q_quad_01", "quadratic_equations", "A", "Correct", "12.5", "80%"),
            ("14", "q_quad_02", "quadratic_equations", "A", "Correct", "13.0", "80%"),
            ("15", "q_fn_01", "function_notation", "B", "Incorrect", "18.5", "50% (Hint)"),
            ("16", "q_fn_02", "function_notation", "B", "Incorrect", "19.8", "45% (Hint)"),
            ("17", "q_qgraph_01", "quadratic_functions_graph", "B", "Incorrect", "21.2", "40% (Hint)"),
            ("18", "q_qgraph_02", "quadratic_functions_graph", "B", "Incorrect", "22.5", "35% (Hint)"),
        ],
        engine_calc=(
            "• Rolling Observation Window [acc, rt_z, hint_rate]: [0.20, +1.85, 0.80].\n"
            "• 2-State Gaussian HMM: State 0 = Engaged, State 1 = Fatigued. Observation vector matches State 1 emission parameters.\n"
            "• Posterior Probability: P(F_t = fatigued) = 0.88 (Jumped from 0.12 earlier in session).\n"
            "• Error Taxonomy Classification: 'Cognitive Overload' (Task complexity and neural fatigue exceed working memory capacity)."
        ),
        decision=(
            "• Action Selected: REST_BREAK (3-minute cognitive micro-break).\n"
            "• POMDP Depth-3 Rollout: Pushing practice during fatigue incurs a -6.0 utility penalty. "
            "A rest break resets attention and yields highest expected terminal utility (+8.0 * P(fatigue))."
        ),
        guidance=(
            "Mentor Hint: 'Cognitive fatigue detected. If you feel tired, take a 3-minute stretch or water break.'\n"
            "Pedagogical Prescription: Temporarily lock new question intake; prompt restorative micro-break."
        ),
        how_to_test=(
            "1. Select 'Charlie (Cognitive Fatigue Onset Demo)' in the sidebar.\n"
            "2. Open Tab 2: Observe Fatigue Risk card: displays '88%' (HMM 2-State Posterior).\n"
            "3. Open Tab 3 (Pedagogical Transparency): Verify top POMDP action is 'rest_break' with explanation citing fatigue mitigation.\n"
            "4. Go to Tab 1: Notice the mentor hint tone shifts to restorative and suggests stepping away for 3 minutes."
        ),
    )

    # -------------------------------------------------------------
    # PROFILE 5: Memory Decay Candidate
    # -------------------------------------------------------------
    create_profile_section(
        doc=doc,
        num=5,
        title="The Memory Decay Candidate ('Mastery ≠ Retention')",
        subtitle="High Initial Competence with Elapsed Retention Half-Life",
        context=(
            "This student achieved 92% mastery on polynomial factoring 10 days ago. However, memory decay follows Ebbinghaus's "
            "exponential curve R(Δt) = exp(-Δt / S). While a naive system assumes the skill is 'checked off', Sarvagya's memory "
            "scheduler flags that retrievability has fallen below the critical 0.65 threshold, necessitating spaced retrieval."
        ),
        trials=[
            ("1", "q_fact_01", "factoring_polynomials", "A (10 days ago)", "Correct", "10.0", "90%"),
            ("2", "q_fact_02", "factoring_polynomials", "A (10 days ago)", "Correct", "11.0", "90%"),
            ("3", "q_fact_01", "factoring_polynomials", "B (Today)", "Incorrect", "17.5", "50%"),
        ],
        engine_calc=(
            "• Mastery Estimate M_k: factoring_polynomials = 0.86 (High prior latent competence).\n"
            "• Memory State S_k: Stability = 4.2 days, Elapsed Δt = 10 days.\n"
            "• Current Retrievability R_k: exp(-10 / 4.2) = 0.092 &rarr; 0.42 (Decayed below target 0.65).\n"
            "• Review Urgency: (0.85 - 0.42) / 0.85 = 0.51 (High review urgency).\n"
            "• Error Taxonomy Classification: 'Retrieval Failure' (Previously mastered, but memory decayed over time)."
        ),
        decision=(
            "• Action Selected: SPACED_RETRIEVAL (Target: factoring_polynomials).\n"
            "• POMDP Utility: Prevents total extinction of the memory trace before introducing advanced quadratic synthesis."
        ),
        guidance=(
            "Mentor Hint: 'You mastered polynomial factoring recently, but memory traces naturally decay without practice. "
            "Recall: what two numbers multiply to +12 and sum to -7?'\n"
            "Pedagogical Prescription: Low-stakes retrieval quiz to restore memory stability S_k to 12+ days."
        ),
        how_to_test=(
            "1. Select 'Alice' or 'Live Student'.\n"
            "2. Navigate to Tab 2 (Cognitive Telemetry): Inspect Skill Mastery Distribution chart and notice skills tagged with 'Decayed: Yes'.\n"
            "3. In Tab 3 (Pedagogical Transparency): Verify POMDP candidate 'spaced_review' evaluated for decaying skills."
        ),
    )

    # -------------------------------------------------------------
    # PROFILE 6: Underconfident Hesitant Learner
    # -------------------------------------------------------------
    create_profile_section(
        doc=doc,
        num=6,
        title="The Underconfident Hesitant Learner",
        subtitle="Negative Metacognitive Calibration Bias",
        context=(
            "This learner gets 85% of questions right with sound reasoning and solid latency. However, when asked to rate self-confidence "
            "before submitting, they consistently rate themselves at 20%–40% ('I'm not sure, probably guessing'). This metacognitive gap "
            "leads to math anxiety and hesitation on standardized exams."
        ),
        trials=[
            ("1", "q_int_01", "arithmetic_integers", "A", "Correct", "14.2", "30%"),
            ("2", "q_pemdas_01", "order_of_operations", "A", "Correct", "15.0", "25%"),
            ("3", "q_frac_01", "fraction_operations", "A", "Correct", "16.1", "35%"),
            ("4", "q_lin_01", "linear_equations_1var", "A", "Correct", "17.0", "30%"),
            ("5", "q_fact_01", "factoring_polynomials", "A", "Correct", "15.5", "40%"),
        ],
        engine_calc=(
            "• Empirical Accuracy: 100% (5/5).\n"
            "• Mean Self-Reported Confidence: 32% (0.32).\n"
            "• Calibration Directional Bias: 0.32 - 1.00 = -0.68 ('Severely Underconfident').\n"
            "• Brier Score: mean((0.32 - 1.0)^2) = 0.462.\n"
            "• Pacing Profile: Caution a = 0.17 (Deliberate), Fluency v = 0.22 (High)."
        ),
        decision=(
            "• Action Selected: REINFORCE_SELF_EFFICACY & REDUCE_SCAFFOLDING.\n"
            "• POMDP Rationale: The student does not need remediation. They need cognitive feedback confirming that their systematic steps are consistently accurate."
        ),
        guidance=(
            "Mentor Feedback: 'Notice: You rated your confidence at only 30%, but your solution was mathematically exact! "
            "Your systematic process is sound—trust your initial derivation.'\n"
            "Pedagogical Prescription: Calibration training; encourage trusting first instincts."
        ),
        how_to_test=(
            "1. Select 'Live Student'.\n"
            "2. Answer 4 questions correctly, but set the confidence slider to 25%–35% on each.\n"
            "3. Open Tab 2: Observe Metacognition card: displays 'Underconfident' with calibration recommendation.\n"
            "4. Expand 'Need a Socratic Hint?' in Tab 1: Notice the reflection prompt urges trusting verified steps."
        ),
    )

    # -------------------------------------------------------------
    # PROFILE 7: Developing Procedural Learner
    # -------------------------------------------------------------
    create_profile_section(
        doc=doc,
        num=7,
        title="The Developing Procedural Learner",
        subtitle="Emerging Mastery with Systematic Sign/Distribution Slip",
        context=(
            "The student understands the broad algebraic goal (isolate the variable x), but repeatedly misapplies a specific procedural "
            "rule: distributing the negative sign across brackets (e.g., treating -(x - 1) as -x - 1 instead of -x + 1). "
            "This is neither random guessing nor total concept absence—it is an isolated procedural bug."
        ),
        trials=[
            ("1", "q_lin_01", "linear_equations_1var", "A", "Correct", "12.0", "75%"),
            ("2", "q_lin_02", "linear_equations_1var", "B (Distribute error)", "Incorrect", "15.4", "65%"),
            ("3", "q_ineq_01", "linear_inequalities", "B (Forgot flip)", "Incorrect", "14.2", "70%"),
            ("4", "q_ineq_02", "linear_inequalities", "B (Direction error)", "Incorrect", "16.1", "60%"),
        ],
        engine_calc=(
            "• Mastery Estimate M_k: linear_equations_1var = 0.58, linear_inequalities = 0.44 (Zone of Proximal Development: 0.40–0.70).\n"
            "• Error Taxonomy Classification: 'Procedural Misconception' (Concept recognized, but algebraic transformation rule misapplied).\n"
            "• Distractor Diagnostic: Captured sign distribution misconception rationale from questions.json.\n"
            "• Caution a: 0.138 (Balanced)."
        ),
        decision=(
            "• Action Selected: SCAFFOLD_HINT / FADED_WORKED_EXAMPLE.\n"
            "• POMDP Rationale: Neither raw repetition nor prerequisite rollback is optimal. Present a worked example highlighting the sign distribution step, then fade guidance."
        ),
        guidance=(
            "Mentor Hint: 'When subtracting a grouped fraction -(x - 1)/2, distribute the negative sign to both terms inside the numerator: -(x) - (-1) = -x + 1.'\n"
            "Misconception Analysis: 'Did not distribute negative sign to -1: treated as -(x) - 1.'"
        ),
        how_to_test=(
            "1. Select 'Live Student' and navigate to question q_lin_02.\n"
            "2. Select Option B (distractor rationale: did not distribute negative sign) and submit with 14s RT and 65% confidence.\n"
            "3. Observe feedback: displays 'Error Diagnosis: Procedural Misconception' with exact rationale."
        ),
    )

    # -------------------------------------------------------------
    # PROFILE 8: Careless Slip Expert
    # -------------------------------------------------------------
    create_profile_section(
        doc=doc,
        num=8,
        title="The Careless Slip Expert",
        subtitle="High Competence with Transient Execution Noise",
        context=(
            "A high-performing student with 94% mastery across all prerequisite skills makes an isolated arithmetic subtraction error "
            "on a single routine trial. In rigid rule-based systems, a single error immediately demotes the student back to remedial modules. "
            "Sarvagya's Bayesian prior and DKT sequence modeling absorb transient slips without disrupting curriculum momentum."
        ),
        trials=[
            ("1", "q_int_01", "arithmetic_integers", "A", "Correct", "7.0", "95%"),
            ("2", "q_pemdas_01", "order_of_operations", "A", "Correct", "8.5", "95%"),
            ("3", "q_fact_01", "factoring_polynomials", "A", "Correct", "9.2", "90%"),
            ("4", "q_fact_02", "factoring_polynomials", "A", "Correct", "8.8", "95%"),
            ("5", "q_int_02", "arithmetic_integers", "B (Arithmetic slip)", "Incorrect", "8.0", "90%"),
            ("6", "q_quad_01", "quadratic_equations", "A", "Correct", "10.5", "95%"),
        ],
        engine_calc=(
            "• Bayesian Beta Posterior: Prior Beta(14, 2) updated with 1 error &rarr; Beta(14, 3) &rarr; Mastery remains 0.824 (High).\n"
            "• Memory Stability S_k: High (18+ days).\n"
            "• Error Taxonomy Classification: 'Careless Slip' (High prior mastery > 0.75 and high retrievability > 0.75 under normal deliberation).\n"
            "• GNN DAG: Post-propagation beliefs across all upstream skills remain comfortably above adequacy threshold (>0.75)."
        ),
        decision=(
            "• Action Selected: ADVANCE_CURRICULUM (Do not demote).\n"
            "• POMDP Rationale: Re-teaching foundational concepts to an expert who suffered an isolated typo creates boredom and frustrates the learner. Note the slip and advance."
        ),
        guidance=(
            "Mentor Feedback: 'Isolated arithmetic slip detected. You have demonstrated solid mastery on this concept—moving directly to the next curriculum milestone.'\n"
            "Pedagogical Prescription: Maintain trajectory toward quadratic graphing."
        ),
        how_to_test=(
            "1. In 'Live Student', answer 4 questions correctly, then get 1 arithmetic question wrong.\n"
            "2. Notice feedback classifies the error as 'Careless Slip' rather than 'Knowledge Deficit'.\n"
            "3. Verify Tab 3 POMDP planner recommends advancing rather than initiating prerequisite repair."
        ),
    )

    # -------------------------------------------------------------
    # PROFILE 9: Cognitively Overloaded Learner
    # -------------------------------------------------------------
    create_profile_section(
        doc=doc,
        num=9,
        title="The Cognitively Overloaded Learner",
        subtitle="Task Complexity Exceeds Working Memory Bandwidth",
        context=(
            "The student jumps from simple 1-variable equations into complex quadratic graph analysis (finding the vertex, axis of "
            "symmetry, and parabolic orientation of y = -x^2 + 6x - 5). The interacting conceptual elements exceed working memory capacity. "
            "Latency exceeds 40 seconds, hint requests occur repeatedly, and the student exhibits extreme hesitation."
        ),
        trials=[
            ("1", "q_lin_01", "linear_equations_1var", "A", "Correct", "12.0", "75%"),
            ("2", "q_qgraph_01", "quadratic_functions_graph", "B", "Incorrect", "38.2", "40% (Hint)"),
            ("3", "q_qgraph_02", "quadratic_functions_graph", "B", "Incorrect", "44.5", "30% (Hint)"),
        ],
        engine_calc=(
            "• Load Sensitivity Index: ΔRT / ΔComplexity > 2.8 (Severe latency spike upon multi-variable introduction).\n"
            "• Hierarchical RT z-score: +2.10 (Abnormally elevated deliberation with zero fluency extraction).\n"
            "• Drift Rate v = 0.021 (Evidence accumulation stalled).\n"
            "• Error Taxonomy Classification: 'Cognitive Overload' (Task complexity and interacting elements exceed working memory capacity)."
        ),
        decision=(
            "• Action Selected: DECOMPOSE_TASK / ISOLATE_COMPONENTS.\n"
            "• POMDP Rationale: Decompose the complex graphing problem into constituent single-variable tasks: "
            "First compute x = -b/(2a) in isolation; then compute f(x); then evaluate parabolic sign."
        ),
        guidance=(
            "Mentor Hint: 'This graphing problem combines three separate rules at once. Let's break it down into steps: "
            "First, calculate the horizontal coordinate x = -b / (2a). What does that give you?'\n"
            "Pedagogical Prescription: Scaffolded task decomposition."
        ),
        how_to_test=(
            "1. Select 'Live Student' and navigate to q_qgraph_02.\n"
            "2. Click 'Request Socratic Guidance', wait 40+ seconds, select Option B, and submit with low confidence (30%).\n"
            "3. Observe feedback: Error diagnosed as 'Cognitive Overload' with decomposition prescription."
        ),
    )

    # -------------------------------------------------------------
    # PROFILE 10: Fluent Mastery Transfer Learner
    # -------------------------------------------------------------
    create_profile_section(
        doc=doc,
        num=10,
        title="The Fluent Mastery Transfer Learner",
        subtitle="Equilibrium State Across Mastery, Pacing & Metacognition",
        context=(
            "The ideal learning state: high mastery across foundational skills, balanced pacing (τ ≈ 0.0), healthy decision boundary "
            "(a ≈ 0.14), high cognitive fluency (v > 0.30), low fatigue (F_t < 0.10), and calibrated metacognition. "
            "The engine operates in full curriculum acceleration mode."
        ),
        trials=[
            ("1", "q_int_01", "arithmetic_integers", "A", "Correct", "9.5", "90%"),
            ("2", "q_pemdas_01", "order_of_operations", "A", "Correct", "11.0", "90%"),
            ("3", "q_frac_01", "fraction_operations", "A", "Correct", "12.0", "85%"),
            ("4", "q_lin_01", "linear_equations_1var", "A", "Correct", "13.2", "90%"),
            ("5", "q_fact_01", "factoring_polynomials", "A", "Correct", "12.4", "90%"),
            ("6", "q_quad_01", "quadratic_equations", "A", "Correct", "14.0", "90%"),
            ("7", "q_fn_01", "function_notation", "A", "Correct", "11.5", "95%"),
            ("8", "q_qgraph_01", "quadratic_functions_graph", "A", "Correct", "15.2", "90%"),
        ],
        engine_calc=(
            "• DKT LSTM Mastery: All 9 skills > 0.85.\n"
            "• EZ-Diffusion Parameters: Drift v = 0.312 (High fluency), Boundary a = 0.145 (Optimal speed-accuracy balance), Ter = 3.10s.\n"
            "• Memory Stability S_k: Solid across all skills (> 14 days).\n"
            "• Metacognition C: Brier score = 0.02, Bias = +0.02 ('Well-Calibrated').\n"
            "• Fatigue F_t = 0.05 (Peak engagement)."
        ),
        decision=(
            "• Action Selected: ADVANCE_NEW_SKILL / CAPSTONE_CHALLENGE.\n"
            "• POMDP Rationale: Maximum curriculum progression utility. Introduce non-standard composite functions and optimization."
        ),
        guidance=(
            "Mentor Guidance: 'Exceptional mastery and pacing balance. All prerequisite foundations verified. Moving to advanced functional modeling.'\n"
            "Pedagogical Prescription: Introduce higher-order curriculum synthesis."
        ),
        how_to_test=(
            "1. In 'Live Student', answer 8 questions correctly across different skills with steady 10–15s timing and 90% confidence.\n"
            "2. Navigate to Tab 2: Observe mastery bars all at 85%+ with 'Well-Calibrated' metacognition and 'High Fluency'.\n"
            "3. Verify Tab 3 POMDP planner recommends Capstone advancement."
        ),
    )

    # -------------------------------------------------------------
    # SUMMARY PROTOCOL TABLE
    # -------------------------------------------------------------
    h_sum = doc.add_heading(level=2)
    h_sum_r = h_sum.add_run("Summary Comparison Matrix: 10 Learner Profiles")
    h_sum_r.font.name = "Calibri"
    h_sum_r.font.size = Pt(15)
    h_sum_r.font.color.rgb = RGBColor(15, 23, 42)

    sum_tbl = doc.add_table(rows=11, cols=6)
    sum_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    sum_tbl.autofit = False

    s_headers = ["Profile #", "Archetype", "Key Telemetry Signal", "Inferred Latent State", "Diagnosed Error", "Optimal Action"]
    s_widths = [Inches(0.6), Inches(1.5), Inches(1.5), Inches(1.1), Inches(1.1), Inches(1.2)]

    for idx, (sh_text, w) in enumerate(zip(s_headers, s_widths)):
        c = sum_tbl.rows[0].cells[idx]
        c.text = sh_text
        c.width = w
        set_cell_background(c, "0F172A")
        set_cell_margins(c, top=80, bottom=80, left=100, right=100)
        p = c.paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        p.runs[0].font.size = Pt(8.5)

    matrix_data = [
        ("1", "Prerequisite-Blocked", "Fails quadratics; integer errors", "M_prereq=0.22, GNN gap", "Knowledge Deficit", "Repair Prerequisite"),
        ("2", "Impulsive Guesser", "RT 3s, Conf 95%, Acc 40%", "Low caution a=0.088, C=+0.48", "Impulsive Error", "Cognitive Friction"),
        ("3", "Deliberate Thinker", "RT 28s, Acc 92%, 0 hints", "High a=0.21, High v=0.28", "None (Correct)", "Advance / Don't Reteach"),
        ("4", "Fatigued Slump", "Late RT spike, hint surge", "P(F_t=fatigue)=0.88", "Cognitive Overload", "3-Min Rest Break"),
        ("5", "Memory Decay", "10 days elapsed, fail retrieval", "M=0.86, R_k=0.42 (decayed)", "Retrieval Failure", "Spaced Retrieval"),
        ("6", "Underconfident", "Acc 100%, Conf 30%", "Calibration C=-0.68", "None (Misaligned C)", "Reinforce Self-Efficacy"),
        ("7", "Developing Procedural", "Sign distribution mistake", "M=0.58 (ZPD 0.4-0.7)", "Procedural Error", "Faded Worked Example"),
        ("8", "Careless Slip", "M=0.94, isolated arithmetic slip", "M remains 0.82 (absorbed)", "Careless Slip", "Advance (Do not demote)"),
        ("9", "Cognitively Overloaded", "RT 40s+, hint panic on graphing", "Load Sensitivity > 2.8", "Cognitive Overload", "Decompose Task"),
        ("10", "Fluent Transfer", "Acc 95%, balanced timing", "All M_k>0.85, v>0.30", "None (Mastered)", "Advance New Skill"),
    ]

    for r_idx, row_vals in enumerate(matrix_data, start=1):
        bg = "FFFFFF" if r_idx % 2 != 0 else "F8FAFC"
        for c_idx, (val, w) in enumerate(zip(row_vals, s_widths)):
            c = sum_tbl.rows[r_idx].cells[c_idx]
            c.text = val
            c.width = w
            set_cell_background(c, bg)
            set_cell_margins(c, top=60, bottom=60, left=100, right=100)
            p = c.paragraphs[0]
            p.runs[0].font.size = Pt(8)
            p.runs[0].font.color.rgb = RGBColor(51, 65, 85)

    # Save document
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_FILE)
    print(f"Learner profiles guide document generated successfully at: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_document()
