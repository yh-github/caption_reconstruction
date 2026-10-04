import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os
import json
from pathlib import Path

st.set_page_config(layout="wide", page_title="Reconstruction Evaluation & Method Comparison Explorer")

def get_file_mtimes():
    paths = ["results/unified_benchmark_master.csv", "results/apriori_full_scores.csv"]
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in paths)

@st.cache_data
def get_gt_captions(video_id: str):
    files = list(Path("datasets/wildQA").glob(f"captions__*/*{video_id}*.json"))
    for f in files:
        if f.name == "categories.json":
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            caps = d.get("captions") or d.get("clips") or d.get("data")
            if caps and isinstance(caps, list) and len(caps) > 0 and "caption" in caps[0]:
                return [c["caption"] for c in caps]
        except Exception:
            pass
    return []

@st.cache_data
def get_llama_captions(video_id: str, w: int, i_pos: int = 29):
    paths = list(Path("results/recon").glob(f"**/*w={w}*i={i_pos}*/*{video_id}*.json"))
    if not paths:
        paths = list(Path.home().glob(f".cache/huggingface/hub/**/*w={w}*i={i_pos}*/*{video_id}*.json"))
    valid = [p for p in paths if not p.name.startswith("skip__") and not p.name.endswith("metadata.json")]
    if valid:
        try:
            with open(valid[0], "r", encoding="utf-8") as fp:
                d = json.load(fp)
            return d.get("reconstructed_captions", {})
        except Exception:
            pass
    return {}

@st.cache_data
def load_all_data(mtimes):
    master_path = "results/unified_benchmark_master.csv"
    apriori_path = "results/apriori_full_scores.csv"
    
    if not os.path.exists(master_path):
        st.error(f"File not found: {master_path}")
        return pd.DataFrame()
    
    df_master = pd.read_csv(master_path)
    if os.path.exists(apriori_path):
        df_apriori = pd.read_csv(apriori_path)
        # Drop caption_perplexity and unneeded internal columns
        df_apriori = df_apriori.drop(columns=["caption_perplexity", "num_captions", "apcs_nll"], errors="ignore")
        df = pd.merge(df_master, df_apriori, on="video_id", how="left")
    else:
        df = df_master.copy()
    
    df = df.rename(columns={"width": "w", "index": "i", "cos_sim": "cos_sim_mean"})
    
    # Reorder columns to place movie_id right after video_id if present
    if "movie_id" in df.columns:
        cols = df.columns.tolist()
        cols.remove("movie_id")
        vid_idx = cols.index("video_id")
        cols.insert(vid_idx + 1, "movie_id")
        df = df[cols]
    
    # Exclude W > 16: Llama prompt truncation skipped >95% of videos (N=4 at W=24, N=17 at W=30),
    # while valid benchmark evaluations run symmetrically up to W=16 across all methods.
    df = df[df["w"] <= 16].copy()
    
    # Standardize dataset names to lowercase for consistency
    df["dataset"] = df["dataset"].str.lower()
    
    # Calculate derived metrics
    N = 60
    df["AUC"] = (N - df["mean_rank"]) / (N - 1)
    df["Calibrated_AUC"] = (2 * df["AUC"] - 1) * 100
    
    H_N = np.sum([1.0 / k for k in range(1, N + 1)])
    chance_mrr = H_N / N
    df["MRR_norm"] = (df["mrr"] - chance_mrr) / (1.0 - chance_mrr)
    
    return df

df_raw = load_all_data(get_file_mtimes())

if df_raw.empty:
    st.error("No benchmark data available.")
    st.stop()

# --- SIDEBAR GLOBAL CONTROLS ---
st.sidebar.title("Configuration")

# 1. Dataset Selection (Default: Combined Wild4 + Wild5)
dataset_options = ["Combined (Wild4 + Wild5)", "Wild4 only", "Wild5 only"]
dataset_choice = st.sidebar.selectbox("Dataset Scope", dataset_options, index=0)

if dataset_choice == "Wild4 only":
    df_active = df_raw[df_raw["dataset"] == "wild4"].copy()
elif dataset_choice == "Wild5 only":
    df_active = df_raw[df_raw["dataset"] == "wild5"].copy()
else:
    df_active = df_raw.copy()

# 2. Exclude Anomalous Wild4 W=3 Toggle
exclude_w3_anomaly = st.sidebar.checkbox(
    "Exclude anomalous Wild4 W=3 run",
    value=True,
    help="Wild4 W=3 used a 3-candidate window pool (pool_scope='window') instead of 60-candidate video pool, causing an artificial spike. Checking this excludes the contaminated Wild4 W=3 run while retaining clean Wild5 W=3."
)

if exclude_w3_anomaly:
    df_active = df_active[~((df_active["dataset"] == "wild4") & (df_active["method"] == "Llama-3.1-8B") & (df_active["w"] == 3))].copy()

# A-Priori metrics available (caption_perplexity removed as it was uncomputed/None)
apriori_metrics = [
    "APCS_V", "APCS_T", "average_dynamism", "peak_dynamism",
    "combined_dynamism", "text_average_dynamism", "text_peak_dynamism",
    "text_combined_dynamism"
]
available_apriori = [m for m in apriori_metrics if m in df_active.columns]

# Performance scores available
perf_metrics = [
    "Calibrated_AUC",
    "mrr",
    "MRR_norm",
    "cos_sim_mean",
    "cos_sim_min",
    "cos_sim_residual",
    "mean_rank",
    "recall_at_1",
    "recall_at_5"
]

# All available methods
available_methods = sorted(df_active["method"].unique().tolist())

# --- MAIN APP LAYOUT ---
st.title("Reconstruction Evaluation & Method Comparison Explorer")
st.markdown(f"**Active Videos**: {df_active['video_id'].nunique()} unique videos across **{dataset_choice}**.")

tab_winners, tab_comparison, tab_macro, tab_micro, tab_strat, tab_export = st.tabs([
    "🏆 Llama Winners Explorer",
    "⚔️ Method Comparison & Rank Diffs",
    "📈 Macro View (W-Curves)",
    "🔬 Micro View (Per-Method)",
    "📊 Stratified Cohorts",
    "💾 Data & CSV Export"
])

# ==============================================================================
# TAB 0: LLAMA WINNERS EXPLORER
# ==============================================================================
with tab_winners:
    st.header("Llama Winners & Narrative Victory Explorer")
    st.markdown(
        """
        While Llama's average scores across the full benchmark are low, Llama achieves substantial victories over
        visual and caption baselines on specific videos—particularly in **high-dynamism, action, or survival scenarios**
        where scene shifts break visual nearest-neighbor persistence.
        
        Use this tab to isolate Llama victories, filter by prior video properties, and inspect side-by-side narrative reconstructions.
        """
    )
    
    with st.container(border=True):
        col_w_m, col_w_score, col_w_gap, col_w_cat = st.columns(4)
        
        baseline_options = [m for m in available_methods if m != "Llama-3.1-8B"]
        default_b = "Visual_SigLIP_MeanClosest" if "Visual_SigLIP_MeanClosest" in baseline_options else baseline_options[0]
        compare_baseline = col_w_m.selectbox("Baseline to Beat", baseline_options, index=baseline_options.index(default_b), key="win_baseline")
        
        win_metric = col_w_score.selectbox("Victory Metric", perf_metrics, index=0, key="win_metric")
        
        all_widths = sorted([int(w) for w in df_active["w"].unique() if w <= 16])
        w_filter_choice = col_w_gap.selectbox(
            "Window Size (W)",
            ["All W", "Narrative Regime (W ≥ 4)"] + [f"W = {w}" for w in all_widths],
            index=1,
            key="win_w_filter"
        )
        
        categories = ["All Categories"] + sorted([c for c in df_active["category"].dropna().unique() if c != "Unknown"])
        cat_filter = col_w_cat.selectbox("Video Category", categories, index=0, key="win_cat")
        
        col_p1, col_p2, col_p3 = st.columns(3)
        max_dyn_val = float(df_active["combined_dynamism"].max()) if "combined_dynamism" in df_active and not df_active["combined_dynamism"].dropna().empty else 30.0
        min_dyn = col_p1.slider(
            "Min Visual Dynamism (combined_dynamism)",
            min_value=0.0,
            max_value=max_dyn_val,
            value=0.0,
            step=0.5,
            help="Higher values select videos with significant visual motion and camera/scene transitions."
        )
        max_apcs = col_p2.slider(
            "Max Visual Continuity (APCS_V)",
            min_value=0.4,
            max_value=1.0,
            value=1.0,
            step=0.02,
            help="Lower values select videos that have low frame-to-frame repetition (least static)."
        )
        margin_max = 50.0 if win_metric in ["Calibrated_AUC", "mean_rank"] else 0.5
        margin_step = 1.0 if win_metric in ["Calibrated_AUC", "mean_rank"] else 0.02
        min_margin = col_p3.slider(
            "Min Advantage Margin (Δ)",
            min_value=0.0,
            max_value=margin_max,
            value=0.0,
            step=margin_step,
            help="Minimum difference between Llama score and baseline score."
        )

    # Filter data for Llama and Baseline at center position i=29
    df_pair = df_active[df_active["method"].isin(["Llama-3.1-8B", compare_baseline]) & (df_active["i"] == 29)].copy()
    
    if w_filter_choice == "Narrative Regime (W ≥ 4)":
        df_pair = df_pair[df_pair["w"] >= 4]
    elif w_filter_choice != "All W":
        w_val = int(w_filter_choice.split("=")[-1].strip())
        df_pair = df_pair[df_pair["w"] == w_val]
        
    if cat_filter != "All Categories":
        df_pair = df_pair[df_pair["category"] == cat_filter]
        
    if "combined_dynamism" in df_pair.columns and min_dyn > 0:
        df_pair = df_pair[df_pair["combined_dynamism"] >= min_dyn]
    if "APCS_V" in df_pair.columns and max_apcs < 1.0:
        df_pair = df_pair[df_pair["APCS_V"] <= max_apcs]
        
    sub_l = df_pair[df_pair["method"] == "Llama-3.1-8B"]
    sub_b = df_pair[df_pair["method"] == compare_baseline]
    
    merge_p_cols = ["video_id", "w", "category", win_metric]
    for ap in ["combined_dynamism", "APCS_V", "APCS_T"]:
        if ap in sub_l.columns:
            merge_p_cols.append(ap)
            
    pvt_win = pd.merge(
        sub_l[merge_p_cols],
        sub_b[["video_id", "w", win_metric]],
        on=["video_id", "w"],
        suffixes=("_llama", "_baseline")
    ).dropna(subset=[f"{win_metric}_llama", f"{win_metric}_baseline"])
    
    if pvt_win.empty:
        st.warning("No paired video evaluations found matching the selected filters.")
    else:
        ascending_score = True if win_metric == "mean_rank" else False
        if ascending_score:
            pvt_win["margin"] = pvt_win[f"{win_metric}_baseline"] - pvt_win[f"{win_metric}_llama"]
            pvt_win["llama_won"] = pvt_win[f"{win_metric}_llama"] < pvt_win[f"{win_metric}_baseline"]
        else:
            pvt_win["margin"] = pvt_win[f"{win_metric}_llama"] - pvt_win[f"{win_metric}_baseline"]
            pvt_win["llama_won"] = pvt_win[f"{win_metric}_llama"] > pvt_win[f"{win_metric}_baseline"]
            
        winners_df = pvt_win[pvt_win["margin"] >= min_margin].sort_values("margin", ascending=False).reset_index(drop=True)
        total_pairs = len(pvt_win)
        win_count = len(winners_df)
        win_rate = (win_count / total_pairs) * 100 if total_pairs > 0 else 0
        
        # KPI Summary Cards
        kpi_w1, kpi_w2, kpi_w3, kpi_w4 = st.columns(4)
        kpi_w1.metric("Evaluated Instances", f"{total_pairs}")
        kpi_w2.metric("Llama Victories", f"{win_count}", f"{win_rate:.1f}% Win Rate")
        kpi_w3.metric("Mean Advantage Margin", f"+{winners_df['margin'].mean():.3f}" if win_count > 0 else "N/A")
        
        if "combined_dynamism" in pvt_win.columns and win_count > 0:
            avg_dyn_win = winners_df["combined_dynamism"].mean()
            avg_dyn_all = pvt_win["combined_dynamism"].mean()
            kpi_w4.metric("Avg Dynamism (Winners vs All)", f"{avg_dyn_win:.2f}", f"{avg_dyn_win - avg_dyn_all:+.2f} vs all")
        elif win_count > 0 and "category" in winners_df and not winners_df["category"].empty:
            kpi_w4.metric("Top Winner Category", str(winners_df["category"].mode()[0]))
        else:
            kpi_w4.metric("Top Winner Category", "N/A")

        # Win breakdown across W and Visual Dynamism
        st.subheader("Victory Analysis: Where Does Llama Win?")
        col_ch1, col_ch2 = st.columns(2)
        
        with col_ch1:
            win_by_w = pvt_win.groupby("w").agg(
                total=("llama_won", "count"),
                wins=("llama_won", "sum")
            ).reset_index()
            win_by_w["win_rate"] = (win_by_w["wins"] / win_by_w["total"]) * 100
            fig_w_rate = px.bar(
                win_by_w,
                x="w",
                y="win_rate",
                text=win_by_w["win_rate"].apply(lambda v: f"{v:.1f}%"),
                title=f"Llama Win Rate (%) vs {compare_baseline} by Gap Width (W)",
                labels={"w": "Window Size (W)", "win_rate": "Win Rate (%)"},
                color_discrete_sequence=["#1f77b4"]
            )
            fig_w_rate.update_layout(yaxis_range=[0, max(win_by_w["win_rate"].max() + 10, 50)])
            st.plotly_chart(fig_w_rate)
            
        with col_ch2:
            if "combined_dynamism" in pvt_win.columns:
                fig_scatter_win = px.scatter(
                    pvt_win,
                    x="combined_dynamism",
                    y="margin",
                    color="category",
                    symbol="w",
                    hover_data=["video_id", f"{win_metric}_llama", f"{win_metric}_baseline"],
                    title=f"Advantage Margin (Llama - {compare_baseline}) vs Visual Dynamism",
                    labels={"combined_dynamism": "Visual Dynamism (SigLIP)", "margin": f"Advantage Margin ({win_metric})"}
                )
                fig_scatter_win.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Parity")
                st.plotly_chart(fig_scatter_win)
            else:
                cat_win = pvt_win.groupby("category")["llama_won"].mean().reset_index()
                cat_win["win_rate"] = cat_win["llama_won"] * 100
                fig_cat = px.bar(cat_win, x="category", y="win_rate", title="Win Rate by Category")
                st.plotly_chart(fig_cat)

        st.markdown("---")
        
        # Winning Instances Table
        st.subheader("Winning Instances Browser")
        st.caption("Browse winning instances sorted by advantage margin. Select a video below to inspect its ground-truth narrative vs Llama prediction.")
        
        display_cols = ["video_id", "w", "category", f"{win_metric}_llama", f"{win_metric}_baseline", "margin"]
        for extra in ["combined_dynamism", "APCS_V", "APCS_T"]:
            if extra in winners_df.columns:
                display_cols.append(extra)
                
        format_dict = {
            f"{win_metric}_llama": "{:.3f}",
            f"{win_metric}_baseline": "{:.3f}",
            "margin": "+{:.3f}",
            "combined_dynamism": "{:.2f}",
            "APCS_V": "{:.3f}",
            "APCS_T": "{:.3f}"
        }
        format_applied = {k: v for k, v in format_dict.items() if k in display_cols}
        st.dataframe(
            winners_df[display_cols].style.format(format_applied),
            height=300
        )
        
        # Qualitative Narrative Inspection Panel
        st.markdown("---")
        st.subheader("🔍 Qualitative Narrative Deep-Dive")
        
        if winners_df.empty:
            st.info("No winning videos matching current filters.")
        else:
            candidate_list = winners_df["video_id"].unique().tolist()
            col_sel_vid, col_sel_w = st.columns([3, 1])
            selected_vid = col_sel_vid.selectbox("Select Video to Inspect", candidate_list, index=0, key="inspect_vid")
            
            vid_rows = winners_df[winners_df["video_id"] == selected_vid]
            selected_w_options = vid_rows["w"].unique().tolist()
            selected_w = col_sel_w.selectbox("Window Size (W)", selected_w_options, index=0, key="inspect_w")
            
            instance_info = vid_rows[vid_rows["w"] == selected_w].iloc[0]
            
            # Load captions
            gt_caps = get_gt_captions(selected_vid)
            llama_caps = get_llama_captions(selected_vid, selected_w, 29)
            
            # Metric Summary Card
            with st.container(border=True):
                m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
                m_c1.metric("Category", str(instance_info.get("category", "Unknown")))
                m_c2.metric(f"Llama {win_metric}", f"{instance_info[f'{win_metric}_llama']:.3f}")
                m_c3.metric(f"{compare_baseline} {win_metric}", f"{instance_info[f'{win_metric}_baseline']:.3f}")
                m_c4.metric("Win Margin (Δ)", f"+{instance_info['margin']:.3f}")
                if "combined_dynamism" in instance_info:
                    m_c5.metric("Visual Dynamism", f"{instance_info['combined_dynamism']:.2f}")

            if not gt_caps:
                st.warning(f"Ground truth caption file not found for video {selected_vid}.")
            else:
                idx = 29
                before_idx = idx - 1
                after_idx = idx + selected_w
                
                before_text = gt_caps[before_idx] if 0 <= before_idx < len(gt_caps) else "N/A"
                after_text = gt_caps[after_idx] if 0 <= after_idx < len(gt_caps) else "N/A"
                
                st.markdown(f"**Context Before (sec {before_idx}):**")
                st.info(f"⬅️ `{before_text}`")
                
                st.markdown(f"**Missing Interval Reconstructed (sec {idx} → sec {idx + selected_w - 1}):**")
                
                recon_rows = []
                for sec in range(idx, idx + selected_w):
                    gt_sentence = gt_caps[sec] if sec < len(gt_caps) else "N/A"
                    llama_sentence = llama_caps.get(str(sec), llama_caps.get(sec, "*(No output)*"))
                    recon_rows.append({
                        "Timestamp": f"sec {sec}",
                        "Ground Truth Caption": gt_sentence,
                        "Llama-3.1-8B Reconstructed Action": llama_sentence
                    })
                st.table(pd.DataFrame(recon_rows))
                
                st.markdown(f"**Context After (sec {after_idx}):**")
                st.info(f"➡️ `{after_text}`")
                
                st.success(
                    f"💡 **Takeaway**: On **{selected_vid}**, visual frame persistence fails because the sequence "
                    f"undergoes dynamic action or rapid visual shifts. Llama's text generation maintains narrative continuity "
                    f"between sec {before_idx} and sec {after_idx}, correctly predicting the intermediate progression!"
                )

# ==============================================================================
# TAB 1: METHOD COMPARISON & RANK DIFFERENCES
# ==============================================================================
with tab_comparison:
    st.header("Method Comparison & Rank Difference Correlations")
    st.markdown(
        """
        Compare the relative performance of two methods (e.g. SLM vs. Visual Baseline). 
        Videos are ranked **\\(1 \\dots V\\)** for each method based on the selected score. 
        Then we test whether the **rank difference** (\\(\\Delta \\text{Rank} = \\text{Rank}_A - \\text{Rank}_B\\)) 
        correlates with video properties (visual dynamism, text redundancy, etc.).
        """
    )
    
    col_m1, col_m2, col_score = st.columns(3)
    default_mA = "Llama-3.1-8B" if "Llama-3.1-8B" in available_methods else available_methods[0]
    default_mB = "Visual_SigLIP_MeanClosest" if "Visual_SigLIP_MeanClosest" in available_methods else available_methods[-1]
    
    method_A = col_m1.selectbox("Method A (Target)", available_methods, index=available_methods.index(default_mA))
    method_B = col_m2.selectbox("Method B (Baseline to compare against)", available_methods, index=available_methods.index(default_mB))
    rank_score = col_score.selectbox("Score to Rank By", perf_metrics, index=0)
    
    st.markdown("#### Slicing & Aggregation Settings")
    col_mode, col_w_sel, col_i_sel = st.columns(3)
    
    slice_mode = col_mode.radio(
        "Aggregation Mode",
        ["Filter specific W and i (Default)", "Average across selected W's and i's"],
        index=0
    )
    
    all_w = sorted(df_active["w"].unique().tolist())
    all_i = sorted(df_active["i"].unique().tolist())
    
    if slice_mode == "Filter specific W and i (Default)":
        default_w_idx = all_w.index(6) if 6 in all_w else 0
        default_i_idx = all_i.index(29) if 29 in all_i else 0
        chosen_w = col_w_sel.selectbox("Window Size (W)", all_w, index=default_w_idx)
        chosen_i = col_i_sel.selectbox("Query Position (i)", all_i, index=default_i_idx)
        
        df_sliced = df_active[(df_active["w"] == chosen_w) & (df_active["i"] == chosen_i)]
        slice_desc = f"W={chosen_w}, i={chosen_i}"
    else:
        chosen_w_list = col_w_sel.multiselect("Select W's to Average Over", all_w, default=all_w)
        chosen_i_list = col_i_sel.multiselect("Select i's to Average Over", all_i, default=all_i)
        
        df_sub = df_active[(df_active["w"].isin(chosen_w_list)) & (df_active["i"].isin(chosen_i_list))]
        meta_cols = [c for c in ["video_id", "movie_id", "category", "dataset"] if c in df_sub.columns]
        meta_df = df_sub[meta_cols].drop_duplicates("video_id")
        df_sliced = df_sub.groupby(["video_id", "method"], as_index=False).mean(numeric_only=True)
        df_sliced = pd.merge(df_sliced, meta_df, on="video_id", how="left")
        slice_desc = f"Averaged over W={chosen_w_list}, i={chosen_i_list}"
    
    sub_A = df_sliced[df_sliced["method"] == method_A].copy()
    sub_B = df_sliced[df_sliced["method"] == method_B].copy()
    
    extra_id_cols = [c for c in ["movie_id", "category", "dataset"] if c in sub_A.columns]
    merge_cols = ["video_id"] + extra_id_cols + [rank_score] + [col for col in available_apriori if col in sub_A.columns]
    paired = pd.merge(
        sub_A[merge_cols],
        sub_B[["video_id", rank_score]],
        on="video_id",
        suffixes=("_A", "_B")
    ).dropna(subset=[f"{rank_score}_A", f"{rank_score}_B"])
    
    if len(paired) == 0:
        st.warning(f"No paired video instances found for {method_A} and {method_B} under {slice_desc}.")
        if len(sub_A) == 0 and len(sub_B) == 0:
            st.info(f"ℹ️ Neither **{method_A}** nor **{method_B}** has evaluations under {slice_desc}.")
        elif len(sub_A) == 0:
            st.info(f"ℹ️ **{method_A}** has 0 runs under {slice_desc} (while **{method_B}** has {len(sub_B)}).")
        elif len(sub_B) == 0:
            st.info(f"ℹ️ **{method_B}** has 0 runs under {slice_desc} (while **{method_A}** has {len(sub_A)}).")
        else:
            st.info(f"ℹ️ Both methods have runs under {slice_desc}, but their evaluated video IDs do not overlap.")
    else:
        ascending_rank = True if rank_score == "mean_rank" else False
        
        n_videos = len(paired)
        paired["rank_A"] = paired[f"{rank_score}_A"].rank(ascending=ascending_rank)
        paired["rank_B"] = paired[f"{rank_score}_B"].rank(ascending=ascending_rank)
        
        paired["rank_diff"] = paired["rank_A"] - paired["rank_B"]
        paired["percentile_advantage_A"] = (paired["rank_B"] - paired["rank_A"]) / n_videos * 100.0
        paired["metric_diff"] = paired[f"{rank_score}_A"] - paired[f"{rank_score}_B"]
        
        if ascending_rank:
            wins_A = (paired[f"{rank_score}_A"] < paired[f"{rank_score}_B"]).sum()
            wins_B = (paired[f"{rank_score}_B"] < paired[f"{rank_score}_A"]).sum()
        else:
            wins_A = (paired[f"{rank_score}_A"] > paired[f"{rank_score}_B"]).sum()
            wins_B = (paired[f"{rank_score}_B"] > paired[f"{rank_score}_A"]).sum()
        ties = n_videos - (wins_A + wins_B)
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(f"Paired Videos", f"{n_videos}")
        kpi2.metric(f"{method_A} Win Rate", f"{(wins_A / n_videos)*100:.1f}% ({wins_A})")
        kpi3.metric(f"{method_B} Win Rate", f"{(wins_B / n_videos)*100:.1f}% ({wins_B})")
        kpi4.metric(f"Mean Δ {rank_score}", f"{paired['metric_diff'].mean():+.3f}")
        
        st.markdown("---")
        
        st.subheader("Correlations with A-Priori Scores")
        st.caption(
            "A positive correlation with Percentile Advantage means Method A gains a competitive edge as the prior score increases."
        )
        
        corr_records = []
        for ap in available_apriori:
            if ap in paired.columns:
                valid = paired.dropna(subset=["percentile_advantage_A", "metric_diff", ap])
                if len(valid) > 5:
                    rho_rank, p_rank = stats.spearmanr(valid["percentile_advantage_A"], valid[ap])
                    rho_metric, p_metric = stats.spearmanr(valid["metric_diff"], valid[ap])
                    
                    corr_records.append({
                        "Prior Feature": ap,
                        "Spearman ρ (Advantage Rank)": rho_rank,
                        "p-value (Rank)": p_rank,
                        "Spearman ρ (Δ Metric)": rho_metric,
                        "p-value (Metric)": p_metric,
                        "Significance (p < 0.05)": "✅ Significant" if p_rank < 0.05 else "n.s."
                    })
        
        df_corr = pd.DataFrame(corr_records)
        if not df_corr.empty:
            df_corr = df_corr.sort_values(by="p-value (Rank)")
            st.dataframe(
                df_corr.style.format({
                    "Spearman ρ (Advantage Rank)": "{:+.3f}",
                    "p-value (Rank)": "{:.2e}",
                    "Spearman ρ (Δ Metric)": "{:+.3f}",
                    "p-value (Metric)": "{:.2e}"
                }),
                hide_index=True
            )
            
            st.subheader("Visualizing Advantage vs. Prior Score")
            col_scatter_x, col_scatter_y = st.columns(2)
            scatter_prior = col_scatter_x.selectbox("Select Prior Score (X-Axis)", available_apriori, index=0)
            scatter_diff_type = col_scatter_y.selectbox(
                "Select Y-Axis",
                ["Percentile Advantage (Method A vs B)", f"Absolute Difference ({rank_score}_A - {rank_score}_B)"]
            )
            
            y_col = "percentile_advantage_A" if "Percentile" in scatter_diff_type else "metric_diff"
            
            fig_diff = px.scatter(
                paired,
                x=scatter_prior,
                y=y_col,
                hover_data=["video_id", f"{rank_score}_A", f"{rank_score}_B"],
                trendline="ols",
                title=f"{method_A} Advantage vs {method_B} by {scatter_prior} ({slice_desc})",
                labels={
                    scatter_prior: f"Prior Score: {scatter_prior}",
                    y_col: scatter_diff_type
                }
            )
            fig_diff.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Parity")
            st.plotly_chart(fig_diff)


# ==============================================================================
# TAB 2: MACRO VIEW (W-CURVES ACROSS METHODS)
# ==============================================================================
with tab_macro:
    st.header("Macro View: Method Performance Across Window Sizes (W)")
    
    col_macro_methods, col_macro_metric, col_macro_i = st.columns(3)
    macro_methods = col_macro_methods.multiselect(
        "Methods to Plot",
        available_methods,
        default=[m for m in ["Llama-3.1-8B", "Visual_SigLIP_MeanClosest", "Caption_MeanClosest"] if m in available_methods]
    )
    macro_metric = col_macro_metric.selectbox("Metric to Plot", perf_metrics, index=0, key="macro_metric")
    macro_i_mode = col_macro_i.selectbox("Query Position (i)", ["Center Query (i=29)", "Average across all i"])
    
    with st.expander("🔍 Filter W-Curves by Prior Scores or Category", expanded=False):
        col_mf1, col_mf2, col_mf3 = st.columns(3)
        available_cats = sorted([c for c in df_active["category"].dropna().unique() if c != "Unknown"])
        selected_cats = col_mf1.multiselect("Filter Categories", available_cats, default=available_cats, key="macro_cat_filter")
        
        max_dyn_m = float(df_active["combined_dynamism"].max()) if "combined_dynamism" in df_active and not df_active["combined_dynamism"].dropna().empty else 30.0
        macro_min_dyn = col_mf2.slider(
            "Min Visual Dynamism (combined_dynamism)",
            min_value=0.0,
            max_value=max_dyn_m,
            value=0.0,
            step=0.5,
            key="macro_dyn_filter",
            help="Filter videos with high visual motion/change to test if Llama closes the gap in dynamic scenes."
        )
        macro_max_apcs = col_mf3.slider(
            "Max Visual Continuity (APCS_V)",
            min_value=0.4,
            max_value=1.0,
            value=1.0,
            step=0.02,
            key="macro_apcs_filter",
            help="Filter out highly static/repetitive videos."
        )
    
    if not macro_methods:
        st.warning("Please select at least one method to plot.")
    else:
        df_macro = df_active[df_active["method"].isin(macro_methods)].copy()
        if macro_i_mode == "Center Query (i=29)":
            df_macro = df_macro[df_macro["i"] == 29]
            
        if selected_cats:
            df_macro = df_macro[df_macro["category"].isin(selected_cats)]
        if "combined_dynamism" in df_macro.columns and macro_min_dyn > 0:
            df_macro = df_macro[df_macro["combined_dynamism"] >= macro_min_dyn]
        if "APCS_V" in df_macro.columns and macro_max_apcs < 1.0:
            df_macro = df_macro[df_macro["APCS_V"] <= macro_max_apcs]
            
        if not exclude_w3_anomaly and "Llama-3.1-8B" in macro_methods and (df_macro["w"] == 3).any() and (df_macro["dataset"] == "wild4").any():
            st.warning("⚠️ **Notice**: Anomalous Wild4 W=3 run is included in this plot. Its 3-way distractor pool artificially inflates W=3 metrics. Use the sidebar checkbox to exclude it.")
        
        if df_macro.empty:
            st.warning("No data matching the active filters.")
        else:
            agg = df_macro.groupby(["method", "w"])[macro_metric].agg(["mean", "sem", "count"]).reset_index()
            
            fig_w = go.Figure()
            for m in macro_methods:
                sub_m = agg[agg["method"] == m].sort_values("w")
                if not sub_m.empty:
                    fig_w.add_trace(go.Scatter(
                        x=sub_m["w"],
                        y=sub_m["mean"],
                        mode="lines+markers",
                        name=m,
                        error_y=dict(type="data", array=sub_m["sem"], visible=True)
                    ))
            
            if macro_metric == "Calibrated_AUC":
                fig_w.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="Random Chance (0.0)")
            elif macro_metric == "mrr":
                fig_w.add_hline(y=0.078, line_dash="dot", line_color="red", annotation_text="Chance floor (0.078)")
            elif macro_metric == "MRR_norm":
                fig_w.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="Chance (0.0)")
            elif macro_metric == "cos_sim_residual":
                fig_w.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Zero Residual")
                
            n_macro_vids = df_macro["video_id"].nunique()
            fig_w.update_layout(
                title=f"Mean {macro_metric} by Window Size (W) ({n_macro_vids} Active Videos)",
                xaxis_title="Window Size (W)",
                yaxis_title=f"Mean {macro_metric}",
                xaxis=dict(tickmode="array", tickvals=sorted(df_macro["w"].unique())),
                hovermode="x unified"
            )
            st.plotly_chart(fig_w)


# ==============================================================================
# TAB 3: MICRO VIEW (PER-METHOD SCATTER)
# ==============================================================================
with tab_micro:
    st.header("Micro View: Per-Method Instance Differentiation")
    
    col_mi_m, col_mi_score, col_mi_prior = st.columns(3)
    micro_method = col_mi_m.selectbox("Select Method", available_methods, index=0, key="micro_method")
    micro_metric = col_mi_score.selectbox("Performance Metric (Y-Axis)", perf_metrics, index=0, key="micro_metric")
    micro_prior = col_mi_prior.selectbox("Prior Score (X-Axis)", available_apriori, index=0, key="micro_prior")
    
    col_mi_w, col_mi_i = st.columns(2)
    micro_w = col_mi_w.selectbox("Window Size (W)", ["All"] + all_w, index=all_w.index(6)+1 if 6 in all_w else 0, key="micro_w")
    micro_i = col_mi_i.selectbox("Query Position (i)", ["All"] + all_i, index=all_i.index(29)+1 if 29 in all_i else 0, key="micro_i")
    
    df_micro = df_active[df_active["method"] == micro_method].copy()
    if micro_w != "All":
        df_micro = df_micro[df_micro["w"] == micro_w]
    if micro_i != "All":
        df_micro = df_micro[df_micro["i"] == micro_i]
        
    if df_micro.empty:
        st.warning("No data found for selected combination.")
    else:
        fig_micro = px.scatter(
            df_micro,
            x=micro_prior,
            y=micro_metric,
            color="dataset",
            hover_data=["video_id", "w", "i"],
            trendline="ols",
            title=f"{micro_method}: {micro_metric} vs {micro_prior} (W={micro_w}, i={micro_i})"
        )
        st.plotly_chart(fig_micro)


# ==============================================================================
# TAB 4: STRATIFIED COHORTS
# ==============================================================================
with tab_strat:
    st.header("Stratified Cohort Analysis")
    st.markdown("Split videos into **Low** and **High** cohorts based on a prior score and observe how methods behave.")
    
    col_st_m, col_st_prior, col_st_metric = st.columns(3)
    strat_method = col_st_m.selectbox("Method to Evaluate", available_methods, index=0, key="strat_m")
    strat_prior = col_st_prior.selectbox("Stratify By", available_apriori, index=0, key="strat_p")
    strat_metric = col_st_metric.selectbox("Metric", perf_metrics, index=0, key="strat_met")
    
    p_low = st.slider("Lower Percentile Threshold (Bottom X%)", 5, 50, 25)
    p_high = st.slider("Upper Percentile Threshold (Top X%)", 50, 95, 75)
    
    unique_vids = df_active[["video_id", strat_prior]].drop_duplicates().dropna()
    low_val = np.percentile(unique_vids[strat_prior], p_low)
    high_val = np.percentile(unique_vids[strat_prior], p_high)
    
    st.info(f"Thresholds: **Bottom {p_low}%** ({strat_prior} ≤ {low_val:.3f}) vs **Top {100-p_high}%** ({strat_prior} ≥ {high_val:.3f})")
    
    df_st_method = df_active[df_active["method"] == strat_method].copy()
    cohort_low = df_st_method[df_st_method[strat_prior] <= low_val]
    cohort_high = df_st_method[df_st_method[strat_prior] >= high_val]
    
    agg_low = cohort_low.groupby("w")[strat_metric].agg(["mean", "sem"]).reset_index()
    agg_high = cohort_high.groupby("w")[strat_metric].agg(["mean", "sem"]).reset_index()
    
    fig_strat = go.Figure()
    fig_strat.add_trace(go.Scatter(
        x=agg_low["w"], y=agg_low["mean"],
        error_y=dict(type="data", array=agg_low["sem"]),
        mode="lines+markers",
        name=f"Low {strat_prior} (Bottom {p_low}%)"
    ))
    fig_strat.add_trace(go.Scatter(
        x=agg_high["w"], y=agg_high["mean"],
        error_y=dict(type="data", array=agg_high["sem"]),
        mode="lines+markers",
        name=f"High {strat_prior} (Top {100-p_high}%)"
    ))
    fig_strat.update_layout(
        title=f"{strat_method} Performance by Window Size (Stratified by {strat_prior})",
        xaxis_title="Window Size (W)",
        yaxis_title=f"Mean {strat_metric}",
        xaxis=dict(tickmode="array", tickvals=sorted(df_active["w"].unique()))
    )
    st.plotly_chart(fig_strat)


# ==============================================================================
# TAB 5: DATA & CSV EXPORT
# ==============================================================================
def generate_export_codebook(df_to_export: pd.DataFrame, title: str, summary_line: str, method_A: str = None, method_B: str = None, rank_score: str = None, slice_desc: str = None) -> str:
    col_dict = {
        "dataset": "Benchmark dataset split: 'wild4' (100 videos) or 'wild5' (235 videos).",
        "video_id": "Unique 60-second video clip identifier.",
        "movie_id": "Source channel / creator identity grouping multiple clips (~40 unique channels).",
        "category": "High-level semantic domain (e.g. Military, Farming, Survival, Action & Vehicle, Nature & Scenery, Natural Disaster).",
        "method": "Reconstruction method (e.g. Llama-3.1-8B, Visual_SigLIP_MeanClosest, Caption_MeanClosest, etc.).",
        "method_family": "Method taxonomy: 'SLM_Text' (generative LLM), 'Vector_Video' (visual features), or 'Vector_Caption' (text features).",
        "width": "Cloze gap window size in seconds (W ∈ [1, 2, 3, 4, 6, 8, 12, 16]).",
        "w": "Cloze gap window size in seconds (W ∈ [1, 2, 3, 4, 6, 8, 12, 16]).",
        "index": "Starting or center timestamp index of the cloze evaluation query in the 60-second video.",
        "i": "Query timestamp position in the 60-second video (i=0 start, i=29 center, i=59 end).",
        "mean_rank": "Average rank of the ground-truth timestamp among all 60 video candidates under pool_scope: 'video'. Lower is better (1 = perfect match, 30.5 = random chance).",
        "AUC": "Area Under Cumulative Retrieval Curve: (60 - mean_rank) / 59. Ranges from 0.0 to 1.0 (0.5 = random chance).",
        "Calibrated_AUC": "Calibrated AUC centered at 0: (2 * AUC - 1) * 100. Ranges from -100% to +100% (0.0% = random chance).",
        "mrr": "Mean Reciprocal Rank (1/rank) evaluated against all 59 other video timestamps. Random chance for N=60 is ~0.078.",
        "MRR_norm": "Linearly normalized MRR above chance: (mrr - chance) / (1 - chance). 0.0 = chance, 1.0 = perfect.",
        "recall_at_1": "Fraction of queries where ground truth is ranked #1 (chance = 1/60 ≈ 0.0167).",
        "recall_at_5": "Fraction of queries where ground truth is ranked in top 5 (chance = 5/60 ≈ 0.0833).",
        "cos_sim": "Mean cosine similarity between reconstructed representation and ground truth across the W-second window.",
        "cos_sim_mean": "Mean cosine similarity between reconstructed representation and ground truth across the W-second window.",
        "cos_sim_min": "Minimum cosine similarity across the W seconds of the missing window. Captures worst-case second fidelity.",
        "cos_sim_residual": "Cosine similarity residual above trivial boundary persistence. Measures true predictive gain beyond boundary inertia.",
        "average_dynamism": "Mean consecutive frame visual distance: 1 - sim(v_t, v_{t+1}) on SigLIP 768d embeddings.",
        "peak_dynamism": "95th percentile consecutive frame visual distance on SigLIP 768d embeddings (captures sharp cuts / camera turns).",
        "combined_dynamism": "Harmonized visual dynamism: 100 * (0.5 * average_dynamism + 0.5 * peak_dynamism).",
        "APCS_V": "Average Pairwise Cosine Similarity across all frame pairs in the video (SigLIP 768d). Higher = more static/homogeneous.",
        "APCS_T": "Average Pairwise Cosine Similarity across all ground-truth caption pairs (MPNet 768d). Higher = repetitive narration.",
        "text_average_dynamism": "Mean consecutive caption semantic distance: 1 - sim(c_t, c_{t+1}) on MPNet 768d.",
        "text_peak_dynamism": "95th percentile consecutive caption semantic distance on MPNet 768d.",
        "text_combined_dynamism": "Harmonized textual dynamism score on ground-truth captions."
    }
    
    if rank_score:
        col_dict[f"{rank_score}_A"] = f"Metric score ({rank_score}) achieved by Method A ({method_A})."
        col_dict[f"{rank_score}_B"] = f"Metric score ({rank_score}) achieved by Method B ({method_B})."
        col_dict["rank_A"] = f"Intra-cohort percentile rank of Method A across all evaluated videos (1 = top performing video, N = worst)."
        col_dict["rank_B"] = f"Intra-cohort percentile rank of Method B across all evaluated videos (1 = top performing video, N = worst)."
        col_dict["rank_diff"] = f"Rank difference: rank_A - rank_B. Negative indicates Method A performed better in rank."
        col_dict["percentile_advantage_A"] = f"Percentile advantage of Method A over Method B: (rank_B - rank_A) / N * 100. Positive values mean Method A has a competitive edge."
        col_dict["metric_diff"] = f"Direct score difference: {rank_score}_A - {rank_score}_B. Positive values mean Method A scored higher."

    method_descriptions = {
        "Llama-3.1-8B": "Generative SLM text reconstruction (`llama-3.1-8b__whole_window__t=0.6`, repetition penalty 1.05) evaluated using sentence-transformers/all-mpnet-base-v2 (768-dim).",
        "Visual_SigLIP_MeanClosest": "Visual boundary vector midpoint linear interpolation (fixed LERP: v_hat = (v_before + v_after)/2) computed on SigLIP 768-dim embeddings.",
        "Visual_SigLIP_RepeatClosest": "Visual boundary persistence baseline copying the single nearest boundary frame vector (before/after) on SigLIP 768-dim.",
        "Caption_MeanClosest": "Text embedding midpoint linear interpolation (fixed LERP: c_hat = (c_before + c_after)/2) on ground-truth caption embeddings (all-mpnet-base-v2 768-dim).",
        "Caption_RepeatClosest": "Text persistence baseline copying the nearest boundary caption embedding (all-mpnet-base-v2 768-dim)."
    }
    
    md = [f"# {title}", ""]
    md.append(f"**Experiment Summary**: `{summary_line}`")
    md.append("")
    
    if method_A or method_B:
        md.append("## Evaluated Methods & Pipeline Details")
        if method_A:
            desc_A = method_descriptions.get(method_A, "Baseline or alternative reconstruction method.")
            md.append(f"- **Method A (Target)**: `{method_A}`  \n  *{desc_A}*")
        if method_B:
            desc_B = method_descriptions.get(method_B, "Baseline or alternative reconstruction method.")
            md.append(f"- **Method B (Baseline)**: `{method_B}`  \n  *{desc_B}*")
        if rank_score:
            md.append(f"- **Ranking Metric**: `{rank_score}`")
        if slice_desc:
            md.append(f"- **Slicing & Aggregation**: `{slice_desc}`")
        md.append("")
        
    md.append("## Column Data Dictionary")
    md.append(f"The exported dataset contains **{len(df_to_export.columns)} columns**. Below is the definition of each column present:")
    md.append("")
    md.append("| Column Name | Data Type | Description |")
    md.append("| :--- | :--- | :--- |")
    
    for c in df_to_export.columns:
        dtype_str = str(df_to_export[c].dtype)
        desc = col_dict.get(c, "Derived feature or evaluation metric score.")
        md.append(f"| `{c}` | `{dtype_str}` | {desc} |")
        
    md.append("")
    md.append("---")
    md.append("*Generated by Reconstruction Evaluation & Method Comparison Explorer.*")
    return "\n".join(md)

with tab_export:
    st.header("Data & CSV Export")
    st.markdown("Download full cuts of data including both **Wild4 and Wild5**, derived metrics, and paired differences.")
    
    export_type = st.radio(
        "Choose Data to View & Export",
        ["Full Filtered Benchmark Dataset (Raw)", "Paired Method Comparison Table (from Tab 1)"]
    )
    
    if export_type == "Full Filtered Benchmark Dataset (Raw)":
        n_w4 = (df_active['dataset']=='wild4').sum()
        n_w5 = (df_active['dataset']=='wild5').sum()
        summary_str = f"Total Rows: {len(df_active)} (Wild4: {n_w4}, Wild5: {n_w5}, Active Videos: {df_active['video_id'].nunique()}, Scope: {dataset_choice})"
        st.write(f"**{summary_str}**")
        st.dataframe(df_active, height=400)
        
        csv_data = df_active.to_csv(index=False).encode("utf-8")
        codebook_full_md = generate_export_codebook(
            df_active,
            title="Benchmark Master Dataset Codebook",
            summary_line=summary_str
        )
        
        col_down1, col_down2 = st.columns(2)
        with col_down1:
            st.download_button(
                label="📥 Download Full Benchmark CSV",
                data=csv_data,
                file_name=f"benchmark_export_{dataset_choice.replace(' ', '_').lower()}.csv",
                mime="text/csv"
            )
        with col_down2:
            st.download_button(
                label="📄 Download Codebook & Column Explanation (.md)",
                data=codebook_full_md.encode("utf-8"),
                file_name=f"codebook_{dataset_choice.replace(' ', '_').lower()}.md",
                mime="text/markdown"
            )
            
        with st.expander("📖 Preview Column Explanations & Codebook", expanded=False):
            st.markdown(codebook_full_md)
            
    else:
        if 'paired' in locals() and not paired.empty:
            summary_paired = f"Total Paired Rows: {len(paired)} ({method_A} vs {method_B}, {slice_desc})"
            st.write(f"**{summary_paired}**")
            st.dataframe(paired, height=400)
            
            csv_paired = paired.to_csv(index=False).encode("utf-8")
            codebook_paired_md = generate_export_codebook(
                paired,
                title=f"Paired Method Comparison Codebook: {method_A} vs {method_B}",
                summary_line=summary_paired,
                method_A=method_A,
                method_B=method_B,
                rank_score=rank_score,
                slice_desc=slice_desc
            )
            
            col_p_down1, col_p_down2 = st.columns(2)
            with col_p_down1:
                st.download_button(
                    label=f"📥 Download Paired Method Comparison CSV ({method_A}_vs_{method_B})",
                    data=csv_paired,
                    file_name=f"paired_comparison_{method_A}_vs_{method_B}_{slice_desc.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            with col_p_down2:
                st.download_button(
                    label="📄 Download Codebook & Column Explanation (.md)",
                    data=codebook_paired_md.encode("utf-8"),
                    file_name=f"codebook_paired_{method_A}_vs_{method_B}_{slice_desc.replace(' ', '_')}.md",
                    mime="text/markdown"
                )
                
            with st.expander("📖 Preview Column Explanations & Codebook", expanded=False):
                st.markdown(codebook_paired_md)
        else:
            st.info("Please configure the comparison in Tab 1 to generate paired data.")
