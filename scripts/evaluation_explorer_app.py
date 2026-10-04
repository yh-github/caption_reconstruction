import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os

st.set_page_config(layout="wide", page_title="Reconstruction Evaluation & Method Comparison Explorer")

def get_file_mtimes():
    paths = ["results/unified_benchmark_master.csv", "results/apriori_full_scores.csv"]
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in paths)

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
        df = pd.merge(df_master, df_apriori, on="video_id", how="left")
    else:
        df = df_master.copy()
    
    df = df.rename(columns={"width": "w", "index": "i", "cos_sim": "cos_sim_mean"})
    
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

# A-Priori metrics available (note: num_captions excluded as all valid clips have 60 captions)
apriori_metrics = [
    "APCS_V", "APCS_T", "average_dynamism", "peak_dynamism",
    "combined_dynamism", "text_average_dynamism", "text_peak_dynamism",
    "text_combined_dynamism", "caption_perplexity"
]
available_apriori = [m for m in apriori_metrics if m in df_active.columns]

# Performance scores available
perf_metrics = ["Calibrated_AUC", "cos_sim_mean", "mrr", "MRR_norm", "mean_rank", "recall_at_1", "recall_at_5"]

# All available methods
available_methods = sorted(df_active["method"].unique().tolist())

# --- MAIN APP LAYOUT ---
st.title("Reconstruction Evaluation & Method Comparison Explorer")
st.markdown(f"**Active Videos**: {df_active['video_id'].nunique()} unique videos across **{dataset_choice}**.")

tab_comparison, tab_macro, tab_micro, tab_strat, tab_export = st.tabs([
    "⚔️ Method Comparison & Rank Diffs",
    "📈 Macro View (W-Curves)",
    "🔬 Micro View (Per-Method)",
    "📊 Stratified Cohorts",
    "💾 Data & CSV Export"
])

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
        df_sliced = df_sub.groupby(["video_id", "method"], as_index=False).mean(numeric_only=True)
        slice_desc = f"Averaged over W={chosen_w_list}, i={chosen_i_list}"
    
    sub_A = df_sliced[df_sliced["method"] == method_A].copy()
    sub_B = df_sliced[df_sliced["method"] == method_B].copy()
    
    merge_cols = ["video_id", rank_score] + [col for col in available_apriori if col in sub_A.columns]
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
    
    if not macro_methods:
        st.warning("Please select at least one method to plot.")
    else:
        df_macro = df_active[df_active["method"].isin(macro_methods)].copy()
        if macro_i_mode == "Center Query (i=29)":
            df_macro = df_macro[df_macro["i"] == 29]
        
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
            
        fig_w.update_layout(
            title=f"Mean {macro_metric} by Window Size (W)",
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
        st.write(f"Total Rows: **{len(df_active)}** (Wild4: {n_w4}, Wild5: {n_w5})")
        st.dataframe(df_active, height=400)
        
        csv_data = df_active.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Benchmark CSV",
            data=csv_data,
            file_name=f"benchmark_export_{dataset_choice.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )
    else:
        if 'paired' in locals() and not paired.empty:
            st.write(f"Total Paired Rows: **{len(paired)}** ({method_A} vs {method_B}, {slice_desc})")
            st.dataframe(paired, height=400)
            
            csv_paired = paired.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Download Paired Method Comparison CSV ({method_A}_vs_{method_B})",
                data=csv_paired,
                file_name=f"paired_comparison_{method_A}_vs_{method_B}_{slice_desc.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        else:
            st.info("Please configure the comparison in Tab 1 to generate paired data.")
