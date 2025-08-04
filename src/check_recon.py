import sys
from config_loader import load_config
from data_loaders import get_data_loader
from exceptions import UserFacingError
from reconstruction_strategies import Reconstructed
from evaluation import ReconstructionEvaluator
from data_models import CaptionedVideo
from evaluation import round_metrics


def str_ts(ts:float) -> str:
    hours = int(ts//3600)
    minutes = int(ts//60)
    seconds = int(ts % 60)
    if hours>0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"

def ls_recon(path):
    with open(path, 'r') as f:
        i = 1
        for line in f:
            r = Reconstructed.model_validate_json(line)
            print(f"{i}. {r.video_id} {r.metrics or 'FAIL'} {list(r.reconstructed_clips.keys())}")
            i += 1

def load_recon(path, index=None, video_id=None):
    if index is None and not video_id:
        raise Exception('need index or video_id')
    with open(path, 'r') as f:
        i = 1
        for line in f:
            r = Reconstructed.model_validate_json(line)
            if i==index or r.video_id==video_id:
                return r, i
            i += 1
    raise Exception('not found')

def build_evaluators(sentences:list[str]):
    eval_models = [
        'microsoft/deberta-large-mnli', # current
        "roberta-large", # BS default
        'microsoft/deberta-v2-xlarge-mnli',
        'distilbert-base-uncased'
    ]
    d = {}
    for m in eval_models:
        p = {'model_type':m, 'idf':False, 'rescale_with_baseline': False}
        d[str(p)] = ReconstructionEvaluator(**p)
        p = {'model_type':m, 'idf':True, 'rescale_with_baseline': False}
        d[str(p)] = ReconstructionEvaluator(**p).calc_idf(sentences)
    return d

def pretty_compare(original_video, reconstructed_data, tab=True):
    j = 0
    for i, original_clip in enumerate(original_video.clips):
        original_desc = original_clip.data.caption
        ts = f"[{str_ts(original_clip.timestamp.start)} - {str_ts(original_clip.timestamp.end)}]"

        # Check if this clip was reconstructed
        if not i in reconstructed_data.reconstructed_clips:
            if not tab:
                print(f'{i+1}. {original_desc}   {ts}')
            else:
                print(f'{i+1}.\t{original_desc}\t{ts}')
            continue

        recon_clip = reconstructed_data.reconstructed_clips[i]
        recon_desc = recon_clip.data.caption

        # Format the metrics for this specific clip
        f1 = reconstructed_data.metrics.get('bs_f1')[j]
        p = reconstructed_data.metrics.get('bs_p')[j]
        r = reconstructed_data.metrics.get('bs_r')[j]
        j += 1
        metrics_str = f"F1={f1:.3f} P={p:.3f} R={r:.3f}"

        if not tab:
            print(f'{i+1}. ~~ {original_desc} ~~   {ts}')
            print(f'LLM> {recon_desc}')
            print(f'     {metrics_str}')
        else:
            print(f'{i + 1}.\t~~ {original_desc} ~~\t{ts}')
            print(f'   \t{recon_desc}\t  ')
            print(f'   \t{metrics_str}\t  ')


def do_eval(evals:dict[str, ReconstructionEvaluator], original_video: CaptionedVideo, reconstructed_data: Reconstructed):
    print(f'{original_video.video_id=} size={len(original_video.clips)} masked={len(reconstructed_data.reconstructed_clips)}')
    for k,v in evals.items():
        candidates, references = reconstructed_data.align(original_video.clips)
        metrics = round_metrics(v.evaluate(reconstructed_data, original_video), 3)
        bs_f1 = metrics['bs_f1']
        bs_p = metrics['bs_p']
        bs_r = metrics['bs_r']
        print(f"{k}")
        assert len(candidates) == len(references)
        assert len(candidates) == len(metrics['bs_f1'])
        for i in range(len(candidates)):
            print(f"~~ {references[i]} ~~")
            print(f">> {candidates[i]} <<")
            print(f"F1={bs_f1[i]}\tP={bs_p[i]}\tR={bs_r[i]}")
        print()


import pandas as pd
from rich.console import Console
from rich.table import Table


def do_eval_to_dataframe(
        evals: dict[str, ReconstructionEvaluator],
        original_video: CaptionedVideo,
        reconstructed_data: Reconstructed
):
    """
    Runs evaluation and returns a styled pandas DataFrame with results,
    including ranked scores.
    """
    all_results = []

    # --- 1. Gather data for all evaluators ---
    for eval_name, evaluator in evals.items():
        candidates, references = reconstructed_data.align(original_video.clips)

        # We assume evaluate() returns raw PyTorch tensors
        metrics = evaluator.evaluate(reconstructed_data, original_video)

        # Extract the score tensors
        bs_f1 = metrics.get('bs_f1', [])
        bs_p = metrics.get('bs_p', [])
        bs_r = metrics.get('bs_r', [])

        for i in range(len(candidates)):
            all_results.append({
                "eval_params": eval_name,
                "sent_ind": i,
                "orig_sent": references[i],
                "recon_sent": candidates[i],
                "F1": bs_f1[i].item() if hasattr(bs_f1, '__len__') and i < len(bs_f1) else None,
                "P": bs_p[i].item() if hasattr(bs_p, '__len__') and i < len(bs_p) else None,
                "R": bs_r[i].item() if hasattr(bs_r, '__len__') and i < len(bs_r) else None,
            })

    # --- 2. Build and process the DataFrame ---
    if not all_results:
        raise Exception("No evaluation results to display.")

    df = pd.DataFrame(all_results)

    # --- 3. Calculate Rank Columns ---
    # The 'rank' method in pandas with method='first' handles ties gracefully
    df['F1_rank'] = df.groupby('eval_params')['F1'].rank(method='first', ascending=False) - 1
    df['P_rank'] = df.groupby('eval_params')['P'].rank(method='first', ascending=False) - 1
    df['R_rank'] = df.groupby('eval_params')['R'].rank(method='first', ascending=False) - 1

    # Convert ranks to integers
    df[['F1_rank', 'P_rank', 'R_rank']] = df[['F1_rank', 'P_rank', 'R_rank']].astype(int)

    # --- 4. Pretty-print the final table ---
    console = Console()
    table = Table(title=f"Qualitative Analysis for Video: {original_video.video_id}", show_lines=True)

    # Define the columns to display
    display_cols = ["eval_params", "sent_ind", "F1_rank", "P_rank", "R_rank", "F1", "P", "R", "orig_sent", "recon_sent"]

    for col in display_cols:
        table.add_column(col)

    # Format and add rows
    for _, row in df.iterrows():
        # Format floats to 2 decimal places for cleaner output
        f1_str = f"{row['F1']:.3f}" if pd.notna(row['F1']) else "N/A"
        p_str = f"{row['P']:.3f}" if pd.notna(row['P']) else "N/A"
        r_str = f"{row['R']:.3f}" if pd.notna(row['R']) else "N/A"

        table.add_row(
            row['eval_params'],
            str(row['sent_ind']),
            str(row['F1_rank']),
            str(row['P_rank']),
            str(row['R_rank']),
            f1_str,
            p_str,
            r_str,
            f"{row['orig_sent']}",
            f"{row['recon_sent']}"
        )

    console.print(table)

    df.to_csv(original_video.video_id+".csv")

    return df  # Return the full DataFrame for further analysis


from pathlib import Path


def load_run_name(mlflow_path:Path) -> str:
    with open(mlflow_path/'tags/mlflow.runName') as f:
        return f.readlines()[0]


def main():
    """
    Loads a dataset, finds a specific video, and compares its original clips to a reconstructed version.
    """
    # --- Setup ---
    if len(sys.argv) < 3:
        raise UserFacingError("[config] [artifact] [cmd]")


    config = load_config(sys.argv[1])
    mlflow_path = Path(sys.argv[2].removeprefix('file://'))
    art_path = mlflow_path/"artifacts/all_recon_videos.jsonl"
    run_name = load_run_name(mlflow_path)
    cmd = sys.argv[3]
    eval_flag =  sys.argv[-1] == '--eval'

    data_loader = get_data_loader(config["data_config"])

    if cmd=='ls':
        ls_recon(art_path)
        return

    if cmd.startswith('i='):
        reconstructed_data, ind = load_recon(path=art_path, index=int(cmd.split('=')[1]))
    else:
        reconstructed_data, ind = load_recon(path=art_path, video_id=cmd)

    print()
    print(f"{run_name}   video_id='{reconstructed_data.video_id}'   video_index={ind}")

    # --- Find the Original Video ---
    original_video = data_loader.find(reconstructed_data.video_id)

    if not original_video:
        print(f"❌ Error: Could not find original video with ID '{reconstructed_data.video_id}' in the dataset.")
        return

    if reconstructed_data.debug_data:
        llm_response = reconstructed_data.debug_data.pop('llm_response_text')
        print(reconstructed_data.debug_data)
        print('LLM_RESPONSE:')
        print(llm_response)
        print(':LLM_RESPONSE')
        print("ORIG:")
        print(original_video.model_dump_json(indent=4))
        print(":ORIG")
        return

    if eval_flag:
        evals = build_evaluators(data_loader.load_all_sentences())
        # do_eval(evals, original_video, reconstructed_data)
        do_eval_to_dataframe(evals, original_video, reconstructed_data)
    else:
        pretty_compare(original_video, reconstructed_data)


if __name__ == "__main__":
    main()
    print()
