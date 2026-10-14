"""Unshuffling study: order 5 s pieces of a 60 s clip from captions or from frames.

Plan: docs/new_direction/paper_plan.md; pilot analysis plan: docs/new_direction/unshuffle_pilot_preregistration.md.

    core.py           puzzles, metrics, exact ordering DPs, channel-cluster bootstrap (no data, no models)
    cues.py           order-cue removal for the leak sets (in_context_nocue, original_nocue)
    data.py           pilot videos, caption sets, embeddings, text tags
    arms.py           no-reader (chaining) and learned (probe, pairwise) arms
    run_free.py       build puzzles and run the free arms; writes results/unshuffle/
    frames.py         the captioner's frames as JPEGs, for the readers
    claude_reader.py  listwise Claude reader through `claude -p` (subscription quota)
    analyze.py        tables with channel-cluster CIs, continuity profiles (Prop. 3)

Scripts put scripts/ on sys.path and import this package as `unshuffle`.
"""
