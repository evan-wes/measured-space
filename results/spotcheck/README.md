# Manual pairing spot check

`spotcheck_0e8b5473ce53.json` is the export of the spot-check page made by `scripts/make_viewer.py` (sample
`0e8b5473ce53`, seed 0, pairing source `lidar`: the v0 baseline's LiDAR pairs on the 106 ARKitScenes videos).
The author judged each pair by hand as *right*, *wrong* or *can't tell*, without being shown which pairs the
pipeline had flagged ambiguous, and wrote a note on every undecided card.

Each answer holds the pair's identifiers (view key, object id, detection index), its ambiguity flag (stored for
the summary, not shown on the cards), the verdict and the note. No ARKitScenes geometry, depth or images.

Summarise it (wrong-pair rate overall and by ambiguous / clear pairs, Wilson 95% intervals, and the rate
weighted by the population) from the repository root:

    python scripts/make_viewer.py --summarize-answers results/spotcheck/spotcheck_0e8b5473ce53.json

The findings are in `docs/LEARNINGS.md` (L7, L31, L32).
