# Manual pairing spot checks

Two exports of the spot-check page made by `scripts/make_viewer.py`, each 100 pairs (50 ambiguous, 50 clear, in
random order, seed 0) from the 106 ARKitScenes videos:

| Export | Pairing source | Pairs in the population (ambiguous / clear) |
|---|---|---|
| `spotcheck_0e8b5473ce53.json` | `lidar`: the v0 baseline on iPad LiDAR depth | 759 / 1,956 |
| `spotcheck_4e824803ceb4.json` | `cutr_sam3`: Cubify Anything boxes named by SAM 3 | 537 / 1,833 |

The author judged each pair by hand as *right*, *wrong* or *can't tell*, without being shown which pairs the
pipeline had flagged ambiguous, and wrote notes on the cards.

Each answer holds the pair's identifiers (view key, object id, detection index), its ambiguity flag (stored for
the summary, not shown on the cards), the verdict and the note. No ARKitScenes geometry, depth or images.

**One rule for part-of pairs.** A pair whose label sits on the right object at another level of detail (one
cabinet door paired with the whole cabinet, or the reverse) counts as *right*: the label is not on a different
object. The second check was judged that way. In the first, the author had left those
cards as *can't tell*; the 9 cards whose notes describe a part-of pair (cards 9, 13, 30, 41, 61, 74, 80, 81, 86,
1-based positions in the export) were changed to *right* afterwards, and each keeps
`"verdict_original": "cant_tell"`.

Summarise an export (wrong-pair rate overall and by ambiguous / clear pairs, Wilson 95% intervals, and the rate
weighted by the population) from the repository root:

    python scripts/make_viewer.py --summarize-answers results/spotcheck/spotcheck_0e8b5473ce53.json
    python scripts/make_viewer.py --summarize-answers results/spotcheck/spotcheck_4e824803ceb4.json

`claims.py` checks the results (A3.3, A3.4).
