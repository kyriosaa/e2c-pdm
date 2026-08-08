# data/external/ — third-party datasets (untracked)

Public benchmark data obtained from elsewhere. Kept separate from
`data/processed/`, which is reserved for datasets this project derives from its
own recordings — mixing the two makes it ambiguous which numbers in a result are
ours.

Nothing here is reproducible from this repo, so it must be re-downloaded rather
than rebuilt if lost.

## Current contents

| File | Source |
|---|---|
| `CWRU_48k_load_1_CNN_data.npz` | Case Western Reserve University bearing dataset, 48 kHz drive-end, load 1 — preprocessed into CNN-ready arrays. 37 MB. |

CWRU is the reference benchmark cited throughout
`docs/notes/operating_profile_rationale.md` and
`collection_protocol.md` as precedent for seeded-fault rig data. It is here for
methodology comparison, **not** as training data for this project's model: it is
a different motor, different sensor, different mounting, and different sample
rate.
