# Local inpainting editor qualification — 9 November 2026

All inputs come from **development-only** working copies. This is a technical
editor test, not a detector evaluation. No foreground label was assigned.

| Attempt | Root and nominal mask area | Configuration | Geometry / containment | Visual decision |
|---|---|---|---|---|
| 1 | `16pro_199`, 4.57% | SD v1.5 inpainting, fp16 compute, 20 steps, default output size | Returned 512×512 from a 512×384 input; numerical warning and black output | Failed; no output retained |
| 2 | `16pro_199`, 4.57% | float32 compute from fp16 weights, explicit 512×384, 20 steps | Correct size; 63.96% of outside-mask pixels changed by mean RGB difference > 2. Official `apply_overlay` derivative preserved outside pixels exactly. | Pineapple replacement not visible; rejected |
| 3 | `16pro_187`, 3.36% | float32 compute, 512×384, 20 steps | Correct size; 86.84% outside-mask change | Requested traffic cone absent; rejected |
| 4 | `16pro_187`, 3.36% | float32 compute, 512×384, 30 steps, guidance 10, `padding_mask_crop=32` | Correct size; 0% outside-mask change | Only a faint mark in target area; traffic cone absent; rejected |
| 5 | `16pro_187`, 3.36% | float32 compute, 512×384, 40 steps, guidance 8, `padding_mask_crop=128` | Correct size; 0% outside-mask change | Requested yellow pavement arrow absent; rejected |

The checkpoint was pinned to revision
`8a4288a76071f7280aedbdb3253bdb9e9d5d84bb`. Seeds and exact prompts
are stored in the per-run JSON files. The five rows include one failed call
and four saved image outputs. None should be entered in the accepted sample
manifest. The accepted-edits count is **zero**.

The outcome shows that this configuration can run locally and can sometimes
preserve geometry and the nominal mask, but it has **not** qualified as a
reliable editor for small, semantically clear local edits. A different editor
or a revised generation protocol needs its own development-only qualification
before full dataset production.
