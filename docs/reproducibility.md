# Reproducibility notes

[Back to the project README](../README.md)

These notes describe the current research snapshot. They are not a record of successful reproduction or a guarantee that the published results can be obtained from this checkout.

## Implementation status

This directory is a research snapshot, not yet a complete paper-reproduction package.

| Capability | Current status |
|---|---|
| Global/local visual branches, PFPM, FIM and CILM | Implemented |
| Frozen CLIP text encoder and image-text contrastive loss | Implemented |
| JSONL annotation loading and prompt serialization utility | Implemented |
| Generated per-image captions used by the data loader | **Not enabled in the current snapshot**; the loader currently returns one fixed caption |
| Separate validation/test gallery construction | **Not implemented**; both loaders currently use the shared `bounding_box_test` gallery |
| Standalone `test.py` with the text-aware batch format | Requires an inference-loop update |
| CUB-200-2011, Stanford Cars, NABirds and VesselReID reproduction configs | Not included |

Consequently, the manuscript's generated-caption and cross-domain tables cannot yet be reproduced directly from this checkout.

## Dataset split accounting

The figures below are reported in the existing project documentation. The downloadable archive has not been independently audited as part of this release preparation.

```text
12,988 training images + 2,149 validation images + 2,104 test images = 17,241
```

The manuscript also reports 691 validation-query and 706 test-query entries. If those entries are added again to the figures above, the directory-entry count becomes `18,638` (`17,241 + 691 + 706`), but the manuscript never states that 18,638 is the number of unique images. This repository interprets the query directories as evaluation copies/subsets that do not increase the unique-image total.

That interpretation exposes an ambiguity in the manuscript: Section 3.3 says that the query images are selected first and that "all remaining images" form the gallery, which would make query and gallery disjoint and would instead imply 18,638 unique images. The paper should explicitly state whether query images are duplicated in/included within the reported gallery counts, and revise either the wording or the split counts accordingly.

## Configuration differences

The current implementation uses `256 x 256` inputs because the dual-axis part coordinates are hard-coded for a `16 x 16` patch grid. The manuscript settings recorded in the project documentation are `256 x 128` inputs and 60 training epochs. The example command uses the current snapshot settings (300 epochs). These differences need to be resolved before claiming paper reproduction.

The standalone `test.py` path requires an update to `processor.do_inference` to accept the text-aware validation batches from `val_collate_fn`. Periodic evaluation in `do_train` already handles that batch format.

## Remaining work

Before treating the repository as the artifact for the manuscript, the following items should be completed:

- connect `annotation_to_texts()` to `ImageDataset.__getitem__`;
- validate JSONL path coverage and fail loudly on missing annotations;
- construct separate validation and test galleries matching the paper protocol;
- align the input size, epoch count and loss weights with the manuscript;
- align PFAM's mathematical description with its implemented assignment algorithm;
- align the inference embedding with the feature concatenation defined in the manuscript;
- repair and test the standalone evaluation entry point;
- publish the exact checkpoints, logs, random seeds and baseline configurations used for every table;
- add the fine-grained retrieval and VesselReID configurations claimed in the manuscript;
- add a software license and a complete citation entry; follow the existing dataset license agreements linked from the README.
