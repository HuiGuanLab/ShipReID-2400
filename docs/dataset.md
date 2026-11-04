# ShipReID-2400: access and preparation

[Back to the project README](../README.md)

## Dataset access

The original repository requires a signed dataset license agreement and approval before access. Print and hand-sign the [English agreement](../ShipReID-2400_Dataset_License_Agreement.pdf) or [中文数据集许可协议](../ShipReID-2400数据集许可协议.pdf), scan it, and email it from your official institutional email address to [liubaolongx@gmail.com](mailto:liubaolongx@gmail.com) or [dongjf24@gmail.com](mailto:dongjf24@gmail.com).

| Download route | Details |
| --- | --- |
| Baidu Netdisk / 百度网盘 | [Download ShipReID-2400](https://pan.baidu.com/s/1E7yLizsIv0E8SVgn45IWpQ?pwd=kfbn); extraction code / 提取码: **`kfbn`** |
| Original Google Drive route | Include your Google Drive account details with the application. The maintainers grant access after approval. |

The Baidu link is supplied as an additional location for approved applicants and does not replace the existing license or application process. Dataset images are distributed separately; they are not embedded in the Git repository. The archive's contents and annotation coverage have not been independently verified in this update.

## Prepare files for the current branch

1. Obtain dataset access and extract the archive.
2. Place or link the dataset under `<DATASETS.ROOT_DIR>/ShipReID2400/`. The directory name has no hyphen.
3. Set `DATASETS.ROOT_DIR` to the **parent** directory, and `DATASETS.NAMES` to `ShipReID2400`.
4. For the text-aware code snapshot, prepare the expected JSONL annotations and ensure each `image_path` matches the loader's path exactly. See [annotation format](data-format.md).

```text
<DATASETS.ROOT_DIR>/ShipReID2400/
├── bounding_box_train/
├── bounding_box_test/
├── val_query/
├── test_query/
├── bounding_box_train.jsonl
├── val_query.jsonl
├── val_gallery.jsonl
├── test_query.jsonl
└── test_gallery.jsonl
```

This is the layout expected by the current code, not a verified manifest of the downloadable archive. Generated per-image captions are not yet connected to the loader. The current validation and test loaders use the shared `bounding_box_test` gallery; see [reproducibility notes](reproducibility.md) before interpreting metrics.

## Original dataset partition

The following protocol and counts are preserved from the upstream README:

The 2400 ship IDs are randomly shuffled and split into **train**, **validation**, and **test** sets in a **0.75:0.125:0.125** ratio:  
- Randomly select **1800 IDs** → **Training Set**  
- Randomly select **300 IDs** → **Validation Set**  
- Randomly select **300 IDs** → **Test Set**  

Sets are **mutually exclusive**. During training:  
- Only the training set is used for weight updates  
- Validation and test sets are **solely for accuracy evaluation** (no impact on training)  

According to the re-identification task definition, both validation and test sets require corresponding **query sets**:  
- Query images are selected as **one image per ID per camera** to simulate real-world query scenarios  
- Validation query set: 691 images (300 IDs)  
- Test query set: 706 images (300 IDs)  

| Set                            | # ID | # Images |
| :----------------------------- | :--: | :------: |
| `bounding_box_train` (Train)   | 1800 | 12,988   |
| `val_query` (Validation Query) | 300  | 691      |
| `test_query` (Test Query)      | 300  | 706       |
| `bounding_box_test` (Gallery)  | 600  | 4,253    |

The training/gallery counts sum to 17,241. Query entries must not be added to this total without checking image overlap. See [split accounting](reproducibility.md#dataset-split-accounting) for the unresolved counting issue recorded in the imported snapshot.
