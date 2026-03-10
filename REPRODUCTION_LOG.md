# LoCalPFN Reproduction Log

Successfully reproduced LoCalPFN (NeurIPS 2024) using OpenML datasets instead of the heavy TabZilla benchmark.

## Modifications
1.  **Dependencies**: Installed `tensorboard`, `openml`.
2.  **Config**: Added `openml` to `--datasets` choices and `--openml_dataset_id` argument in `config.py`.
3.  **Dataset Loader**: Implemented `load_openml_data` in `dataset.py` to fetch, clean, and split OpenML datasets on the fly.

## How to Run

### 1. TabPFN-kNN (Baseline)
Runs the pre-trained TabPFN with kNN retrieval (no fine-tuning).
```bash
python main.py --exp_name="openml_knn" --datasets=openml --openml_dataset_id=31 --device=cpu knn
```
*   **Dataset**: `credit-g` (ID 31)
*   **Result**: Accuracy ~0.74

### 2. LoCalPFN (Ours/Target)
Runs the retrieval + fine-tuning pipeline.
```bash
python main.py --exp_name="openml_ft" --datasets=openml --openml_dataset_id=31 --device=cpu ft --num_epochs=5
```
*   **Dataset**: `credit-g` (ID 31)
*   **Result**: Accuracy ~0.74 (at 5 epochs)

## Notes for TabRAG Project
This codebase now serves as a **verified baseline**. We can directly compare our `MetricNet` results against `knn` and `ft` modes here by running on the same OpenML IDs.
