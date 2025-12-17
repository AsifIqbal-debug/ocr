Custom NID OCR Fine-Tuning Pipeline
=================================

Overview
--------
This directory contains scaffolding to fine-tune a PaddleOCR recognition model on Bangladeshi National ID (NID) cards. The goal is to adapt the OCR network to the mixed Bangla/English typography and security-printing artifacts that commonly confuse generic recognizers.

Directory Layout
----------------
- configs/rec_nid_train.yml       -> PaddleOCR configuration tailored for fine-tuning recognition.
- data/
    - raw/                         -> Drop original scans/photos of NID cards here.
    - annotations/                 -> Ground-truth label files (UTF-8, one line per sample).
    - lmdb/                        -> Generated LMDB dataset used for training (auto-created).
- scripts/
    - prepare_dataset.py           -> Converts image+label pairs into PaddleOCR-compatible LMDB.
    - split_dataset.py             -> Splits labelled samples into train/val/test lists.
    - train_recognition.py         -> Launches PaddleOCR training with the supplied config.
    - export_model.py              -> Exports the best checkpoint to inference format.
- requirements-train.txt          -> Pip requirements for the training environment.

Prerequisites
-------------
1. Install CUDA-enabled PaddlePaddle if you plan to train on GPU.
   - CPU-only command (slow):
     pip install paddlepaddle==3.0.0b1 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
   - CUDA 12.x example:
     pip install paddlepaddle-gpu==3.0.0b1 -f https://www.paddlepaddle.org.cn/whl/windows/gpu/stable.html
2. Install training dependencies:
     pip install -r requirements-train.txt
3. Ensure each NID sample has a clear crop that focuses on text fields (name, parents, ID, DOB). For best results annotate individual text regions instead of full cards.

Dataset Preparation
-------------------
1. Place cropped text images in `data/raw/`. Suggested naming convention: `<split>_<field>_<uuid>.jpg` (e.g., `train_name_c3f1.jpg`).
2. Create a UTF-8 text file listing the corresponding ground-truth transcription for each image. Example:
     data/annotations/labels.txt
       train_name_c3f1.jpg	মির্জা ইমতিয়াজ আহমেদ
       train_name_c3f2.jpg	MIRZA IMTIAZ AHMED
3. Run the split helper to generate manifest files:
     python training/scripts/split_dataset.py \
        --labels data/annotations/labels.txt \
        --train-ratio 0.8 --val-ratio 0.1 \
        --output-dir data/annotations
   Result: `train_list.txt`, `val_list.txt`, `test_list.txt` with `image_path	label` per line.
4. Build LMDB datasets that PaddleOCR can consume:
     python training/scripts/prepare_dataset.py \
        --label-file data/annotations/train_list.txt \
        --image-root data/raw \
        --dest data/lmdb/train
     python training/scripts/prepare_dataset.py \
        --label-file data/annotations/val_list.txt \
        --image-root data/raw \
        --dest data/lmdb/val

Fine-Tuning
-----------
1. Update `configs/rec_nid_train.yml` if you want to tweak batch size, learning rate, or base model.
2. Launch training:
     python training/scripts/train_recognition.py \
        --config configs/rec_nid_train.yml \
        --save-dir checkpoints/nid_rec
   Checkpoints and logs will be written under `checkpoints/nid_rec`.
3. After training, export the best model for inference:
     python training/scripts/export_model.py \
        --config configs/rec_nid_train.yml \
        --model-dir checkpoints/nid_rec/best_accuracy \
        --save-dir exported/nid_rec

Integrating with ocr.py
-----------------------
- Once you obtain an exported recognition model, point EasyOCR or PaddleOCR inference toward it.
- For EasyOCR integration you can load a custom recognition network via the `model_storage_directory` and `user_network_directory` arguments in `easyocr.Reader`. A helper patch is left in `ocr.py` (see TODO comment) to wire your exported model.

Notes
-----
- Quality labels are vital. Include Bangla diacritics precisely and double-check Unicode normalization (NFC).
- Consider augmenting samples (blur, jpeg artifacts, color jitter) to mimic real-world capture noise; PaddleOCR can apply on-the-fly augmentations via the config file.
- For field-level accuracy, maintain separate datasets per field (Name, Father's Name, Mother's Name, DOB, NID number) and train specialised recognizers.
