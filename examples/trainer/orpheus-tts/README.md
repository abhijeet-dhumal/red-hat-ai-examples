# Orpheus Turkish TTS Fine-Tuning with LoRA

Fine-tune [Orpheus-3B](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained) for **Turkish text-to-speech** using LoRA and the Kubeflow Trainer v2 on Red Hat OpenShift AI.

## Overview

Orpheus-3B is a 3.3B-parameter codec language model built on Llama-3 that generates speech as discrete SNAC audio tokens. By default it only supports English. This example fine-tunes it on Turkish text-audio pairs so it produces intelligible Turkish speech.

### Key features

| Feature | Description |
| --- | --- |
| **LoRA fine-tuning** | Distributed across multiple nodes with PyTorch DDP via `TransformersTrainer` |
| **Progress tracking** | Live steps, epochs, loss, and ETA in the OpenShift AI Jobs dashboard |
| **JIT checkpointing** | Pause-safe / preemption-safe saves to the shared PVC; auto-resume on restart |
| **Kueue scheduling** | TrainJobs are admitted and queued through Kueue (gang scheduling, quotas, prioritization) |
| **MLflow tracking** | Loss, in-training WAV samples, and Whisper WER/CER |
| **Post-training inference** | Load base model + LoRA adapter (no weight merge) and generate Turkish speech in the workbench |

> [!IMPORTANT]
> This example has been tested with OpenShift AI 3.2+ and Kubeflow Trainer v2 on NVIDIA A100-80GB.
> If you have different hardware, adjust `batch_size`, `grad_accum`, and `resources_per_node` accordingly.

### How it works

```text
Turkish text
    → tokenize (Llama-3 vocabulary)
    → Orpheus-3B + LoRA  (predicts SNAC audio tokens)
    → SNAC decoder (24 kHz)
    → speech waveform
```

Orpheus treats speech as a sequence of discrete audio tokens produced by the SNAC codec. During fine-tuning, each training example places Turkish text tokens and the corresponding SNAC audio tokens in a single sequence. The model is trained to generate the audio tokens conditioned on the text; the loss is computed only on audio positions so the objective focuses on speech generation rather than reconstructing the prompt.

At inference, load the pretrained Orpheus base weights together with the trained LoRA adapter (no full weight merge). The model emits SNAC tokens for the input text; the SNAC decoder converts those tokens into a 24 kHz waveform.

### Model and dataset

| Component | Details |
|-----------|---------|
| **Base model** | [unsloth/orpheus-3b-0.1-pretrained](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained) (3.3B, Llama-3 backbone) |
| **Audio codec** | [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz) |
| **Dataset** | [afkfatih/turkish-tts-combined-raw](https://huggingface.co/datasets/afkfatih/turkish-tts-combined-raw) (~81K text-audio pairs) |
| **Fine-tuning method** | LoRA on attention + MLP projections (notebook default: r=64, alpha=128) |

## Requirements

### OpenShift AI cluster

* Red Hat OpenShift AI (RHOAI) 3.2+ with:
  * `dashboard`, `trainer`, and `workbenches` components enabled
* At least 2 worker nodes with NVIDIA GPUs (A100 recommended)
* A dynamic storage provisioner that supports **ReadWriteMany (RWX)** PVCs
* Optional: MLflow tracking server reachable from training pods (platform MLflow, or an in-namespace server at `http://mlflow.<namespace>.svc.cluster.local:5000`)

### Hardware requirements

#### Workbench

| Image | GPU | CPU | Memory | Notes |
| --- | --- | --- | --- | --- |
| Training \| Jupyter \| PyTorch \| CUDA \| Python | 1× GPU | 4 cores | 32Gi | GPU required for post-training inference (base + LoRA) |

#### Training job

| Component | Configuration | GPU per node | Total GPU | CPU | Memory |
| --- | --- | --- | --- | --- | --- |
| Training pods | 2 nodes × 2 GPUs (notebook default) | 2 | 4 | 4 cores/pod | 32Gi/pod |

> [!NOTE]
> Tested on A100-80GB. Works on other Ampere+ GPUs (L40S, H100) with adjusted batch size. For GPUs with less than 40GB VRAM, reduce `batch_size` to 1 or increase `grad_accum`. You can also set `gpus_per_node: 1` for a smaller footprint.

#### Storage

| Purpose | Size | Access mode | Notes |
| --- | --- | --- | --- |
| Shared PVC (`shared`) | 150Gi recommended (60Gi minimum) | ReadWriteMany (RWX) | HF cache, preprocessed data, LoRA checkpoints |

Layout on the PVC (workbench mount `/opt/app-root/src/shared`, TrainJob mount `/mnt/kubeflow-checkpoints`):

```text
shared/orpheus-tts/
  hf-cache/
  preprocessed/          # versioned sentinel .done-<hash>
  checkpoints/           # TrainJob output_dir (LoRA adapters; use checkpoints/final)
  logs/                  # rank0.log + rank0-FATAL.txt
```

## Environment variables

The notebook authenticates to the cluster API using:

| Variable | Description |
| --- | --- |
| `NOTEBOOK_USER_TOKEN` | Auth token (often auto-set in the workbench); otherwise the notebook falls back to the service-account token |
| `MLFLOW_TRACKING_URI` | Optional; forwarded into training pods when set |

Workbench pods can also reach the API at `https://kubernetes.default.svc` with the mounted service-account token.

## PVC mount paths (workbench vs training pods)

* **Workbench mount (user-configured):** a PVC named `shared` is typically mounted at `/opt/app-root/src/shared`.
* **Training pod mount (SDK, fixed):** `TransformersTrainer(output_dir="pvc://shared/orpheus-tts/checkpoints")` mounts that PVC at `/mnt/kubeflow-checkpoints` inside training pods.
* **Checkpoint convention:** persisted adapters land under `orpheus-tts/checkpoints/` on the PVC. Prefer `checkpoints/final` for inference.

## Setup

### 1. Create a Data Science Project

Access the OpenShift AI dashboard, open **Data Science Projects**, and create a project for this example (any name; you will set it as `namespace` in the notebook). After creation, the project overview looks like this:

![](./images/create_project.png)

### 2. Create a workbench

Create a workbench with the **Training | Jupyter | PyTorch | CUDA | Python** image. Select a hardware profile with a **GPU** and enough memory (notebook path uses **4 CPU / 32 GiB**):

![](./images/create_workbench_image.png)

### 3. Create shared storage (RWX)

In the project, create a PVC with **ReadWriteMany (RWX)** access. Name it **`shared`**, size at least **60 GiB** (**150 GiB** recommended for full-data runs), and mount it on the workbench at **`/opt/app-root/src/shared`**:

![](./images/create_storage_mount.png)

When the workbench status is **Running**, open it:

![](./images/workbench_ready.png)

### 4. Clone the repository

From the workbench, clone this repository (terminal or the JupyterLab **Git → Clone a Repository** dialog):

```bash
git clone https://github.com/red-hat-data-services/red-hat-ai-examples.git
```

Navigate to `examples/trainer/orpheus-tts` and open `orpheus_tts_distributed.ipynb`:

![](./images/clone_and_open_notebook.png)

### 5. Edit configuration

In the `%%yaml parameters` cell, set at least:

* `namespace` — your Data Science Project name
* `mlflow_experiment` — experiment name for tracking (optional if you disable MLflow)

For a quick smoke run, set `max_train_samples` to a small value (for example `2000`) and `num_epochs` to `1`. Use `max_train_samples: 0` for the full ~81K dataset.

## Running the example

The notebook walks you through:

1. **Install dependencies** — Kubeflow SDK + `yamlmagic`
2. **Configure parameters** — `%%yaml parameters` (infrastructure, LoRA, training, logging)
3. **Define `train_func`** — self-contained training logic (must be inline; `TransformersTrainer` serializes via `inspect.getsource()`)
4. **Authenticate** — `TrainerClient` via `NOTEBOOK_USER_TOKEN` or the workbench service-account token
5. **Submit distributed training** — `TransformersTrainer` with DDP, periodic + JIT checkpointing, progression tracking, MLflow env
6. **Monitor training** — OpenShift AI **Jobs** UI + notebook logs / MLflow
7. **Resolve LoRA adapter** — pick `checkpoints/final` (or latest checkpoint); no weight merge
8. **Generate Turkish speech** — load base + LoRA via PEFT and listen to audio in the notebook
9. **Cleanup** — delete the training job

## Monitoring

### Training progress

Open **Develop & train → Jobs**. Select your TrainJob to watch steps, epochs, and loss in real time:

![](./images/jobs_details_progress.png)

### Job list and logs

The Jobs page shows the job list beside live pod logs (package install, model load, LoRA size, dataset size, checkpoint path):

![](./images/jobs_list_logs.png)

> [!TIP]
> Keep the notebook `get_job_logs(..., follow=True)` cell running until the job finishes. TrainJob pods are garbage-collected after backoff; durable copies also land under `shared/orpheus-tts/logs/`.

### Pause and resume (JIT checkpointing)

Use **Pause** from the row actions menu to suspend a running job and free GPUs. JIT checkpointing writes the current state to the shared PVC so a later resume can continue:

![](./images/jobs_pause_menu.png)

### Workload admission (Kueue)

Under **Observe & monitor → Workload metrics**, confirm that Kueue has **Admitted** your workloads (workbench, TrainJob, and optional MLflow). This view shows queue utilization, pending vs admitted jobs, and whether the TrainJob cleared Kueue scheduling before pods start:

![](./images/workload_metrics.png)

## Expected outcomes

After training completes:

* LoRA adapter at `<PVC>/orpheus-tts/checkpoints/final/` (inference = base model + adapter)
* Intermediate checkpoints at `<PVC>/orpheus-tts/checkpoints/checkpoint-*`
* MLflow runs with loss curves, audio samples, and WER/CER (when a tracker is configured)
* Generated audio samples playable in the notebook

### Reference results (notebook standalone run)

Executed notebook configuration: **full dataset (~77K train) · 5 epochs · 4× A100 · LoRA r=64 / α=128**.

On a fixed 4-sentence Whisper probe during training:

| Checkpoint | WER mean | CER mean |
|------------|----------|----------|
| Mid-training (best WER, ~step 3000) | **~0.34** | **~0.21** |
| Near end (~step 6000) | ~0.43 | ~0.11 |

Best `eval_loss` during the run was **~3.96**. Prefer mid/late checkpoints for listening.

![Training Loss](./images/training_loss.png)

![WER CER Progress](./images/wer_cer_progress.png)

## Customization

Edit the `%%yaml parameters` cell (passed to `train_func` as `func_args`):

| Parameter | Notebook default | Description |
|-----------|------------------|-------------|
| `namespace` | _(edit)_ | OpenShift AI project / Kubernetes namespace |
| `mlflow_experiment` | `orpheus-turkish-tts` | MLflow experiment name |
| `max_train_samples` | `0` (full ~81K) | Training samples (`0` = full dataset) |
| `num_nodes` | `2` | Distributed training nodes |
| `gpus_per_node` | `2` | GPUs per training node |
| `batch_size` | `4` | Per-device batch size (gradient checkpointing enables larger batches) |
| `grad_accum` | `4` | Gradient accumulation steps |
| `num_epochs` | `5` | Training epochs |
| `learning_rate` | `1e-4` | AdamW learning rate (LoRA typically 1e-4–2e-4) |
| `lora_r` | `64` | LoRA rank |
| `lora_alpha` | `128` | LoRA alpha |
| `lora_dropout` | `0.05` | LoRA dropout |
| `save_steps` | `500` | Checkpoint cadence |
| `eval_steps` | `250` | Loss evaluation cadence |
| `audio_log_steps` | `1000` | In-training wav + Whisper cadence (keep sparse) |

For quicker experiments, lower `max_train_samples`, `num_epochs`, or LoRA rank.

## Troubleshooting

### SNAC preprocessing is slow

SNAC encoding runs on GPU inside the TrainJob (rank 0). For large datasets (>10K samples), the first run may take 15–30 minutes. Subsequent runs skip preprocessing when a matching versioned `.done-<hash>` sentinel exists on the PVC.

### Lost pod logs after failure

TrainJob pods are deleted after backoff. `train_func` tees logs to the shared PVC:

```text
shared/orpheus-tts/logs/rank0.log
shared/orpheus-tts/logs/rank0-FATAL.txt   # written on uncaught exception
```

Also keep the notebook `get_job_logs(..., follow=True)` cell running until the job finishes, or fetch logs immediately on failure before GC.

### Job failed mid-training — will it start from scratch?

No, if the shared PVC still has artifacts:

| Artifact | Location | Behavior on retry |
|----------|----------|-------------------|
| HF model/dataset cache | `orpheus-tts/hf-cache/` | Reused (no re-download) |
| Preprocessed SNAC data | `orpheus-tts/preprocessed/` + `.done-<hash>` | Skipped if fingerprint matches |
| Trainer checkpoints | `orpheus-tts/checkpoints/checkpoint-*` | Auto-resumed via `get_last_checkpoint` |

Resubmit the TrainJob with the same YAML fingerprint (`dataset|samples|seq|model`). A config change invalidates the preprocess sentinel and starts a fresh encode; checkpoints from a different hyperparameter set should be cleared manually if you do not want to resume them.

### Permission denied on the shared PVC

Some NFS storage classes provision volumes as `root:<fixed-gid>` with mode `755`. Training pods then cannot write. Ensure the PVC root (or `orpheus-tts/`) is group/world-writable for your namespace UIDs, or use a storage class that respects `fsGroup`.

### Out of memory during training

Reduce `batch_size` to 1 and increase `grad_accum` to maintain the same effective batch size. Alternatively, use GPUs with more VRAM.

### Out of memory during inference

Base + LoRA load runs in the **workbench**, not the TrainJob. Use at least **32Gi** workbench memory (see hardware profile limits).

### MLflow connection errors

Training pods use `MLFLOW_TRACKING_URI` from the workbench env when set; otherwise they fall back to `http://mlflow.<namespace>.svc.cluster.local:5000`. Point the workbench (or trainer `env`) at a reachable tracker, or temporarily set `report_to="none"` in `train_func` for a smoke run.

### NCCL errors / "No space left on device" on `/dev/shm`

When using multiple GPUs per node (`gpus_per_node > 1`), NCCL allocates shared-memory segments under `/dev/shm` for intra-node communication. The default Kubernetes `emptyDir` is 64MB — too small. The notebook injects a `PodTemplateOverrides` with a 1Gi memory-backed `/dev/shm` volume. If you remove that override and see:

```
Error while creating shared memory segment /dev/shm/nccl-… No space left on device
```

Re-add the `shm_override` in the submit cell.

For other NCCL issues:

```bash
oc logs <pod-name> -c node | grep -i "nccl"
```

Add `NCCL_DEBUG=INFO` to the trainer `env` dict for diagnostics.

### Generated audio is silent or garbled

* Verify the preprocessed dataset has a matching `.done-<hash>` sentinel on the PVC
* Check that the SNAC model was downloaded correctly
* Ensure the base model is `unsloth/orpheus-3b-0.1-pretrained` (Llama-3 vocab, not Llama-2)
* Keep `gen_min_new_tokens` low (notebook uses `28`) so `TOK_EOA` can stop early

## References

* [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) — Lacombe & Kumar, 2025
* [SNAC: Multi-Scale Neural Audio Codec](https://github.com/hubertsiuzdak/snac) — Siuzdak, 2024
* [Kubeflow Trainer v2](https://github.com/kubeflow/trainer) — TrainJob API
* [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft) — HuggingFace
* [Kubeflow Trainer documentation](https://www.kubeflow.org/docs/components/trainer/)
