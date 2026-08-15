"""Pretraining hyperparameters.

The batch-size arithmetic is the part worth checking before every run:

    global_batch_tokens = micro_bs * grad_accum * world_size * seq_len
                        =    8     *     16     *      4     *  4096
                        = 2,097,152 tokens/step   (~2M, standard for 1B models)

    steps = total_tokens / global_batch_tokens
          = 100e9 / 2.097e6
          = 47,684 steps

If you change the GPU count, change `grad_accum` in the opposite direction to
keep the global batch fixed. Changing the global batch means the learning rate
schedule is no longer tuned, and you will see it as a worse loss curve.
"""

from dataclasses import dataclass, asdict
import json


@dataclass
class TrainConfig:
    # ---- run identity -----------------------------------------------------
    run_name: str = "Sutra-1.3B-Base"
    out_dir: str = "checkpoints"
    data_dir: str = "data/tokens"

    # MoE vs dense. MoE trains far faster per token but has the quality of a
    # smaller dense model.
    use_moe: bool = True
    # "main" 1.32B total / 0.28B active, 48 experts. Trains in the same 5 days
    #        as "600m" because active params (= compute) are identical; the
    #        extra capacity is paid for in VRAM, which this box has spare.
    # "600m" 0.56B / 0.28B, 16 experts. Fallback if "main" hits OOM.
    # "1b"   1.02B / 0.39B, higher active params -- only reaches ~15B tokens in
    #        the budget and ends up undertrained.
    model_size: str = "main"
    router_log_every: int = 200      # watch this for expert collapse

    # ---- batch ------------------------------------------------------------
    # This project uses GPUs 6,7,8,9 ONLY (GPU 10 is broken - see src/train.py). Every other GPU on this box is
    # off limits. Always launch with:
    #     CUDA_VISIBLE_DEVICES=6,7,8,9 torchrun --nproc_per_node=4 ...
    #
    #   8 * 8 * 4 * 4096 = 1.05M tokens/step
    # ~1M is the right global batch at this model size; the 2M used for 1B+
    # models would mean too few optimizer steps at this token budget.
    micro_batch_size: int = 8
    grad_accum_steps: int = 8
    seq_len: int = 4096

    # ---- schedule ---------------------------------------------------------
    # Budget: ~1 day data prep, ~5 days pretraining, ~1 day SFT + DPO + eval.
    # On 4 GPUs (not 5) five days buys ~18B tokens, so that is the target.
    # 18B / 1.32B total params is still ~14 tokens per parameter, and ~64 per
    # ACTIVE parameter, which is a well-trained regime for this shape.
    total_tokens: float = 18e9
    warmup_steps: int = 500
    lr: float = 3e-4              # peak; standard for ~1B at 2M batch
    min_lr_ratio: float = 0.1     # cosine decays to 10% of peak, not to 0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95           # NOT 0.999 -- 0.95 is markedly more stable for LLMs
    grad_clip: float = 1.0

    # ---- efficiency -------------------------------------------------------
    # torch.compile is OFF for MoE. Routing sends a data-dependent number of
    # tokens to each of 48 experts every step, so shapes change constantly and
    # Dynamo either recompiles repeatedly or falls back with graph breaks. On a
    # multi-day run that is a reliability risk for an uncertain gain. Revisit
    # only after the run is stable and throughput has been measured.
    compile_model: bool = False
    # Non-negotiable here: without it the MoE expert loop keeps every
    # intermediate for all 48 experts across 15 layers and peaks around 46 GB.
    # With it, 28.5 GB at micro_batch 8.
    activation_checkpointing: bool = True
    dtype: str = "bfloat16"

    # ---- logging / checkpointing -----------------------------------------
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 50
    ckpt_every: int = 1000        # ~7 hours apart at ~25s/step
    keep_last_n: int = 5
    sample_every: int = 500       # generate text so you can watch it learn

    # ---- wandb ------------------------------------------------------------
    # The run id is persisted to <out_dir>/wandb_run_id.txt on first start and
    # reused on every resume, with resume="must". That is what keeps a crashed
    # and restarted run as ONE continuous curve instead of a new chart each
    # time. Metrics are always logged against the global step, so a resumed run
    # picks the line up exactly where it stopped.
    wandb_enabled: bool = True
    wandb_project: str = "abhi-llm"
    wandb_run_name: str = "Sutra-1.3B-Base"

    # ---- stability --------------------------------------------------------
    # If loss jumps by more than this factor over the running average, roll back
    # to the last checkpoint and skip the offending data. Loss spikes are the
    # most common way a long pretraining run dies.
    spike_threshold: float = 1.5
    spike_patience: int = 3

    seed: int = 1337

    def global_batch_tokens(self, world_size: int) -> int:
        return self.micro_batch_size * self.grad_accum_steps * world_size * self.seq_len

    def total_steps(self, world_size: int) -> int:
        return int(self.total_tokens / self.global_batch_tokens(world_size))

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


if __name__ == "__main__":
    cfg = TrainConfig()
    for ws in (1, 2, 4, 5):
        gb = cfg.global_batch_tokens(ws)
        print(f"{ws} GPU: global batch {gb/1e6:.2f}M tokens, "
              f"{cfg.total_steps(ws):,} steps")
