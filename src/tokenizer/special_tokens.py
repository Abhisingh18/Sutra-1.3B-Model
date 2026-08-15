"""The special token vocabulary.

Read this before training the tokenizer, because it is the one decision in the
project that cannot be undone. These tokens must exist in the vocabulary from
the very first pretraining step. If you add them later -- at SFT time, when you
suddenly need a chat template -- their embeddings start from random noise while
every other embedding has seen 100B tokens, and they never fully catch up.

The 32 reserved slots and the 4096 audio slots cost 8.4M parameters, which is
0.7% of the model. That is cheap insurance against having to retrain from
scratch six months from now.
"""

# ---- structural -----------------------------------------------------------
BOS = "<|begin_of_text|>"
EOS = "<|end_of_text|>"
PAD = "<|pad|>"

# ---- chat template --------------------------------------------------------
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END_TURN = "<|end_turn|>"

# ---- reasoning ------------------------------------------------------------
# Chain-of-thought is wrapped in these so it can be hidden from the user at
# serving time, and so the model learns that "thinking" is a distinct mode.
THINK = "<|think|>"
THINK_END = "<|/think|>"

# ---- tool use (unused for now, reserved so it stays possible) --------------
TOOL_CALL = "<|tool_call|>"
TOOL_RESULT = "<|tool_result|>"

CORE_SPECIALS = [
    BOS, EOS, PAD,
    SYSTEM, USER, ASSISTANT, END_TURN,
    THINK, THINK_END,
    TOOL_CALL, TOOL_RESULT,
]

# ---- reserved -------------------------------------------------------------
RESERVED = [f"<|reserved_{i}|>" for i in range(32)]

# Space for a future speech/audio front-end (SLAM-LLM style). An audio encoder
# quantised to 4096 codes can map directly into these ids without resizing the
# embedding matrix or disturbing any trained weight.
AUDIO = [f"<|audio_{i}|>" for i in range(4096)]

ALL_SPECIALS = CORE_SPECIALS + RESERVED + AUDIO

# Vocab budget: 48,000 total
#   ~43,861 learned BPE merges
#      4,139 special tokens (11 core + 32 reserved + 4096 audio)
VOCAB_SIZE = 48_000
N_SPECIAL = len(ALL_SPECIALS)
N_LEARNED = VOCAB_SIZE - N_SPECIAL


def chat_template(messages, add_generation_prompt: bool = True) -> str:
    """Render a conversation into the exact string the model is trained on.

    messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]

    This must stay byte-identical between SFT, DPO and inference. A stray
    newline difference here is one of the most common causes of a chat model
    that behaves worse in production than it did in eval.
    """
    role_token = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}

    out = BOS
    for m in messages:
        out += f"{role_token[m['role']]}\n{m['content']}{END_TURN}\n"
    if add_generation_prompt:
        out += f"{ASSISTANT}\n"
    return out


if __name__ == "__main__":
    print(f"core specials : {len(CORE_SPECIALS)}")
    print(f"reserved      : {len(RESERVED)}")
    print(f"audio         : {len(AUDIO)}")
    print(f"total special : {N_SPECIAL}")
    print(f"learned merges: {N_LEARNED}")
    print()
    print(chat_template([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Bharat ki rajdhani kya hai?"},
    ]))
