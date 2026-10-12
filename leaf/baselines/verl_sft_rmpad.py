"""verl's FSDP SFT trainer with remove-padding honoured at sequence parallel 1.

``FSDPSFTTrainer._compute_loss_and_backward`` takes its unpadded path only when
``use_remove_padding and ulysses_sequence_parallel_size > 1``. At SP = 1 it
therefore runs the padded forward: every SEED SFT sample is padded to
``data.max_length`` (32,768) and its full-vocabulary logits are materialised,
about 4 s per sample on an H200 for sequences that average a few thousand
tokens. The unpadded branch is correct at SP = 1 -- the Ulysses gather returns
its input unchanged without a group and the pad-and-slice pads nothing -- so
this launcher takes that branch whenever remove-padding is on and keeps the SP
context manager for SP > 1 only. The loss is the same masked token mean.

The vendored fork is left untouched: the one-condition edit is applied to the
method's own source at import, and the edit fails loudly if upstream changes.
"""

from __future__ import annotations

import inspect
import textwrap

from verl.trainer import fsdp_sft_trainer as trainer

EDITS = (
    (
        "use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1",
        "use_rmpad = self.use_remove_padding\n"
        "    use_sp = use_rmpad and self.config.ulysses_sequence_parallel_size > 1",
    ),
    ("if not use_sp:", "if not use_rmpad:"),
)


def patch() -> None:
    source = textwrap.dedent(inspect.getsource(trainer.FSDPSFTTrainer._compute_loss_and_backward))
    for old, new in EDITS:
        if source.count(old) != 1:
            raise RuntimeError(f"verl SFT trainer changed; cannot apply {old!r}")
        source = source.replace(old, new)
    namespace: dict = {}
    exec(compile(source, trainer.__file__, "exec"), trainer.__dict__, namespace)
    trainer.FSDPSFTTrainer._compute_loss_and_backward = namespace["_compute_loss_and_backward"]


if __name__ == "__main__":
    patch()
    trainer.main()
