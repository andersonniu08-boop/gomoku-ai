"""Fixed-capacity replay buffer for AlphaZero self-play training examples.

Stores ``TrainingExample`` triples and provides random batching suitable
for feeding a PyTorch ``DataLoader`` or a manual training loop.
"""

from __future__ import annotations

import random
from typing import Iterator

import torch

from selfplay.selfplay import SYMMETRIES, TrainingExample


def _augment_example(ex: TrainingExample) -> TrainingExample:
    """Apply one random D₄ symmetry to *ex*, returning a new example.

    The 8-fold augmentation is applied on retrieval (one transform per
    sample) rather than eagerly on write.  This gives the same effective
    data diversity at 1/8th the memory.
    """
    state_fn, policy_fn = random.choice(SYMMETRIES)
    policy_grid = ex.policy.view(15, 15)
    return TrainingExample(
        state=state_fn(ex.state).clone(),
        policy=policy_fn(policy_grid).reshape(-1).clone(),
        value=ex.value,
    )


class ReplayBuffer:
    """Bounded FIFO buffer of self-play training examples.

    Newest examples displace the oldest once *max_size* is reached.
    Uses a plain list for O(1) indexed access during sampling.

    Parameters:
        max_size: Maximum number of ``TrainingExample`` entries to retain.
    """

    def __init__(self, max_size: int = 500_000):
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._buffer: list[TrainingExample] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_examples(self, examples: list[TrainingExample]) -> None:
        """Append a batch of examples, evicting the oldest if at capacity."""
        self._buffer.extend(examples)
        if len(self._buffer) > self.max_size:
            self._buffer = self._buffer[-self.max_size:]

    def clear(self) -> None:
        """Empty the buffer entirely."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def sample(
        self, batch_size: int, *, augment: bool = True
    ) -> list[TrainingExample]:
        """Return *batch_size* examples chosen uniformly without replacement.

        If *augment* is True (default) each example is randomly transformed
        by one of the 8 D₄ symmetries, giving 8× effective data diversity
        without inflating buffer memory.

        If the buffer contains fewer than *batch_size* entries the whole
        buffer is returned.
        """
        n = len(self._buffer)
        if batch_size >= n:
            result = list(self._buffer)
        else:
            indices = random.sample(range(n), batch_size)
            result = [self._buffer[i] for i in indices]
        if augment:
            result = [_augment_example(ex) for ex in result]
        return result

    def get_batch(
        self, batch_size: int, *, augment: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch and return it as ``(states, policies, values)``.

        Shapes:
            *states*    ``(batch_size, 3, 15, 15)``
            *policies*  ``(batch_size, 225)``
            *values*    ``(batch_size, 1)``
        """
        examples = self.sample(batch_size, augment=augment)
        return _collate(examples)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[TrainingExample]:
        return iter(self._buffer)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a serialisable snapshot of the buffer contents."""
        return {
            "max_size": self.max_size,
            "examples": list(self._buffer),
        }

    @classmethod
    def from_state_dict(cls, data: dict) -> ReplayBuffer:
        """Restore a buffer from a ``state_dict()`` snapshot."""
        buf = cls(max_size=data["max_size"])
        buf.add_examples(data["examples"])
        return buf


# ---------------------------------------------------------------------------
# Collation helper
# ---------------------------------------------------------------------------


def _collate(
    examples: list[TrainingExample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a list of examples into batched tensors."""
    if not examples:
        return (
            torch.empty(0, 3, 15, 15),
            torch.empty(0, 225),
            torch.empty(0, 1),
        )

    states = torch.stack([ex.state for ex in examples])
    policies = torch.stack([ex.policy for ex in examples])
    values = torch.tensor(
        [ex.value for ex in examples], dtype=torch.float32
    ).unsqueeze(1)
    return states, policies, values
