"""The rollout graph: every model call a harness made, linked by token prefix.

A rollout is not a list of turns. Harnesses retry, spawn subagents, generate titles, and compact
context, and all of it arrives on one wire looking identical. A flat list forces you to guess which
turns belong together; a graph records it.

    node        one model call: the prompt the server tokenized, the tokens it sampled back
    parent      the call whose prompt+completion is a token-prefix of this call's prompt
    root        a call that extends nothing (a new conversation: the agent's, a subagent's,
                a title generator's, or the continuation after a context compaction)
    path        root -> leaf, which is exactly one training sequence

Everything downstream is a walk. `sequences()` concatenates a path into
`input_ids / loss_mask / logprobs`; branch structure tells you which paths are the agent's work and
which are discards.

WHY A GRAPH AND NOT PREFIX-MERGED CHAINS. Three things fall out of it that a chain cannot express:

  * **Retries become visible.** A retried turn is a sibling that never continued: same parent, no
    children. A proxy cannot ask the harness what it discarded, but the shape of the graph shows it.
  * **Subagents separate themselves.** A subagent has its own system prompt, so its first call
    extends nothing and starts its own root. No system-prompt keyword matching required.
  * **Compaction is representable.** A rewritten history is not a prefix extension, so it opens a
    new root instead of corrupting the chain it came from.

TOKEN FIDELITY. We never tokenize. The inference server tokenizes each prompt as a side effect of
serving it and returns `prompt_token_ids`, so turn k+1's prompt IS the canonical tokenization of
everything up to that point, including the tool results the harness inserted. Assistant bodies come
back as sampled `token_ids` with aligned logprobs. So for any path:

    concat(node.context_ids + node.sampled_ids for node in path)

reproduces, exactly, the token sequences the model actually saw and produced. Nothing is re-rendered
through a local chat template, which is the single largest source of silent train/inference skew.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


def common_prefix_len(a: list[int], b: list[int]) -> int:
    """How many leading tokens `a` and `b` share."""
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


@dataclass
class TurnNode:
    """One model call, and where it sits relative to the call before it."""

    node_id: str
    prompt_ids: list[int]
    sampled_ids: list[int]
    sampled_logprobs: list[float] | None = None
    parent_id: str | None = None

    # Provenance, for attribution and for the per-harness notes. Never used in token math.
    index: int = 0  # arrival order within the session
    model: str | None = None
    finish_reason: str | None = None
    harness_session_id: str | None = (
        None  # the harness's OWN session id, when it sends one
    )
    system_digest: str | None = (
        None  # cheap identity for the conversation this belongs to
    )
    n_tools: int = 0
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    # Retained because TRL's `_turns_from_trace` passes `tools` to `apply_chat_template`: the tool
    # manifest is part of the rendered prompt, so a re-tokenization without it does not match what
    # the engine actually saw.
    request_tools: list[dict[str, Any]] | None = None
    response_message: dict[str, Any] = field(default_factory=dict)

    @property
    def end_ids(self) -> list[int]:
        """Cumulative token sequence after this turn: its prompt plus what it sampled."""
        return self.prompt_ids + self.sampled_ids

    def context_ids(self, parent: "TurnNode | None") -> list[int]:
        """Tokens this node adds to the sequence BEFORE the model starts generating.

        For a root that is the whole prompt. For a child it is the interstitial span: the tool
        results, user turns and template scaffolding the harness inserted since the parent stopped
        generating. These are real tokens the model conditioned on, but it did not produce them, so
        they are context (mask 0) rather than targets.
        """
        if parent is None:
            return list(self.prompt_ids)
        return self.prompt_ids[len(parent.end_ids) :]


@dataclass
class TrainingSequence:
    """One path through the graph, flattened. Maps onto TRL's `TrainingSequence` fields."""

    input_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    node_ids: list[str]
    prompt_len: int  # tokens before the first sampled token
    root_id: str
    n_turns: int

    @property
    def n_trainable(self) -> int:
        return sum(self.loss_mask)

    def turn_lengths(self) -> list[int]:
        """Sampled-token count per turn, in order. The join key against a harness trace."""
        lengths, run = [], 0
        for m in self.loss_mask:
            if m:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
        return lengths


class RolloutGraph:
    """All model calls for one rollout, linked by prefix.

    `add_turn` is the whole ingestion path: it finds the parent by token prefix and appends. Calls
    arrive in wire order, but the graph does not depend on that order being meaningful, which matters
    because harnesses issue concurrent requests (parallel subagents, background summarisation).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, TurnNode] = {}
        self._order: list[str] = []
        self._children: dict[str, list[str]] = {}

    # --- construction --------------------------------------------------
    def add_turn(self, node: TurnNode) -> TurnNode:
        node.index = len(self._order)
        node.parent_id = self._find_parent(node)
        self._nodes[node.node_id] = node
        self._order.append(node.node_id)
        self._children.setdefault(node.node_id, [])
        if node.parent_id is not None:
            self._children[node.parent_id].append(node.node_id)
        return node

    def _find_parent(self, node: TurnNode) -> str | None:
        """The existing node whose prompt+completion is the LONGEST exact prefix of this prompt.

        Longest wins so a deep chain attaches to its immediate predecessor rather than to an early
        ancestor that also matches. Requiring an exact prefix (not a fuzzy match) is deliberate: a
        harness that mutates its history has genuinely produced a different sequence, and quietly
        attaching it would fabricate a trajectory the model never saw. Such a call becomes a new root
        instead, which `roots()` surfaces rather than hides.
        """
        best_id, best_len = None, 0
        for candidate_id in self._order:
            candidate = self._nodes[candidate_id]
            end = candidate.end_ids
            if len(end) > len(node.prompt_ids) or len(end) <= best_len:
                continue
            if common_prefix_len(end, node.prompt_ids) == len(end):
                best_id, best_len = candidate_id, len(end)
        return best_id

    # --- structure -----------------------------------------------------
    def nodes(self) -> list[TurnNode]:
        return [self._nodes[i] for i in self._order]

    def get(self, node_id: str) -> TurnNode:
        return self._nodes[node_id]

    def children(self, node_id: str) -> list[TurnNode]:
        return [self._nodes[i] for i in self._children.get(node_id, [])]

    def roots(self) -> list[TurnNode]:
        return [self._nodes[i] for i in self._order if self._nodes[i].parent_id is None]

    def leaves(self) -> list[TurnNode]:
        return [self._nodes[i] for i in self._order if not self._children.get(i)]

    def root_of(self, node_id: str) -> str:
        """Which conversation this node belongs to. Walks parents to the top."""
        node = self._nodes[node_id]
        while node.parent_id is not None:
            node = self._nodes[node.parent_id]
        return node.node_id

    def path_to(self, leaf_id: str) -> list[TurnNode]:
        path: list[TurnNode] = []
        node: TurnNode | None = self._nodes[leaf_id]
        while node is not None:
            path.append(node)
            node = self._nodes[node.parent_id] if node.parent_id else None
        return list(reversed(path))

    def paths(self) -> Iterator[list[TurnNode]]:
        """Every root-to-leaf path. One per distinct trajectory, including discarded branches."""
        for leaf in self.leaves():
            yield self.path_to(leaf.node_id)

    def forks(self) -> list[tuple[str, list[str]]]:
        """Nodes with more than one child: retries, resamples, or parallel branches."""
        return [(pid, kids) for pid, kids in self._children.items() if len(kids) > 1]

    def discarded_nodes(self) -> list[TurnNode]:
        """Sampled turns that led nowhere.

        A sibling with no children whose parent has another child that DID continue was generated and
        thrown away: a retry after a parse failure, or a resample. Training it with the rollout's
        reward credits work that never happened. Detected purely from shape, with no harness
        cooperation, which matters because a proxy is otherwise blind to retries.

        The final turn of a real trajectory is also childless, so a sibling is only called discarded
        when at least one of its siblings continued.
        """
        discarded: list[TurnNode] = []
        for _, kids in self.forks():
            continued = [k for k in kids if self._children.get(k)]
            if not continued:
                continue  # all siblings are terminal: ambiguous, keep them all
            discarded.extend(self._nodes[k] for k in kids if not self._children.get(k))
        return discarded

    # --- flattening ----------------------------------------------------
    def sequence_for(self, leaf_id: str) -> TrainingSequence:
        """Flatten one root-to-leaf path into token ids, mask and logprobs.

        Invariant enforced here rather than trusted: a turn whose logprobs are missing or misaligned
        contributes its tokens as CONTEXT (mask 0), never as targets. A trainable token without a
        real behaviour-policy logprob would make GRPO's importance ratio `exp(new - old)` a ratio
        against a number we invented.
        """
        path = self.path_to(leaf_id)
        input_ids: list[int] = []
        loss_mask: list[int] = []
        logprobs: list[float] = []
        prompt_len = 0

        parent: TurnNode | None = None
        for position, node in enumerate(path):
            context = node.context_ids(parent)
            input_ids.extend(context)
            loss_mask.extend([0] * len(context))
            logprobs.extend([0.0] * len(context))
            if position == 0:
                prompt_len = len(context)

            usable = node.sampled_logprobs is not None and len(
                node.sampled_logprobs
            ) == len(node.sampled_ids)
            input_ids.extend(node.sampled_ids)
            loss_mask.extend([1 if usable else 0] * len(node.sampled_ids))
            logprobs.extend(
                node.sampled_logprobs if usable else [0.0] * len(node.sampled_ids)
            )
            parent = node

        return TrainingSequence(
            input_ids=input_ids,
            loss_mask=loss_mask,
            logprobs=logprobs,
            node_ids=[n.node_id for n in path],
            prompt_len=prompt_len,
            root_id=path[0].node_id,
            n_turns=len(path),
        )

    def sequences(self) -> list[TrainingSequence]:
        return [self.sequence_for(leaf.node_id) for leaf in self.leaves()]

    def stats(self) -> dict[str, Any]:
        return {
            "n_turns": len(self._order),
            "n_roots": len(self.roots()),
            "n_leaves": len(self.leaves()),
            "n_forks": len(self.forks()),
            "n_discarded": len(self.discarded_nodes()),
        }
