#!/usr/bin/env python3
"""Retained-branch B&B for maximum atomic score search.

For an exactly additive group score and an atomic lower bound ``L``,

    max(score_i in G) <= score(G) - (|G| - 1) * L.

Real validation-loss effects are not exactly additive. The controller can use
the empirical multiplicative envelope

    U(G) = (1 + epsilon) * score(G)                    (when L = 0),

plus additive interaction and measurement margins. Prunable nodes are kept in
a deferred heap. If a measured descendant reveals a larger
descendant/ancestor ratio, epsilon grows and deferred nodes are reconsidered.
Zero, negative, or numerically tiny group scores cannot support a ratio bound
and are conservatively expanded.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Callable, Generic, Iterable, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class GroupMeasurement:
    """One group query result and an optional caller-owned payload."""

    score: float
    payload: Any = None


@dataclass(frozen=True)
class SearchNode(Generic[T]):
    units: tuple[T, ...]
    measurement: GroupMeasurement
    upper_bound: float
    depth: int
    query_index: int
    lower_bound_violation: bool = False
    ancestor_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class BranchBoundResult(Generic[T]):
    best_unit: T | None
    best_score: float | None
    best_payload: Any
    group_queries: int
    expanded_groups: int
    pruned_groups: int
    pruned_units: int
    lower_bound_violations: int
    exhaustive: bool
    deferred_events: int = 0
    reopened_groups: int = 0
    reopened_units: int = 0
    epsilon_initial: float = 0.0
    epsilon_final: float = 0.0
    epsilon_updates: int = 0
    max_observed_ratio: float = 1.0
    nonpositive_group_expansions: int = 0


Measure = Callable[[tuple[T, ...], int, int], GroupMeasurement]
Splitter = Callable[[tuple[T, ...]], tuple[tuple[T, ...], tuple[T, ...]]]
EventHook = Callable[[str, SearchNode[T], float | None], None]


def balanced_split(units: tuple[T, ...]) -> tuple[tuple[T, ...], tuple[T, ...]]:
    """Split a non-singleton sequence into two non-empty balanced children."""

    if len(units) < 2:
        raise ValueError("cannot split a singleton or empty group")
    middle = (len(units) + 1) // 2
    return units[:middle], units[middle:]


def group_upper_bound(
    measured_sum: float,
    unit_count: int,
    *,
    leaf_lower_bound: float = 0.0,
    interaction_slack: float = 0.0,
    interaction_slack_per_unit: float = 0.0,
    measurement_margin: float = 0.0,
    epsilon: float = 0.0,
    expand_nonpositive: bool = False,
    epsilon_denominator_floor: float = 1e-12,
) -> tuple[float, bool]:
    """Compute a conservative maximum-singleton bound.

    With ``leaf_lower_bound=0`` and a positive group score, the core bound is
    exactly ``(1 + epsilon) * measured_sum``. A group that contradicts the
    configured atomic lower bound, or cannot safely be used as a denominator
    for the empirical ratio model, receives an infinite bound and is expanded.
    """

    if unit_count < 1:
        raise ValueError("unit_count must be positive")
    margins = (
        float(interaction_slack)
        + float(interaction_slack_per_unit) * unit_count
        + float(measurement_margin)
    )
    if margins < 0:
        raise ValueError("upper-bound margins must be non-negative")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if epsilon_denominator_floor < 0:
        raise ValueError("epsilon_denominator_floor must be non-negative")

    measured_sum = float(measured_sum)
    leaf_lower_bound = float(leaf_lower_bound)
    if unit_count == 1:
        return measured_sum + float(measurement_margin), False

    if expand_nonpositive and measured_sum <= epsilon_denominator_floor:
        return math.inf, False

    minimum_sum = unit_count * leaf_lower_bound
    violation = measured_sum + margins < minimum_sum
    if violation:
        return math.inf, True

    upper = (
        measured_sum
        - (unit_count - 1) * leaf_lower_bound
        + epsilon * max(measured_sum, 0.0)
        + margins
    )
    return upper, False


def maximize_with_branch_bound(
    units: Sequence[T] | Iterable[T],
    measure: Measure[T],
    *,
    split: Splitter[T] = balanced_split,
    leaf_lower_bound: float = 0.0,
    interaction_slack: float = 0.0,
    interaction_slack_per_unit: float = 0.0,
    measurement_margin: float = 0.0,
    initial_epsilon: float = 0.0,
    adaptive_epsilon: bool = True,
    epsilon_safety_factor: float = 2.0,
    epsilon_denominator_floor: float = 1e-12,
    initial_leaves: Sequence[tuple[T, GroupMeasurement]] = (),
    keep_ties: bool = False,
    on_event: EventHook[T] | None = None,
) -> BranchBoundResult[T]:
    """Find a maximum singleton with retained branches and adaptive epsilon.

    Both children are measured at expansion. The larger-bound child is
    traversed first. Nodes that currently cannot beat the singleton incumbent
    move to a deferred max-heap. Any epsilon increase recomputes their bounds
    and reopens those that can win under the enlarged envelope.
    """

    root_units = tuple(units)
    if initial_epsilon < 0:
        raise ValueError("initial_epsilon must be non-negative")
    if epsilon_safety_factor < 1:
        raise ValueError("epsilon_safety_factor must be at least one")
    if measurement_margin < 0:
        raise ValueError("measurement_margin must be non-negative")
    if epsilon_denominator_floor < 0:
        raise ValueError("epsilon_denominator_floor must be non-negative")
    if not root_units:
        return BranchBoundResult(
            best_unit=None,
            best_score=None,
            best_payload=None,
            group_queries=0,
            expanded_groups=0,
            pruned_groups=0,
            pruned_units=0,
            lower_bound_violations=0,
            exhaustive=True,
            epsilon_initial=float(initial_epsilon),
            epsilon_final=float(initial_epsilon),
        )

    query_count = 0
    expanded_groups = 0
    deferred_events = 0
    reopened_groups = 0
    reopened_units = 0
    lower_bound_violations = 0
    nonpositive_group_expansions = 0
    epsilon = float(initial_epsilon)
    epsilon_updates = 0
    max_observed_ratio = 1.0
    best_unit: T | None = None
    best_score: float | None = None
    best_payload: Any = None
    stack: list[SearchNode[T]] = []
    deferred: list[tuple[float, int, SearchNode[T]]] = []

    for unit, leaf in initial_leaves:
        if not isinstance(leaf, GroupMeasurement):
            raise TypeError("initial leaf values must be GroupMeasurement instances")
        score = float(leaf.score)
        if best_score is None or score > best_score:
            best_unit = unit
            best_score = score
            best_payload = leaf.payload

    def incumbent_lower() -> float | None:
        return None if best_score is None else best_score - measurement_margin

    def upper_for(measurement: GroupMeasurement, size: int) -> tuple[float, bool]:
        return group_upper_bound(
            measurement.score,
            size,
            leaf_lower_bound=leaf_lower_bound,
            interaction_slack=interaction_slack,
            interaction_slack_per_unit=interaction_slack_per_unit,
            measurement_margin=measurement_margin,
            epsilon=epsilon,
            expand_nonpositive=adaptive_epsilon or initial_epsilon > 0.0,
            epsilon_denominator_floor=epsilon_denominator_floor,
        )

    def refreshed(node: SearchNode[T]) -> SearchNode[T]:
        upper, violation = upper_for(node.measurement, len(node.units))
        return SearchNode(
            units=node.units,
            measurement=node.measurement,
            upper_bound=upper,
            depth=node.depth,
            query_index=node.query_index,
            lower_bound_violation=violation,
            ancestor_scores=node.ancestor_scores,
        )

    def is_prunable(node: SearchNode[T]) -> bool:
        lower = incumbent_lower()
        if lower is None:
            return False
        return node.upper_bound < lower if keep_ties else node.upper_bound <= lower

    def defer_node(node: SearchNode[T]) -> None:
        nonlocal deferred_events
        deferred_events += 1
        heapq.heappush(deferred, (-node.upper_bound, node.query_index, node))
        if on_event is not None:
            on_event("defer", node, best_score)

    def reconsider_deferred() -> None:
        nonlocal deferred, reopened_groups, reopened_units
        if not deferred:
            return
        saved = [entry[2] for entry in deferred]
        deferred = []
        reopened: list[SearchNode[T]] = []
        for saved_node in saved:
            node = refreshed(saved_node)
            if is_prunable(node):
                heapq.heappush(deferred, (-node.upper_bound, node.query_index, node))
                continue
            reopened.append(node)
            reopened_groups += 1
            reopened_units += len(node.units)
            if on_event is not None:
                on_event("reopen", node, best_score)
        reopened.sort(key=lambda item: (item.upper_bound, -item.query_index))
        stack.extend(reopened)

    def measured_node(
        group: tuple[T, ...],
        depth: int,
        parent: SearchNode[T] | None = None,
    ) -> SearchNode[T]:
        nonlocal query_count, lower_bound_violations
        nonlocal epsilon, epsilon_updates, max_observed_ratio
        nonlocal nonpositive_group_expansions

        if not group:
            raise ValueError("splitter returned an empty group")
        query_index = query_count
        result = measure(group, depth, query_index)
        query_count += 1
        if not isinstance(result, GroupMeasurement):
            raise TypeError("measure must return GroupMeasurement")

        ancestor_scores = () if parent is None else (
            parent.ancestor_scores + (float(parent.measurement.score),)
        )
        child_score = float(result.score)
        if adaptive_epsilon and child_score > epsilon_denominator_floor:
            for ancestor_score in ancestor_scores:
                if ancestor_score <= epsilon_denominator_floor:
                    continue
                ratio = child_score / ancestor_score
                max_observed_ratio = max(max_observed_ratio, ratio)
                required = max(0.0, ratio - 1.0) * epsilon_safety_factor
                if required > epsilon:
                    epsilon = required
                    epsilon_updates += 1

        upper, violation = upper_for(result, len(group))
        if violation:
            lower_bound_violations += 1
        if len(group) > 1 and not math.isfinite(upper) and not violation:
            nonpositive_group_expansions += 1
        node = SearchNode(
            units=group,
            measurement=result,
            upper_bound=upper,
            depth=depth,
            query_index=query_index,
            lower_bound_violation=violation,
            ancestor_scores=ancestor_scores,
        )
        if on_event is not None:
            on_event("measure", node, best_score)
        return node

    # Match the experiment: begin with two 20-layer half queries rather than
    # spending an additional query on the full 40-layer root.
    if len(root_units) == 1:
        stack = [measured_node(root_units, 0)]
    else:
        first, second = split(root_units)
        if not first or not second or len(first) + len(second) != len(root_units):
            raise ValueError("splitter must return two non-empty exhaustive children")
        roots = [measured_node(first, 1), measured_node(second, 1)]
        roots.sort(key=lambda item: (item.upper_bound, -item.query_index))
        stack = roots

    while stack:
        node = refreshed(stack.pop())
        if is_prunable(node):
            defer_node(node)
            continue

        if len(node.units) == 1:
            score = float(node.measurement.score)
            if best_score is None or score > best_score:
                best_unit = node.units[0]
                best_score = score
                best_payload = node.measurement.payload
                if on_event is not None:
                    on_event("incumbent", node, best_score)
            continue

        expanded_groups += 1
        first, second = split(node.units)
        if not first or not second or len(first) + len(second) != len(node.units):
            raise ValueError("splitter must return two non-empty exhaustive children")
        epsilon_before = epsilon
        children = [
            measured_node(first, node.depth + 1, node),
            measured_node(second, node.depth + 1, node),
        ]
        if epsilon > epsilon_before:
            stack[:] = [refreshed(saved) for saved in stack]
            reconsider_deferred()
            if on_event is not None:
                on_event("epsilon_update", refreshed(node), best_score)
        children = [refreshed(child) for child in children]
        # Low bound goes in first, so the high-bound child is popped first.
        children.sort(key=lambda item: (item.upper_bound, -item.query_index))
        stack.extend(children)

    pruned_nodes = [entry[2] for entry in deferred]
    pruned_groups = len(pruned_nodes)
    pruned_units = sum(len(node.units) for node in pruned_nodes)
    for node in pruned_nodes:
        if on_event is not None:
            on_event("prune", refreshed(node), best_score)

    return BranchBoundResult(
        best_unit=best_unit,
        best_score=best_score,
        best_payload=best_payload,
        group_queries=query_count,
        expanded_groups=expanded_groups,
        pruned_groups=pruned_groups,
        pruned_units=pruned_units,
        lower_bound_violations=lower_bound_violations,
        exhaustive=pruned_units == 0,
        deferred_events=deferred_events,
        reopened_groups=reopened_groups,
        reopened_units=reopened_units,
        epsilon_initial=float(initial_epsilon),
        epsilon_final=epsilon,
        epsilon_updates=epsilon_updates,
        max_observed_ratio=max_observed_ratio,
        nonpositive_group_expansions=nonpositive_group_expansions,
    )
