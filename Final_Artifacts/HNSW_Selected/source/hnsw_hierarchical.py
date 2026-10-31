"""Faithful custom hierarchical HNSW construction over centroid vectors."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HNSWIndex:
    centroids: np.ndarray
    adjacency: np.ndarray
    degrees: np.ndarray
    levels: np.ndarray
    entry_point: int
    entry_points: np.ndarray
    entry_levels: np.ndarray
    max_degree: int
    max_level: int
    ef_construction: int


def _d2(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.dot(diff, diff))


def _greedy(centroids: np.ndarray, adjacency: list[list[list[int]]], degrees: list[np.ndarray], current: int, target: int, layer: int) -> int:
    while True:
        best = current
        best_d2 = _d2(centroids[target], centroids[current])
        for neighbor in adjacency[layer][current][: int(degrees[layer][current])]:
            value = _d2(centroids[target], centroids[neighbor])
            if value < best_d2:
                best = int(neighbor)
                best_d2 = value
        if best == current:
            return current
        current = best


def _search_layer(centroids: np.ndarray, adjacency: list[list[list[int]]], degrees: list[np.ndarray], target: int, entry: int, layer: int, ef: int) -> list[int]:
    """HNSW best-first SearchLayer used during insertion."""
    target_vector = centroids[target]
    visited = {int(entry)}
    distance = {_id: _d2(target_vector, centroids[_id]) for _id in [int(entry)]}
    candidates: list[tuple[float, int]] = [(distance[int(entry)], int(entry))]
    results: list[tuple[float, int]] = [(-distance[int(entry)], int(entry))]

    while candidates:
        current_d2, current = heapq.heappop(candidates)
        worst_d2 = -results[0][0]
        if len(results) >= ef and current_d2 > worst_d2:
            break
        for neighbor in adjacency[layer][current][: int(degrees[layer][current])]:
            neighbor = int(neighbor)
            if neighbor in visited:
                continue
            visited.add(neighbor)
            neighbor_d2 = _d2(target_vector, centroids[neighbor])
            if len(results) < ef or neighbor_d2 < -results[0][0]:
                heapq.heappush(candidates, (neighbor_d2, neighbor))
                heapq.heappush(results, (-neighbor_d2, neighbor))
                if len(results) > ef:
                    heapq.heappop(results)
    return [node for _, node in sorted(((-d2, node) for d2, node in results))]


def _select_neighbors(centroids: np.ndarray, target: int, candidates: list[int], limit: int) -> list[int]:
    """HNSW diversity heuristic: retain neighbors covering distinct directions."""
    candidates = sorted(set(int(x) for x in candidates if int(x) != target), key=lambda x: _d2(centroids[target], centroids[x]))
    selected: list[int] = []
    for candidate in candidates:
        target_distance = _d2(centroids[target], centroids[candidate])
        diverse = all(_d2(centroids[candidate], centroids[chosen]) >= target_distance for chosen in selected)
        if diverse:
            selected.append(candidate)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= limit:
                break
    return selected[:limit]


def _prune_node(centroids: np.ndarray, node: int, neighbors: list[int], limit: int) -> list[int]:
    return _select_neighbors(centroids, node, neighbors, limit)


def build_hnsw(
    centroids: np.ndarray,
    *,
    m: int,
    ef_construction: int,
    seed: int = 20260915,
    max_degree_base: int | None = None,
    levels: np.ndarray | None = None,
    entry_count: int = 1,
) -> HNSWIndex:
    """Build a real multi-layer HNSW graph over centroid nodes."""
    centroids = np.ascontiguousarray(centroids.astype(np.float32))
    node_count = int(centroids.shape[0])
    m = max(2, int(m))
    m0 = max_degree_base or (2 * m)
    rng = np.random.default_rng(seed)
    if levels is None:
        level_scale = 1.0 / np.log(float(m))
        uniforms = np.clip(rng.random(node_count), np.finfo(np.float32).tiny, 1.0)
        levels = np.floor(-np.log(uniforms) * level_scale).astype(np.int32)
    else:
        levels = np.asarray(levels, dtype=np.int32)
    max_level = int(levels.max(initial=0))
    adjacency: list[list[list[int]]] = [[[] for _ in range(node_count)] for _ in range(max_level + 1)]
    degrees: list[np.ndarray] = [np.zeros(node_count, dtype=np.int32) for _ in range(max_level + 1)]
    entry = 0
    current_max_level = int(levels[0])
    insertion_order = np.arange(node_count, dtype=np.int32)
    rng.shuffle(insertion_order)
    inserted: set[int] = set()

    for raw_node in insertion_order:
        node = int(raw_node)
        node_level = int(levels[node])
        if not inserted:
            inserted.add(node)
            entry = node
            current_max_level = node_level
            continue

        ep = entry
        for layer in range(current_max_level, node_level, -1):
            ep = _greedy(centroids, adjacency, degrees, ep, node, layer)
        for layer in range(min(node_level, current_max_level), -1, -1):
            candidates = _search_layer(centroids, adjacency, degrees, node, ep, layer, max(ef_construction, m))
            limit = m0 if layer == 0 else m
            chosen = _select_neighbors(centroids, node, candidates, limit)
            adjacency[layer][node] = chosen
            degrees[layer][node] = len(chosen)
            for neighbor in chosen:
                reverse = _prune_node(centroids, neighbor, adjacency[layer][neighbor] + [node], m0 if layer == 0 else m)
                adjacency[layer][neighbor] = reverse
                degrees[layer][neighbor] = len(reverse)
            if candidates:
                ep = min(candidates, key=lambda x: _d2(centroids[node], centroids[int(x)]))
        inserted.add(node)
        if node_level > current_max_level:
            entry = node
            current_max_level = node_level

    max_degree = m0
    adjacency_array = np.full((max_level + 1, node_count, max_degree), -1, dtype=np.int32)
    degree_array = np.zeros((max_level + 1, node_count), dtype=np.int32)
    for layer in range(max_level + 1):
        for node in range(node_count):
            values = adjacency[layer][node][:max_degree]
            degree_array[layer, node] = len(values)
            if values:
                adjacency_array[layer, node, :len(values)] = values
    entry_count = max(1, min(int(entry_count), node_count))
    # Canonical HNSW still uses its single highest-level entry. When testing
    # the multi-entry extension, choose additional starts by farthest-point
    # sampling in centroid space, using level only to break equal-distance
    # ties. This avoids redundant probes clustered around the top-level node.
    ranked_entries = [entry]
    if entry_count > 1:
        selected_set = {entry}
        nearest_selected = np.sum((centroids - centroids[entry]) ** 2, axis=1)
        nearest_selected[entry] = -1.0
        while len(ranked_entries) < entry_count:
            best = max(
                (node for node in range(node_count) if node not in selected_set),
                key=lambda node: (float(nearest_selected[node]), int(levels[node]), -node),
            )
            ranked_entries.append(best)
            selected_set.add(best)
            d2 = np.sum((centroids - centroids[best]) ** 2, axis=1)
            nearest_selected = np.minimum(nearest_selected, d2)
            nearest_selected[np.asarray(ranked_entries, dtype=np.int32)] = -1.0
    entry_points = np.asarray(ranked_entries, dtype=np.int32)
    entry_levels = np.asarray([int(levels[node]) for node in entry_points], dtype=np.int32)
    return HNSWIndex(
        centroids=centroids,
        adjacency=np.ascontiguousarray(adjacency_array),
        degrees=np.ascontiguousarray(degree_array),
        levels=np.ascontiguousarray(levels),
        entry_point=int(entry),
        entry_points=np.ascontiguousarray(entry_points),
        entry_levels=np.ascontiguousarray(entry_levels),
        max_degree=int(max_degree),
        max_level=int(max_level),
        ef_construction=int(ef_construction),
    )


def _greedy_dynamic(centroids: np.ndarray, adjacency: list[list[list[int]]], current: int, target: int, layer: int) -> int:
    while True:
        best = current
        best_d2 = _d2(centroids[target], centroids[current])
        for neighbor in adjacency[layer][current]:
            value = _d2(centroids[target], centroids[neighbor])
            if value < best_d2:
                best, best_d2 = neighbor, value
        if best == current:
            return current
        current = best


def _search_layer_dynamic(
    centroids: np.ndarray,
    adjacency: list[list[list[int]]],
    target: int,
    entry: int,
    layer: int,
    ef: int,
) -> list[int]:
    target_vector = centroids[target]
    first_d2 = _d2(target_vector, centroids[entry])
    visited = {entry}
    candidates = [(first_d2, entry)]
    results = [(-first_d2, entry)]
    while candidates:
        current_d2, current = heapq.heappop(candidates)
        if len(results) >= ef and current_d2 > -results[0][0]:
            break
        for neighbor in adjacency[layer][current]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            distance = _d2(target_vector, centroids[neighbor])
            if len(results) < ef or distance < -results[0][0]:
                heapq.heappush(candidates, (distance, neighbor))
                heapq.heappush(results, (-distance, neighbor))
                if len(results) > ef:
                    heapq.heappop(results)
    return [node for _, node in sorted((-distance, node) for distance, node in results)]


def _diverse_entries(centroids: np.ndarray, levels: np.ndarray, entry_point: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    count = max(1, min(int(count), len(centroids)))
    selected = [int(entry_point)]
    if count > 1:
        nearest = np.sum((centroids - centroids[entry_point]) ** 2, axis=1)
        nearest[entry_point] = -1.0
        chosen = {int(entry_point)}
        while len(selected) < count:
            node = max(
                (i for i in range(len(centroids)) if i not in chosen),
                key=lambda i: (float(nearest[i]), int(levels[i]), -i),
            )
            selected.append(node)
            chosen.add(node)
            distance = np.sum((centroids - centroids[node]) ** 2, axis=1)
            nearest = np.minimum(nearest, distance)
            nearest[np.asarray(selected, dtype=np.int32)] = -1.0
    points = np.asarray(selected, dtype=np.int32)
    return points, np.asarray(levels[points], dtype=np.int32)


class IncrementalHNSWBuilder:
    """Insert centroid nodes into an existing HNSW without rebuilding old nodes."""

    def __init__(
        self,
        initial: HNSWIndex,
        *,
        m: int,
        ef_construction: int,
        max_degree_base: int,
        seed: int = 20260915,
        entry_count: int = 16,
    ) -> None:
        self.m = max(2, int(m))
        self.m0 = max(self.m, int(max_degree_base))
        self.ef_construction = max(int(ef_construction), self.m)
        if initial.max_degree != self.m0:
            raise ValueError("initial graph degree cap does not match max_degree_base")
        self.centroids = np.ascontiguousarray(initial.centroids.astype(np.float32))
        self.levels = initial.levels.astype(np.int32).tolist()
        self.max_level = int(initial.max_level)
        self.entry_point = int(initial.entry_point)
        self.entry_count = max(1, min(int(entry_count), len(self.centroids)))
        self.adjacency: list[list[list[int]]] = []
        for layer in range(self.max_level + 1):
            rows = []
            for node in range(len(self.centroids)):
                degree = int(initial.degrees[layer, node])
                rows.append(initial.adjacency[layer, node, :degree].astype(np.int32).tolist())
            self.adjacency.append(rows)
        self._level_rng = np.random.default_rng(int(seed) + 0x5F3759DF)

    def insert(self, centroid: np.ndarray, *, level: int | None = None) -> int:
        vector = np.ascontiguousarray(np.asarray(centroid, dtype=np.float32).reshape(1, -1))
        if vector.shape[1] != self.centroids.shape[1]:
            raise ValueError("new centroid dimension does not match the graph")
        if level is None:
            u = float(np.clip(self._level_rng.random(), np.finfo(np.float32).tiny, 1.0))
            level = int(np.floor(-np.log(u) / np.log(float(self.m))))
        node_level = max(0, int(level))
        node = len(self.centroids)
        self.centroids = np.ascontiguousarray(np.vstack((self.centroids, vector)))
        self.levels.append(node_level)
        while len(self.adjacency) <= node_level:
            self.adjacency.append([[] for _ in range(node)])
        for rows in self.adjacency:
            rows.append([])

        old_max_level = self.max_level
        ep = self.entry_point
        for layer in range(old_max_level, node_level, -1):
            ep = _greedy_dynamic(self.centroids, self.adjacency, ep, node, layer)
        for layer in range(min(node_level, old_max_level), -1, -1):
            candidates = _search_layer_dynamic(
                self.centroids, self.adjacency, node, ep, layer,
                max(self.ef_construction, self.m),
            )
            limit = self.m0 if layer == 0 else self.m
            chosen = _select_neighbors(self.centroids, node, candidates, limit)
            self.adjacency[layer][node] = chosen
            for neighbor in chosen:
                self.adjacency[layer][neighbor] = _prune_node(
                    self.centroids, neighbor, self.adjacency[layer][neighbor] + [node], limit
                )
            if candidates:
                ep = min(candidates, key=lambda other: _d2(self.centroids[node], self.centroids[other]))

        if node_level > old_max_level:
            self.entry_point = node
            self.max_level = node_level
        return node

    def snapshot(self, *, entry_count: int | None = None) -> HNSWIndex:
        node_count = len(self.centroids)
        max_degree = self.m0
        adjacency = np.full((self.max_level + 1, node_count, max_degree), -1, dtype=np.int32)
        degrees = np.zeros((self.max_level + 1, node_count), dtype=np.int32)
        for layer, rows in enumerate(self.adjacency):
            for node, neighbors in enumerate(rows):
                values = neighbors[:max_degree]
                degrees[layer, node] = len(values)
                if values:
                    adjacency[layer, node, :len(values)] = values
        levels = np.asarray(self.levels, dtype=np.int32)
        points, point_levels = _diverse_entries(
            self.centroids, levels, self.entry_point,
            self.entry_count if entry_count is None else entry_count,
        )
        return HNSWIndex(
            centroids=self.centroids.copy(),
            adjacency=adjacency,
            degrees=degrees,
            levels=levels,
            entry_point=self.entry_point,
            entry_points=points,
            entry_levels=point_levels,
            max_degree=max_degree,
            max_level=self.max_level,
            ef_construction=self.ef_construction,
        )


def save_hnsw(index: HNSWIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        centroids=index.centroids,
        adjacency=index.adjacency,
        degrees=index.degrees,
        levels=index.levels,
        entry_point=np.asarray([index.entry_point], dtype=np.int32),
        entry_points=index.entry_points,
        entry_levels=index.entry_levels,
        max_level=np.asarray([index.max_level], dtype=np.int32),
        ef_construction=np.asarray([index.ef_construction], dtype=np.int32),
    )
