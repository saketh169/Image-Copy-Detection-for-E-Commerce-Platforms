#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using Node = std::pair<float, int>;

inline float l2sq(const float* a, const float* b, int dim) {
    float acc = 0.0f;
    #pragma omp simd reduction(+:acc)
    for (int d = 0; d < dim; ++d) {
        const float x = a[d] - b[d];
        acc += x * x;
    }
    return acc;
}

inline float lower_bound(float centroid_d2, float radius) {
    const float delta = std::sqrt(std::max(centroid_d2, 0.0f)) - radius;
    return delta <= 0.0f ? 0.0f : delta * delta;
}

inline float conservative_radius_lower_bound(float centroid_d2, float radius) {
    const float distance = std::sqrt(std::max(centroid_d2, 0.0f));
    const float scale = std::max(1.0f, std::max(distance, radius));
    const float slack = 32.0f * std::numeric_limits<float>::epsilon() * scale;
    const float delta = distance - radius - slack;
    if (delta <= 0.0f) return 0.0f;
    return std::nextafter(delta * delta, 0.0f);
}

inline float aabb_lower_bound(const float* q, const float* lo, const float* hi, int dim) {
    float acc = 0.0f;
    #pragma omp simd reduction(+:acc)
    for (int d = 0; d < dim; ++d) {
        const float delta = q[d] < lo[d] ? lo[d] - q[d] : (q[d] > hi[d] ? q[d] - hi[d] : 0.0f);
        acc += delta * delta;
    }
    const float slack = 16.0f * std::numeric_limits<float>::epsilon() * std::max(1.0f, acc);
    return std::max(0.0f, acc - slack);
}
}

extern "C" {

void direct_l2_threshold_omp(
    const float* queries, int nq, const float* bank, int nb, int dim,
    float threshold2, float* out_min_d2, int* out_dup
) {
    #pragma omp parallel for schedule(static)
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = queries + static_cast<long long>(qi) * dim;
        float best = std::numeric_limits<float>::infinity();
        for (int bi = 0; bi < nb; ++bi) {
            const float d2 = l2sq(q, bank + static_cast<long long>(bi) * dim, dim);
            if (d2 < best) best = d2;
        }
        out_min_d2[qi] = best;
        out_dup[qi] = best <= threshold2 ? 1 : 0;
    }
}

void hnsw_hierarchical_threshold_omp(
    const float* queries, int nq, const float* bank, int nb, int dim,
    const float* centroids, const float* radii, int nc,
    const float* bvh_lo, const float* bvh_hi, const int* bvh_left,
    const int* bvh_right, const int* bvh_bucket, int bvh_nodes,
    const float* point_lo, const float* point_hi, const int* point_left,
    const int* point_right, const int* point_start, const int* point_end,
    const int* point_order, int point_nodes,
    const int* bucket_indices, const int* offsets,
    const int* adjacency, const int* degrees, int layer_count, int max_degree,
    const int* entry_points, const int* entry_levels, int entry_count,
    int ef_search, int bucket_limit, float threshold2,
    int* out_dup, int* out_seen, int* out_visited, int* out_hops,
    double* out_navigation_ns, double* out_bucket_rank_ns,
    double* out_verification_ns, double* out_certificate_ns
) {
    const bool profile = out_navigation_ns && out_bucket_rank_ns &&
        out_verification_ns && out_certificate_ns;
    #pragma omp parallel
    {
        std::vector<int> centroid_marks(nc, 0), image_marks(nb, 0);
        std::vector<int> search_marks(nc, 0);
        std::vector<float> centroid_d2(nc, 0.0f);
        std::vector<int> distance_marks(nc, 0);
        std::vector<Node> proof_candidates;
        proof_candidates.reserve(nc);
        int centroid_stamp = 1, image_stamp = 1, distance_stamp = 1, search_stamp = 1;

        #pragma omp for schedule(static)
        for (int qi = 0; qi < nq; ++qi) {
            if (++centroid_stamp == 2147483000) { std::fill(centroid_marks.begin(), centroid_marks.end(), 0); centroid_stamp = 1; }
            if (++image_stamp == 2147483000) { std::fill(image_marks.begin(), image_marks.end(), 0); image_stamp = 1; }
            if (++distance_stamp == 2147483000) { std::fill(distance_marks.begin(), distance_marks.end(), 0); distance_stamp = 1; }
            const float* q = queries + static_cast<long long>(qi) * dim;

            auto cd2 = [&](int c) {
                if (distance_marks[c] != distance_stamp) {
                    centroid_d2[c] = l2sq(q, centroids + static_cast<long long>(c) * dim, dim);
                    distance_marks[c] = distance_stamp;
                }
                return centroid_d2[c];
            };
            auto mark_centroid = [&](int c) {
                if (centroid_marks[c] != centroid_stamp) {
                    centroid_marks[c] = centroid_stamp;
                    return 1;
                }
                return 0;
            };

            int visited = 0;
            int hops = 0;
            const auto navigation_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

            std::vector<Node> merged_results;
            const int starts = std::max(1, entry_count);
            for (int probe = 0; probe < starts; ++probe) {
                int current = entry_points[probe];
                if (current < 0 || current >= nc) current = 0;
                const int start_level = std::min(layer_count - 1, std::max(0, entry_levels[probe]));
                visited += mark_centroid(current);

                // Each probe performs the canonical top-down HNSW descent
                // from the highest layer where its start node exists.
                for (int layer = start_level; layer > 0; --layer) {
                    bool improved = true;
                    while (improved) {
                        improved = false;
                        const int row = (layer * nc + current) * max_degree;
                        const int deg = degrees[layer * nc + current];
                        const float here = cd2(current);
                        int best = current;
                        float best_d2 = here;
                        for (int k = 0; k < deg; ++k) {
                            const int n = adjacency[row + k];
                            if (n < 0 || n >= nc) continue;
                            const float nd2 = cd2(n);
                            if (nd2 < best_d2) { best = n; best_d2 = nd2; }
                        }
                        if (best != current) {
                            current = best;
                            visited += mark_centroid(current);
                            ++hops;
                        }
                    }
                }

                // Layer zero uses bounded best-first HNSW search. Results
                // from all starts are merged before bucket ranking. Each
                // entry point needs its own SearchLayer visited set; sharing
                // centroid_marks here would suppress valid exploration from
                // later probes.
                if (++search_stamp == 2147483000) {
                    std::fill(search_marks.begin(), search_marks.end(), 0);
                    search_stamp = 1;
                }
                std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
                std::priority_queue<Node> results;
                const float start = cd2(current);
                candidates.push({start, current});
                results.push({start, current});
                search_marks[current] = search_stamp;
                while (!candidates.empty()) {
                    const Node item = candidates.top();
                    candidates.pop();
                    if (static_cast<int>(results.size()) >= ef_search && item.first > results.top().first) break;
                    const int c = item.second;
                    const int row = c * max_degree;
                    const int deg = degrees[c];
                    for (int k = 0; k < deg; ++k) {
                        const int n = adjacency[row + k];
                        if (n < 0 || n >= nc || search_marks[n] == search_stamp) continue;
                        search_marks[n] = search_stamp;
                        visited += mark_centroid(n);
                        const float nd2 = cd2(n);
                        if (static_cast<int>(results.size()) < ef_search || nd2 < results.top().first) {
                            candidates.push({nd2, n});
                            results.push({nd2, n});
                            if (static_cast<int>(results.size()) > ef_search) results.pop();
                        }
                    }
                }
                while (!results.empty()) {
                    merged_results.push_back(results.top());
                    results.pop();
                }
            }

            const auto bucket_rank_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            std::vector<Node> bucket_candidates;
            for (const Node& item : merged_results) {
                bucket_candidates.push_back({lower_bound(item.first, radii[item.second]), item.second});
            }
            std::sort(bucket_candidates.begin(), bucket_candidates.end(),
                [](const Node& a, const Node& b) { return a.second < b.second; });
            bucket_candidates.erase(
                std::unique(bucket_candidates.begin(), bucket_candidates.end(),
                    [](const Node& a, const Node& b) { return a.second == b.second; }),
                bucket_candidates.end());
            std::sort(bucket_candidates.begin(), bucket_candidates.end());
            const auto verification_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

            int seen = 0;
            int duplicate = 0;
            int checked_buckets = 0;
            for (const Node& bucket : bucket_candidates) {
                if (checked_buckets >= bucket_limit || bucket.first > threshold2) break;
                ++checked_buckets;
                const int c = bucket.second;
                for (int pos = offsets[c]; pos < offsets[c + 1]; ++pos) {
                    const int bi = bucket_indices[pos];
                    if (bi < 0 || bi >= nb || image_marks[bi] == image_stamp) continue;
                    image_marks[bi] = image_stamp;
                    ++seen;
                    if (l2sq(q, bank + static_cast<long long>(bi) * dim, dim) <= threshold2) {
                        duplicate = 1;
                        break;
                    }
                }
                if (duplicate) break;
            }
            const auto certificate_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

            // A graph route only orders likely buckets. An unresolved negative
            // needs an exact certificate.  The point BVH gives that proof at a
            // much finer granularity than an entire centroid bucket: its leaves
            // hold small groups of bank vectors and an AABB is rejected only
            // when it is provably farther than the threshold.
            if (!duplicate) {
                std::priority_queue<Node, std::vector<Node>, std::greater<Node>> proof_nodes;
                const float root_bound = aabb_lower_bound(q, point_lo, point_hi, dim);
                if (root_bound <= threshold2) proof_nodes.push({root_bound, 0});
                while (!proof_nodes.empty() && !duplicate) {
                    const Node node = proof_nodes.top();
                    proof_nodes.pop();
                    if (node.first > threshold2) break;
                    const int node_id = node.second;
                    const int start = point_start[node_id];
                    if (start < 0) {
                        const int left = point_left[node_id];
                        const int right = point_right[node_id];
                        if (left >= 0) {
                            const float bound = aabb_lower_bound(q, point_lo + static_cast<long long>(left) * dim,
                                point_hi + static_cast<long long>(left) * dim, dim);
                            if (bound <= threshold2) proof_nodes.push({bound, left});
                        }
                        if (right >= 0) {
                            const float bound = aabb_lower_bound(q, point_lo + static_cast<long long>(right) * dim,
                                point_hi + static_cast<long long>(right) * dim, dim);
                            if (bound <= threshold2) proof_nodes.push({bound, right});
                        }
                        continue;
                    }
                    for (int pos = start; pos < point_end[node_id]; ++pos) {
                        const int bi = point_order[pos];
                        if (bi < 0 || bi >= nb || image_marks[bi] == image_stamp) continue;
                        image_marks[bi] = image_stamp;
                        ++seen;
                        if (l2sq(q, bank + static_cast<long long>(bi) * dim, dim) <= threshold2) {
                            duplicate = 1;
                            break;
                        }
                    }
                }
            }
            const auto query_end = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            out_dup[qi] = duplicate;
            out_seen[qi] = seen;
            out_visited[qi] = visited;
            out_hops[qi] = hops;
            if (profile) {
                out_navigation_ns[qi] = std::chrono::duration<double, std::nano>(bucket_rank_start - navigation_start).count();
                out_bucket_rank_ns[qi] = std::chrono::duration<double, std::nano>(verification_start - bucket_rank_start).count();
                out_verification_ns[qi] = std::chrono::duration<double, std::nano>(certificate_start - verification_start).count();
                out_certificate_ns[qi] = std::chrono::duration<double, std::nano>(query_end - certificate_start).count();
            }
        }
    }
}

void erng_flat_threshold_omp(
    const float* queries, int nq, const float* bank, int nb, int dim,
    const float* centroids, const float* radii, int nc,
    const int* bucket_indices, const int* offsets,
    const int* adjacency, const int* degrees, int max_degree,
    const int* starts, int probes, int max_hops, int bucket_limit,
    float threshold2, int* out_dup, int* out_seen, int* out_visited, int* out_hops
) {
    #pragma omp parallel
    {
        std::vector<int> centroid_marks(nc, 0), image_marks(nb, 0);
        std::vector<float> centroid_d2(nc, 0.0f);
        std::vector<int> distance_marks(nc, 0);
        int centroid_stamp = 1, image_stamp = 1, distance_stamp = 1;

        #pragma omp for schedule(static)
        for (int qi = 0; qi < nq; ++qi) {
            if (++centroid_stamp == 2147483000) { std::fill(centroid_marks.begin(), centroid_marks.end(), 0); centroid_stamp = 1; }
            if (++image_stamp == 2147483000) { std::fill(image_marks.begin(), image_marks.end(), 0); image_stamp = 1; }
            if (++distance_stamp == 2147483000) { std::fill(distance_marks.begin(), distance_marks.end(), 0); distance_stamp = 1; }
            const float* q = queries + static_cast<long long>(qi) * dim;

            auto cd2 = [&](int c) {
                if (distance_marks[c] != distance_stamp) {
                    centroid_d2[c] = l2sq(q, centroids + static_cast<long long>(c) * dim, dim);
                    distance_marks[c] = distance_stamp;
                }
                return centroid_d2[c];
            };
            auto mark_centroid = [&](int c) {
                if (centroid_marks[c] != centroid_stamp) {
                    centroid_marks[c] = centroid_stamp;
                    return 1;
                }
                return 0;
            };

            std::vector<Node> endpoints;
            int visited = 0;
            int total_hops = 0;
            for (int p = 0; p < probes; ++p) {
                int current = starts[static_cast<long long>(qi) * probes + p];
                if (current < 0 || current >= nc) current = (qi + p) % nc;
                visited += mark_centroid(current);
                int hops = 0;
                for (; hops < max_hops; ++hops) {
                    const int row = current * max_degree;
                    const int deg = degrees[current];
                    int best = current;
                    float best_d2 = cd2(current);
                    for (int k = 0; k < deg; ++k) {
                        const int n = adjacency[row + k];
                        if (n < 0 || n >= nc) continue;
                        const float nd2 = cd2(n);
                        if (nd2 < best_d2) { best = n; best_d2 = nd2; }
                    }
                    if (best == current) break;
                    current = best;
                    visited += mark_centroid(current);
                }
                total_hops += hops;
                endpoints.push_back({cd2(current), current});
            }
            std::sort(endpoints.begin(), endpoints.end());

            std::vector<Node> bucket_candidates;
            for (const Node& endpoint : endpoints) {
                bool duplicate_endpoint = false;
                for (const Node& kept : bucket_candidates) {
                    if (kept.second == endpoint.second) { duplicate_endpoint = true; break; }
                }
                if (!duplicate_endpoint) {
                    bucket_candidates.push_back({lower_bound(endpoint.first, radii[endpoint.second]), endpoint.second});
                }
                if (static_cast<int>(bucket_candidates.size()) >= bucket_limit) break;
            }
            std::sort(bucket_candidates.begin(), bucket_candidates.end());

            int seen = 0;
            int duplicate = 0;
            int checked_buckets = 0;
            for (const Node& bucket : bucket_candidates) {
                if (checked_buckets >= bucket_limit || bucket.first > threshold2) break;
                ++checked_buckets;
                const int c = bucket.second;
                for (int pos = offsets[c]; pos < offsets[c + 1]; ++pos) {
                    const int bi = bucket_indices[pos];
                    if (bi < 0 || bi >= nb || image_marks[bi] == image_stamp) continue;
                    image_marks[bi] = image_stamp;
                    ++seen;
                    if (l2sq(q, bank + static_cast<long long>(bi) * dim, dim) <= threshold2) {
                        duplicate = 1;
                        break;
                    }
                }
                if (duplicate) break;
            }

            // Exact negative certificate; see the HNSW branch above.  ERNG
            // supplies the ordering, while centroid-radius bounds guarantee
            // that no viable bucket is omitted from the final decision.
            if (!duplicate) {
                std::vector<Node> proof_buckets;
                proof_buckets.reserve(nc);
                for (int c = 0; c < nc; ++c) {
                    const float bound = lower_bound(cd2(c), radii[c]);
                    if (bound <= threshold2) proof_buckets.push_back({bound, c});
                }
                std::sort(proof_buckets.begin(), proof_buckets.end());
                for (const Node& bucket : proof_buckets) {
                    const int c = bucket.second;
                    for (int pos = offsets[c]; pos < offsets[c + 1]; ++pos) {
                        const int bi = bucket_indices[pos];
                        if (bi < 0 || bi >= nb || image_marks[bi] == image_stamp) continue;
                        image_marks[bi] = image_stamp;
                        ++seen;
                        if (l2sq(q, bank + static_cast<long long>(bi) * dim, dim) <= threshold2) {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (duplicate) break;
                }
            }
            out_dup[qi] = duplicate;
            out_seen[qi] = seen;
            out_visited[qi] = visited;
            out_hops[qi] = total_hops;
        }
    }
}

}
