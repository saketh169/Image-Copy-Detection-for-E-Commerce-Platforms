#include <cassert>
#include <iostream>

extern "C" {
void direct_l2_threshold_omp(
    const float*, int, const float*, int, int, float, float*, int*);
void hnsw_hierarchical_threshold_omp(
    const float*, int, const float*, int, int, const float*, const float*, int,
    const float*, const float*, const int*, const int*, const int*, int,
    const int*, const int*, const int*, const int*, int, int,
    const int*, const int*, int, int, int, float,
    int*, int*, int*, int*, double*, double*, double*, double*);
}

int main() {
    const float centroids[] = {0, 0, 1, 0, 2, 0, 3, 0};
    const float bank[] = {0, 0, 1, 0, 2, 0, 3, 0};
    const float queries[] = {3, 0, 10, 10};
    const float radii[] = {0, 0, 0, 0};
    const float bvh_lo[] = {0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 1, 0, 2, 0};
    const float bvh_hi[] = {3, 0, 1, 0, 3, 0, 3, 0, 0, 0, 1, 0, 2, 0};
    const int bvh_left[] = {1, 2, -1, -1, 5, -1, -1};
    const int bvh_right[] = {4, 3, -1, -1, 6, -1, -1};
    const int bvh_bucket[] = {-1, -1, 0, 1, -1, 2, 3};
    const int bucket_indices[] = {0, 1, 2, 3};
    const int offsets[] = {0, 1, 2, 3, 4};
    const int adjacency[] = {
        1, -1, 0, 2, 1, 3, 2, -1,
        3, -1, -1, -1, -1, -1, 0, -1};
    const int degrees[] = {1, 2, 2, 1, 1, 0, 0, 1};
    const int entry_points[] = {0, 3};
    const int entry_levels[] = {1, 0};

    float min_d2[2]{};
    int direct[2]{}, graph[2]{}, seen[2]{}, visited[2]{}, hops[2]{};
    double navigation_ns[2]{}, bucket_rank_ns[2]{}, verify_ns[2]{}, certificate_ns[2]{};
    direct_l2_threshold_omp(
        queries, 2, bank, 4, 2, 0.25f, min_d2, direct);
    hnsw_hierarchical_threshold_omp(
        queries, 2, bank, 4, 2, centroids, radii, 4,
        bvh_lo, bvh_hi, bvh_left, bvh_right, bvh_bucket, 7,
        bucket_indices, offsets, adjacency, degrees, 2, 2,
        entry_points, entry_levels, 2, 2, 8, 0.25f,
        graph, seen, visited, hops,
        navigation_ns, bucket_rank_ns, verify_ns, certificate_ns);

    assert(direct[0] == 1 && direct[1] == 0);
    assert(graph[0] == direct[0] && graph[1] == direct[1]);
    for (int i = 0; i < 2; ++i) {
        assert(navigation_ns[i] >= 0.0);
        assert(bucket_rank_ns[i] >= 0.0);
        assert(verify_ns[i] >= 0.0);
        assert(certificate_ns[i] >= 0.0);
    }
    std::cout << "HNSW/direct exact-decision smoke test passed\n";
}
