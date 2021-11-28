__kernel void add_raindrops(__global double* buffer, __global double* centers,
    __global int* done, const int K, const int dimension, const int num_operations) {
    int point_id = get_global_id(0);
    int old_cluster_id = round(points[point_id * (dimension + 1) + dimension]);
    int new_cluster_id = 0;
    double shortest_dist_sqr = 0.0;

    // calculate shortest distance to point_{id} from all centers
    if (point_id < num_operations) {
        for (int k = 0; k < K; k++) {
            double sum = 0.0;

            for (int j = 0; j < dimension; j++) {
                sum += pow(centers[k * dimension + j] - points[point_id * (dimension + 1) + j], 2.0);
            }
            if (k == 0 || sum < shortest_dist_sqr) {
                shortest_dist_sqr = sum;
                new_cluster_id = k;
            }
        }
        if (new_cluster_id != old_cluster_id) {
            points[point_id * (dimension + 1) + dimension] = new_cluster_id;
            done[0] = 0;
        }
    }
}