import numpy as np
import matplotlib.pyplot as plt


# TODO
# 1 ) fill-in the missing code for calculate_distance and calculate_loss
# 2 ) optimize k-means (within k_means)
# 3 ) complete function k_means_v2

# Useful functions
# np.sum(x)
# np.sqrt(x)
# np.array_equal(x, y)
# x ** 2


def initialize_random_centroids(training_data, n_clusters):
    index_array = np.arange(len(training_data))
    np.random.shuffle(index_array)
    random_indices_to_use = index_array[:n_clusters]
    return np.take(training_data, random_indices_to_use, axis=0)


def calculate_euclidean_distance(a, b):
    '''
    :param a: numpy ndarray
    :param b: numpy ndarray (same dimension as a)
    :return: return a float representing the different between a and b
    '''
    diff_value =  # Your Code Here
    squared_value =  # Your Code Here
    sum_value =  # Your Code Here
    sqrt_value =  # Your Code Here
    return sqrt_value


def find_nearest_centroid(point, centroids):
    shortest_distance = -1
    centroid_index = -1
    for index, centroid in enumerate(centroids):
        distance = calculate_euclidean_distance(point, centroid)
        if shortest_distance < 0 or shortest_distance > distance:
            centroid_index = index
            shortest_distance = distance
    return centroid_index


def assign_to_nearest(training_data, centroids):
    labels = []
    for data in training_data:
        labels.append(find_nearest_centroid(data, centroids))
    return np.array(labels)


def update_centroids(training_data, labels, n_clusters):
    new_centroids = []
    for i in range(n_clusters):
        indices = np.where(labels == i)[0]
        data_in_centroid = np.take(training_data, indices, axis=0)
        new_centroid = np.average(data_in_centroid, axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)


def calculate_loss(training_data, centroids, labels):
    '''
    :param training_data: numpy ndarray containing training data
    :param centroids: numpy ndarray containing the center positions
    :param labels: numpy ndarray size of (len(training_data), 1) - index relates to training_data, value relates to the
    cluster group (centroid index) - [0, 2, 0, 1]
    :return:
    '''
    loss = 0
    for index, group in enumerate(labels):
        diff_value =  # Your Code Here
        squared_value =  # Your Code Here
        summ_value =  # Your Code Here
        loss +=  # Your Code Here
    return loss


def k_means(training_data, n_clusters, max_iterations, include_plot=True):
    centroids = initialize_random_centroids(training_data, n_clusters)
    labels = None
    loss = -1

    if include_plot:
        plt.scatter(training_data[:, :-1], training_data[:, -1])
        plt.scatter(centroids[:, :-1], centroids[:, -1])
        plt.title('Random Centroids')
        plt.show()

    for step in range(max_iterations):
        new_labels = assign_to_nearest(training_data, centroids)
        new_centroids = update_centroids(training_data, new_labels, n_clusters)
        loss = calculate_loss(training_data, new_centroids, new_labels)
        # Your Code Here
        centroids = new_centroids
        labels = new_labels
        print('Step:{} Loss: {}'.format(step, loss))

    if include_plot:
        plt.scatter(training_data[:, :-1], training_data[:, -1])
        plt.scatter(centroids[:, :-1], centroids[:, -1])
        plt.title('Clustered Centroids')
        plt.show()

    return centroids, labels, loss


def k_means_v2(training_data, n_clusters, max_iterations, k_means_attempts, include_plot=True):
    '''
      :param training_data: numpy matrix; n records by m features
      :param n_clusters: number of clusters to use
      :param max_iterations: not used, a relative_tolerance is used instead of setting a max iteration, not sure I like that
      :param k_means_attempts: number of times to run k-means (returns cluster with lowest inertia)
      :param include_plot: plot data (only for 2-d data)
    '''
    centroids = None
    labels = None
    loss = -1
    best_run = -1
    # Your Code Here
    print('Best run was run number {}'.format(best_run))
    return centroids, labels, loss


def training_set_one(points_per_cen):
    x_1 = np.random.uniform(low=8, high=10, size=points_per_cen)
    y_1 = np.random.uniform(low=0, high=2, size=points_per_cen)

    x_2 = np.random.uniform(low=3, high=5, size=points_per_cen)
    y_2 = np.random.uniform(low=8, high=10, size=points_per_cen)

    x_3 = np.random.uniform(low=0, high=3, size=points_per_cen)
    y_3 = np.random.uniform(low=1, high=2, size=points_per_cen)
    training_data = np.concatenate(
        (np.column_stack((x_1, y_1)), np.column_stack((x_2, y_2)), np.column_stack((x_3, y_3))))
    return training_data


def training_set_two(points_per_cen):
    x_1 = np.random.uniform(low=8, high=10, size=points_per_cen)
    y_1 = np.random.uniform(low=0, high=2, size=points_per_cen)

    x_2 = np.random.uniform(low=3, high=5, size=points_per_cen)
    y_2 = np.random.uniform(low=8, high=10, size=points_per_cen)

    x_3 = np.random.uniform(low=0, high=3, size=points_per_cen)
    y_3 = np.random.uniform(low=1, high=2, size=points_per_cen)

    x_4 = np.random.uniform(low=1, high=2, size=points_per_cen)
    y_4 = np.random.uniform(low=3, high=4, size=points_per_cen)
    training_data = np.concatenate((np.column_stack((x_1, y_1)), np.column_stack((x_2, y_2)),
                                    np.column_stack((x_3, y_3)), np.column_stack((x_4, y_4))))
    return training_data


def training_set_three(points_per_cen):
    x_1 = np.random.uniform(low=8, high=10, size=points_per_cen)
    y_1 = np.random.uniform(low=0, high=2, size=points_per_cen)
    z_1 = np.random.uniform(low=0, high=2, size=points_per_cen)

    x_2 = np.random.uniform(low=3, high=5, size=points_per_cen)
    y_2 = np.random.uniform(low=8, high=10, size=points_per_cen)
    z_2 = np.random.uniform(low=8, high=10, size=points_per_cen)

    x_3 = np.random.uniform(low=0, high=3, size=points_per_cen)
    y_3 = np.random.uniform(low=1, high=2, size=points_per_cen)
    z_3 = np.random.uniform(low=1, high=2, size=points_per_cen)

    x_4 = np.random.uniform(low=1, high=2, size=points_per_cen)
    y_4 = np.random.uniform(low=3, high=4, size=points_per_cen)
    z_4 = np.random.uniform(low=3, high=4, size=points_per_cen)
    training_data = np.concatenate((np.column_stack((x_1, y_1, z_1)), np.column_stack((x_2, y_2, z_2)),
                                    np.column_stack((x_3, y_3, z_3)), np.column_stack((x_4, y_4, z_4))))
    return training_data


print('Training Set 1:')
training_data = training_set_one(50)
np.random.shuffle(training_data)
k_means(training_data, 3, 10)

print('\nTraining Set 2:')
training_data = training_set_two(50)
np.random.shuffle(training_data)
k_means_v2(training_data, 4, 10, 3)

print('\nTraining Set 3:')
training_data = training_set_two(500)
np.random.shuffle(training_data)
k_means_v2(training_data, 4, 10, 3, False)