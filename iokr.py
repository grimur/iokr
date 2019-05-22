import numpy


class DataStore(object):
    """
    This is probably outdated!
    """
    def __init__(self, kernel_matrix, latent_vectors):
        self.kernel_matrix = kernel_matrix
        # normalise the kernel matrix
        for i in range(self.kernel_matrix.shape[0]):
            for j in range(i + 1):
                self.kernel_matrix[i, j] = self.kernel_matrix[i, j] / numpy.sqrt(self.kernel_matrix[i, i] * self.kernel_matrix[j, j])
                self.kernel_matrix[j, i] = self.kernel_matrix[j, i] / numpy.sqrt(self.kernel_matrix[i, i] * self.kernel_matrix[j, j])
        self.latent_vectors = latent_vectors

        self.data_size, self.dimensions = latent_vectors.shape

    def get_latent_vector(self, index):
        return self.latent_vectors[index]

    def kernel_product(self, index_1, index_2):
        return self.kernel_matrix[index_1, index_2]

    def get_kernel_matrix(self, indices):
        return self.kernel_matrix[numpy.ix_(indices, indices)]

    def get_latent_vectors(self, indices):
        return self.latent_vectors[indices, :]

    def get_dimension(self):
        return self.dimensions


class InputOutputKernelRegression(object):
    def __init__(self, data):
        self.data = data

    def set_training_indices(self, indices=None, _lambda=0.1):
        self._lambda = _lambda
        if indices is None:
            indices = range(self.data.data_size)
        self.training_set = indices

    def fit(self):
        training_data_kernel = self.data.get_kernel_matrix(self.training_set)
        training_data_latent = self.data.get_latent_vectors(self.training_set)

        eye = numpy.eye(len(training_data_kernel))

        training_data_latent = numpy.array(training_data_latent).T

        latent_basis = numpy.linalg.inv(self._lambda * eye + training_data_kernel)
        self.latent_basis = latent_basis
        self.basis = numpy.dot(training_data_latent, latent_basis)

    def calculate_fingerprint_kernel_vector(self, fingerprint, kernel='gaussian'):
        if kernel == 'gaussian':
            def k(a, b, gamma=0.1):
                # kernel function
                d_sq = numpy.sum(numpy.power((a - b), 2))
                return numpy.exp(- gamma * d_sq)

            def k_vec(a_mat, b, gamma=0.1):
                # vectorised kernel function
                d_sq = numpy.sum(numpy.power((a_mat - b), 2), axis=1)
                return numpy.exp(- gamma * d_sq)

        training_data_latent = self.data.get_latent_vectors_vec(self.training_set)
        kernel_vector = k_vec(training_data_latent, fingerprint.T)
        return kernel_vector

    def project_candidate(self, index, fingerprint):
        fingerprint_kernel_vector = self.calculate_fingerprint_kernel_vector(fingerprint)
        x_kernel_vector = self.data.kernel_product_set(index, self.training_set)
        res = numpy.dot(numpy.dot(fingerprint_kernel_vector, self.latent_basis), x_kernel_vector)
        return res

    def rank_candidates(self, index, candidate_fingerprints):
        candidate_distances = [self.project_candidate(index, fingerprint) for fingerprint in candidate_fingerprints]
        return [x[1] for x in sorted(zip(candidate_distances, range(len(candidate_distances))), key=lambda x: x[0], reverse=True)]

    def project(self, index):
        x_kernel_vector = self.data.kernel_product_set(index, self.training_set)
        projection = numpy.dot(self.basis, x_kernel_vector)
        return projection

    def test(self, index, cutoff=0.01):
        proj = self.project(index)
        return [1 if x > cutoff else 0 for x in proj]


def main():
    target_vectors = numpy.array([[0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1]], dtype='float')
    repr_vectors = numpy.array([[0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 0, 1]], dtype='float')

    kernel_matrix = numpy.zeros((7, 7))
    for i in range(7):
        for j in range(i + 1):
            kernel_matrix[i, j] = kernel_matrix[j, i] = numpy.dot(repr_vectors[i], repr_vectors[j])

    data = DataStore(kernel_matrix, target_vectors)
    okr = InputOutputKernelRegression(data)
    okr.set_training_indices([0, 2, 3], _lambda=0.0)
    okr.fit()
    print(okr.test(0), target_vectors[0])
    print(okr.test(1), target_vectors[1])
    print(okr.test(2), target_vectors[2])
    print(okr.test(3), target_vectors[3])

if __name__ == '__main__':
    main()
