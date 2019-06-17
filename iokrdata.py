import scipy.io
import os
import numpy


# Hold the GNPS records
class GNPS(object):
    def __init__(self, filename):
        self.data_gnps = scipy.io.loadmat(filename)
        self.data_fp_array = numpy.array(self.data_gnps['fp'].todense())

    def get(self, index):
        # maybe add output type conversions (from sparse mat.)?
        fingerprint = self.data_fp_array[:, [index]]
        inchi = self.data_gnps['inchi'][index][0][0]
        formula = self.data_gnps['mf'][index][0][0]

        return inchi, formula, fingerprint

    def get_fingerprints(self, indices):
        return self.data_fp_array[:, indices].T


def load_folds(filename):
    with open(filename, 'r') as f:
        fold_ids = f.readlines()
    fold_ids = [x.strip() for x in fold_ids]
    return fold_ids


def load_kernel(filename):
    with open(filename, 'r') as f:
        raw = f.read().strip().split()

    data = numpy.array([float(x.strip()) for x in raw])
    dim = int(numpy.sqrt(len(data)))
    data = data.reshape((dim, dim))
    return data


def load_avg_kernel(path):
    kernel_files = ["ALIGND.txt",
                    "CP2Plus.txt",
                    "CPI.txt",
                    "CPK.txt",
                    "LB.txt",
                    "LI.txt",
                    "NI.txt",
                    "RLB.txt",
                    "ALIGN.txt",
                    "CP2.txt",
                    "CPJB.txt",
                    "CSC.txt",
                    "LC.txt",
                    "LW.txt",
                    "NSF.txt",
                    "RLI.txt",
                    "CEC.txt",
                    "CPC.txt",
                    "CPJ.txt",
                    "FIPP.txt",
                    "LIPP.txt",
                    "NB.txt",
                    "PPKr.txt",
                    "WPC.txt"]

    kernel = 0
    for i in kernel_files:
        kernel += load_kernel(path + os.sep + i)
    return kernel / len(kernel_files)


def load_candidates(path):
    candidate_sets = {}
    for file in os.listdir(path):
        candidate_name = extract_candidate_name(file)
        candidate_sets[candidate_name] = []
        if not file.endswith('mat'):
            continue
        data = scipy.io.loadmat(path + os.sep + file)
        num_candidates, _ = data['inchi'].shape
        for i in range(num_candidates):
            cand_inchi = data['inchi'][i, 0][0]
            cand_fingerprint = data['fp'][:, [i]]
            candidate_sets[candidate_name].append((cand_inchi, cand_fingerprint))

    return candidate_sets


def load_candidate_file(filename):
    candidates = []
    data = scipy.io.loadmat(filename)
    num_candidates, _ = data['inchi'].shape
    # fp_vectors = numpy.array(data['fp'].todense())
    fp_vectors = numpy.array(data['fp'].todense())
    for i in range(num_candidates):
        cand_inchi = data['inchi'][i, 0][0]
        # cand_fingerprint = data['fp'][:, i]
        cand_fingerprint = fp_vectors[:, i]
        candidates.append((cand_inchi, cand_fingerprint))
    return candidates


def extract_candidate_name(filename):
    return filename.split('.')[0].split('_')[-1]


def load_spectra(filename):
    spectra = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            if l.startswith('SPECTRUM_ID'):
                continue
            spectrum_id, compound, inchi = l.strip().split("\t")
            spectra.append((spectrum_id, compound, inchi))
    return spectra


# Holds the data that the IOKR can request.
# Needs to be central so we can get the kernel values for new samples
# TODO:
# - Also handle output kernel values (queriable in the same way?)
# - more flexible way of loading (input) kernels
# - Calclulate novel input kernels
class IOKRDataServer(object):
    def __init__(self, path, kernel=None):
        self.path = path
        self.gnps = GNPS(path + os.sep + 'data_GNPS.mat')
        # self.candidates = load_candidates(path + os.sep + 'candidates')
        self.spectra = load_spectra(path + os.sep + 'spectra.txt')

        self.folds = numpy.array(load_folds(path + os.sep + 'cv_ind.txt'))
        if kernel is None:
            print('No kernel specified. Please initialise manually.')
            self.kernel = None
        elif kernel == "avg":
            print('Loading average kernel.')
            self.kernel = load_avg_kernel(path + os.sep + 'input_kernels')
        else:
            print('Loading kernel %s' % kernel)
            self.kernel = load_kernel(path + os.sep + 'input_kernels' + os.sep + kernel)

        self.candidate_path = path + os.sep + 'candidates' + os.sep + 'candidate_set_%s.mat'

        self.dimension = None

    def get_candidates(self, formula):
        for candidate_inchi, candidate_fp in load_candidate_file(self.candidate_path % formula):
            yield candidate_inchi, candidate_fp

    def get_sample(self, idx, skip_candidates=False):
        gnps_inchi, formula, fingerprint = self.gnps.get(idx)
        if skip_candidates:
            candidates = []
        else:
            candidates = self.get_candidates(formula)
        spectrum_id, spectrum_name, spectrum_inchi = self.spectra[idx]

        assert(gnps_inchi == spectrum_inchi)
        return {'inchi': gnps_inchi,
                'formula': formula,
                'fingerprint': fingerprint,
                'candidates': candidates,
                'spectrum_id': spectrum_id,
                'spectrum_name': spectrum_name}

    def get_cv_set(self, label, complement=False):
        if label not in self.folds:
            return []
        if not complement:
            indices = numpy.where(self.folds == label)[0]
        else:
            indices = numpy.where(self.folds != label)[0]
        kernel_submatrix = self.kernel[numpy.ix_(indices, indices)]
        sample = [self.get_sample(x, skip_candidates=True) for x in indices]
        fp_matrix = numpy.hstack([x['fingerprint'] for x in sample]).T

        return kernel_submatrix, fp_matrix

    def get_latent_vector(self, index):
        return (self.get_sample(index)['fingerprint']).T

    def kernel_product(self, index_1, index_2):
        return self.kernel[index_1, index_2]

    def kernel_product_set(self, index_1, indices):
        return self.kernel[index_1, indices]

    def get_kernel_matrix(self, indices):
        return self.kernel[numpy.ix_(indices, indices)]

    def get_latent_vectors(self, indices):
        sample = [self.get_sample(x, skip_candidates=True) for x in indices]
        return numpy.hstack([x['fingerprint'] for x in sample]).T

    def get_latent_vectors_vec(self, indices):
        fingerprints = self.gnps.get_fingerprints(indices)
        return fingerprints

    def get_dimension(self):
        if self.dimension is None:
            self.dimension = len(self.get_sample(0, skip_candidates=True)['fingerprint'])
        return self.dimension

    def get_indices(self, label, complement=False):
        if complement:
            indices = numpy.where(self.folds != label)[0]
        else:
            indices = numpy.where(self.folds == label)[0]
        return indices


# datapath = "/home/grimur/iokr/data"
# iokrdata = IOKRDataServer(datapath, kernel='PPKr.txt')
