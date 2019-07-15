import multiprocessing
import spectrum_filters
import iokrdata as data
import scipy
import numpy
import os
import itertools
from numba import jit

import time

def _ppk(i_peaks, j_peaks, sm, si):
        X1 = i_peaks
        X2 = j_peaks
        # N1 = numpy.size(X1, 0); N2 = numpy.size(X2, 0)
        N1 = X1.shape[0]
        N2 = X2.shape[0]
        if N1 == 0 or N2 == 0:
            raise Exception("[ERROR]:No peaks when computing the kernel.(try not clean the peaks)")
        constant = 1.0/(N1*N2)*0.25/(numpy.pi*numpy.sqrt(sm*si))
        mass_term = 1.0/sm * numpy.power(numpy.kron(X1[:, 0].flatten(), numpy.ones(N2)) - numpy.kron(numpy.ones(N1), X2[:, 0].flatten()), 2)
        inte_term = 1.0/si * numpy.power(numpy.kron(X1[:, 1].flatten(), numpy.ones(N2)) - numpy.kron(numpy.ones(N1), X2[:, 1].flatten()), 2)
        return constant*numpy.sum(numpy.exp(-0.25*(mass_term + inte_term)))


def ppk(*args):
    # t0 = time.time()
    # a = ppk_loop(*args)
    # t1 = time.time()
    try:
        b = ppk_limit(*args)
    except ValueError:
        print(args)
        raise
    # t2 = time.time()
    # print(t1-t0, t2-t1, a-b/a)
    return b


@jit(nopython=True)
def ppk_loop(spectrum_1, spectrum_2, sigma_mass, sigma_int):
    # the inputs are really sigma^2, though
    # sigma_mass = 0.00001
    # sigma_int = 100000
    sigma_array = numpy.array([[sigma_mass, 0], [0, sigma_int]])
    sigma_inv = numpy.linalg.inv(sigma_array)
    len_1 = spectrum_1.shape[0]
    len_2 = spectrum_2.shape[0]
    constant_term = 1.0 / (len_1 * len_2 * 4 * numpy.pi * numpy.sqrt(sigma_mass * sigma_int))
    sum_term = 0
    # for p_1, p_2 in itertools.product(spectrum_1, spectrum_2):
    for p_1_idx in range(len_1):
        p_1 = spectrum_1[p_1_idx, :]
        for p_2_idx in range(len_2):
            p_2 = spectrum_2[p_2_idx, :]
            d = p_1 - p_2
            sum_term += numpy.exp(-0.25 * numpy.sum(d * sigma_inv * d))
    # print(sum_term)
    # print(numpy.sum(sum_term))
    return constant_term * sum_term


@jit(nopython=True)
def ppk_limit(spectrum_1, spectrum_2, sigma_mass, sigma_int):
    # the inputs are really sigma^2, though
    # sigma_mass = 0.00001
    # sigma_int = 100000
    sigma_array = numpy.array([[sigma_mass, 0], [0, sigma_int]])
    sigma_inv = numpy.linalg.inv(sigma_array)
    len_1 = spectrum_1.shape[0]
    len_2 = spectrum_2.shape[0]
    constant_term = 1.0 / (len_1 * len_2 * 4 * numpy.pi * numpy.sqrt(sigma_mass * sigma_int))
    sum_term = 0

    tol = 5 * numpy.sqrt(sigma_mass)

    for p_1_idx, p_2_idx in find_pairs(spectrum_1, spectrum_2, tol):
        p_1 = spectrum_1[p_1_idx, :]
        p_2 = spectrum_2[p_2_idx, :]
        d = p_1 - p_2
        sum_term += numpy.exp(-0.25 * numpy.sum(d * sigma_inv * d))
    # print(sum_term)
    # print(numpy.sum(sum_term))
    return constant_term * sum_term


@jit(nopython=True)
def find_pairs(spec1, spec2, tol, shift=0):
    matching_pairs = []
    spec2_lowpos = 0
    spec2_length = len(spec2)

    for idx in range(len(spec1)):
        mz, intensity = spec1[idx, :]
        while spec2_lowpos < spec2_length and spec2[spec2_lowpos][0] + shift < mz - tol:
            spec2_lowpos += 1
        if spec2_lowpos == spec2_length:
            break
        spec2_pos = spec2_lowpos
        while spec2_pos < spec2_length and spec2[spec2_pos][0] + shift < mz + tol:
            matching_pairs.append((idx, spec2_pos))
            spec2_pos += 1

    return matching_pairs


def ppk_nloss(spec1, spec2, prec1, prec2, sigma_mass, sigma_int):
    spec1_loss = ([prec1, 0] - spec1) * [1, -1]
    spec2_loss = ([prec2, 0] - spec2) * [1, -1]
    k_nloss = ppk_limit(spec1_loss[::-1], spec2_loss[::-1], sigma_mass, sigma_int)
    return k_nloss


def ppk_r(spec1, spec2, prec1, prec2, sigma_mass, sigma_int):
    k_peaks = ppk(spec1, spec2, sigma_mass, sigma_int)
    k_nloss = ppk_nloss(spec1, spec2, prec1, prec2, sigma_mass, sigma_int)
    k_diff = ppk_diff(spec1, spec2, sigma_mass, sigma_int)
    return k_peaks + k_nloss + k_diff


def ppk_diff(spec1, spec2, sigma_mass, sigma_int):
    spec1_diff = numpy.array([y - x for x, y in itertools.combinations(spec1, 2)])
    spec2_diff = numpy.array([y - x for x, y in itertools.combinations(spec2, 2)])
    k_diff = ppk(spec1_diff, spec2_diff, sigma_mass, sigma_int)
    return k_diff



def strip_leading(line):
    return ' '.join(line.split()[1:])


class MSSpectrum(object):
    def __init__(self):
        self.compound = None
        self.formula = None
        self.ionisation = None
        self.raw_parentmass = None
        self.filename = None
        self.id = None

        self.raw_spectrum = None
        self.output_spectrum = None

        # self.filter is a function that can be used
        # to filter the raw spectrum, e.g. denoising,
        # removal of adduct, etc.
        self.filter = None

        self.correct_for_ionisation = False

    def load(self, filename):
        self.output_spectrum = None
        self.filename = filename
        spectrum = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) is 0:
                    pass
                elif line.startswith('>compound'):
                    self.compound = strip_leading(line)
                elif line.startswith('>formula'):
                    self.formula = strip_leading(line)
                elif line.startswith('>ionization'):
                    self.ionisation = strip_leading(line)
                elif line.startswith('>parentmass'):
                    self.raw_parentmass = float(strip_leading(line))
                elif line.startswith('>'):
                    pass
                elif line.startswith('#inchi'):
                    self.inchi = strip_leading(line)
                elif line.startswith('#SpectrumID'):
                    self.id = strip_leading(line)
                elif line.startswith('#'):
                    pass
                else:
                    mass, charge = line.split()
                    mass = float(mass)
                    charge = float(charge)
                    spectrum.append((mass, charge))
        self.raw_spectrum = numpy.array(spectrum)

    @property
    def parentmass(self):
        if self.correct_for_ionisation:
            return self.raw_parentmass - self.ionisation_mass
        else:
            return self.raw_parentmass

    @property
    def spectrum(self):
        if self.filter is None:
            if self.correct_for_ionisation:
                return self.shifted_spectrum
            else:
                return self.raw_spectrum
        else:
            if self.output_spectrum is None:
                self.output_spectrum = self.filter(self)
            return self.output_spectrum


    @property
    def shifted_spectrum(self):
        return self.raw_spectrum - [self.ionisation_mass, 0]

    @property
    def ionisation_mass(self):
        return IONISATION_MASSES[self.ionisation]


PROTON_MASS = 1.00727645199076
IONISATION_MASSES = {
        "[M+H]+": PROTON_MASS,
        "[M+H-H2O]+": PROTON_MASS - 18.01056468638, 
        "[M+K]+": 38.963158,
        "[M+Na]+": 22.989218
        }


def create_ppk_matrix():
    iokr_data_path = '/home/grimur/iokr/data'
    data_gnps = scipy.io.loadmat("/home/grimur/iokr/data/data_GNPS.mat")
    ms_path = '/home/grimur/iokr/data/SPEC'

    sigma_mass = 0.00001
    sigma_int = 100000.0

    iokrdata = data.IOKRDataServer(iokr_data_path, kernel='PPKr.txt')
    ker_size = len(iokrdata.spectra)
    
    kernel_matrix_peaks = numpy.zeros((ker_size, ker_size))
    kernel_matrix_nloss = numpy.zeros((ker_size, ker_size))
    kernel_matrix_ppkr = numpy.zeros((ker_size, ker_size))

    j_ms = MSSpectrum()
    j_ms.filter = spectrum_filters.filter_by_collected_dag
    i_ms = MSSpectrum()
    i_ms.filter = spectrum_filters.filter_by_collected_dag

    import time
    t0 = time.time()
    cnt = 0
    for i in range(len(iokrdata.spectra)):
        i_name = iokrdata.spectra[i][0]
        i_ms.load(ms_path + os.sep + i_name + '.ms')
        # for j in range(i + 1):
        if True:
            j = i
            cnt += 1

            j_name = iokrdata.spectra[j][0]
            j_ms.load(ms_path + os.sep + j_name + '.ms')

            # print('%s vs %s' % (len(i_ms.spectrum), len(j_ms.spectrum)))

            if len(i_ms.spectrum) == 0 or len(j_ms.spectrum) == 0:
                print('empty')
                ij_peaks = 0
                ij_nloss = 0
            else:
                ij_peaks = ppk(i_ms.spectrum, j_ms.spectrum, sigma_mass, sigma_int)
                ij_nloss = ppk_nloss(i_ms.spectrum, j_ms.spectrum, i_ms.parentmass, j_ms.parentmass, sigma_mass, sigma_int)

            kernel_matrix_peaks[i, j] = ij_peaks
            kernel_matrix_peaks[j, i] = ij_peaks

            kernel_matrix_nloss[i, j] = ij_nloss
            kernel_matrix_nloss[j, i] = ij_nloss

            kernel_matrix_ppkr[i, j] = ij_peaks + ij_nloss 
            kernel_matrix_ppkr[j, i] = ij_peaks + ij_nloss

            if cnt % 100 == 0:
                print('done %s/%s, %s' % (cnt, (ker_size ** 2) / 2, time.time() - t0))
                t0 = time.time()
            
    # numpy.savetxt('ppk_peaks.csv', kernel_matrix_peaks, delimiter=',')
    # numpy.savetxt('ppk_nloss.csv', kernel_matrix_nloss, delimiter=',')
    # numpy.savetxt('ppk_r.csv', kernel_matrix_ppkr, delimiter=',')
    numpy.save('ppk_dag_peaks.npy', kernel_matrix_peaks)
    numpy.save('ppk_dag_nloss.npy', kernel_matrix_nloss)




def create_ppk_matrix_stripe_serial(filter_func, shift, output_name):
    iokr_data_path = '/home/grimur/iokr/data'
    data_gnps = scipy.io.loadmat("/home/grimur/iokr/data/data_GNPS.mat")
    ms_path = '/home/grimur/iokr/data/SPEC'

    iokrdata = data.IOKRDataServer(iokr_data_path)
    ker_size = len(iokrdata.spectra)

    kernel_matrix_peaks = numpy.zeros((ker_size, ker_size))
    kernel_matrix_nloss = numpy.zeros_like(kernel_matrix_peaks)

    p = multiprocessing.Pool(8)
    active_jobs = []

    t0 = time.time()
    names = [x[0] for x in iokrdata.spectra]
    cnt = 0
    for i in range(len(iokrdata.spectra)):
        if i == len(iokrdata.spectra) - 1:
            wait_for_clear = 0
        else:
            wait_for_clear = 10

        # active_jobs.append((i, p.apply_async(do_stripe, (i, names))))
        res = do_stripe(i, names, filter_func, shift)
        i_res = i

        if True:
        # if len(active_jobs) > wait_for_clear:
        #     active_jobs, results = gather_results_2(active_jobs, queue_length=wait_for_clear)
        #     for i_res, res in results:
                for j_res, values in enumerate(res):
                    ij_peaks, ij_nloss = values

                    kernel_matrix_peaks[i_res, j_res] = ij_peaks
                    kernel_matrix_peaks[j_res, i_res] = ij_peaks

                    kernel_matrix_nloss[i_res, j_res] = ij_nloss
                    kernel_matrix_nloss[j_res, i_res] = ij_nloss

                    cnt += 1
                    if cnt % 100 == 0:
                        print('done %s/%s, %s' % (cnt, (ker_size ** 2) / 2, time.time() - t0))
                        t0 = time.time()


    numpy.save(output_name + '_peaks.npy', kernel_matrix_peaks)
    numpy.save(output_name + '_nloss.npy', kernel_matrix_nloss)


def gather_results(active_jobs):
    while len(active_jobs) > 500:
        done_jobs = []
        remaining_jobs = []
        for i, j, job in active_jobs:
            if job.ready():
                res = job.get()
                done_jobs.append((i, j, res))
            else:
                remaining_jobs.append((i, j, job))
        active_jobs = remaining_jobs
        # time.sleep(1)
    return active_jobs, done_jobs


def gather_results_2(active_jobs, queue_length):
    while len(active_jobs) > queue_length:
        done_jobs = []
        remaining_jobs = []
        for i, job in active_jobs:
            if job.ready():
                res = job.get()
                done_jobs.append((i, res))
            else:
                remaining_jobs.append((i, job))
        active_jobs = remaining_jobs
        time.sleep(0.5)
    return active_jobs, done_jobs


def do_stripe(i, names, filter_func, shift):
    iokr_data_path = '/home/grimur/iokr/data'
    data_gnps = scipy.io.loadmat("/home/grimur/iokr/data/data_GNPS.mat")
    ms_path = '/home/grimur/iokr/data/SPEC'

    sigma_mass = 0.00001
    sigma_int = 100000.0

    i_ms = MSSpectrum()
    i_ms.correct_for_ionisation = shift
    i_ms.filter = filter_func
    j_ms = MSSpectrum()
    j_ms.correct_for_ionisation = shift
    j_ms.filter = filter_func
    
    i_ms.load(ms_path + os.sep + names[i] + '.ms')
    results = []
    for j in range(i + 1):
        j_ms.load(ms_path + os.sep + names[j] + '.ms')

        ij_peaks = ppk(i_ms.spectrum, j_ms.spectrum, sigma_mass, sigma_int)
        ij_nloss = ppk_nloss(i_ms.spectrum, j_ms.spectrum, i_ms.parentmass, j_ms.parentmass, sigma_mass, sigma_int)

        results.append((ij_peaks, ij_nloss))

    return results




def do_pair(i_spectrum, j_spectrum, i_parentmass, j_parentmass, sigma_mass, sigma_int):
    if len(i_spectrum) == 0 or len(j_spectrum) == 0:
        print('empty spectrum!')
        return 0, 0
    ij_peaks = ppk(i_spectrum, j_spectrum, sigma_mass, sigma_int)
    ij_nloss = ppk_nloss(i_spectrum, j_spectrum, i_parentmass, j_parentmass, sigma_mass, sigma_int)
    # ij_diff = ppk_diff(i_spectrum, j_spectrum, sigma_mass, sigma_int)
    return ij_peaks, ij_nloss  # , ij_diff


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='filter_type', default=None)
    parser.add_argument('-c', dest='collected', action='store_true', default=False)
    parser.add_argument('-s', dest='shift', action='store_true', default=False)
    parser.add_argument('-o', dest='output', required=True)
    args = parser.parse_args()

    if args.filter_type == 'dag':
        if args.collected:
            filter_func = spectrum_filters.filter_by_collected_dag
        else:
            filter_func = spectrum_filters.filter_by_dag
    elif args.filter_type == 'tree':
        if args.collected:
            filter_func = spectrum_filters.filter_by_collected_tree
        else:
            if args.shift == False:
                filter_func = spectrum_filters.filter_by_tree_unshifted
            else:
                filter_func = spectrum_filters.filter_by_tree
    elif args.filter_type is None:
        filter_func = None
    else:
        raise SystemExit('Unknown filter: %s' % args.filter_type)

    # create_ppk_matrix_parallell()
    create_ppk_matrix_stripe_serial(filter_func, args.shift, args.output)
    # create_ppk_matrix()
