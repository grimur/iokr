import multiprocessing
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


@jit(nopython=True)
def ppk(spectrum_1, spectrum_2, sigma_mass, sigma_int):
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


def ppk_nloss(spec1, spec2, prec1, prec2, sigma_mass, sigma_int):
    spec1_loss = ([prec1, 0] - spec1) * [1, -1]
    spec2_loss = ([prec2, 0] - spec2) * [1, -1]
    k_nloss = ppk(spec1_loss, spec2_loss, sigma_mass, sigma_int)
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
        self.parentmass = None
        self.filename = None

        self.raw_spectrum = None
        self.output_spectrum = None

        # self.filter is a function that can be used
        # to filter the raw spectrum, e.g. denoising,
        # removal of adduct, etc.
        self.filter = None

    def load(self, filename):
        self.filneame = filename
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
                    self.ionization = strip_leading(line)
                elif line.startswith('>parentmass'):
                    self.parentmass = float(strip_leading(line))
                elif line.startswith('>'):
                    pass
                elif line.startswith('#inchi'):
                    self.inchi = strip_leading(line)
                elif line.startswith('#'):
                    pass
                else:
                    mass, charge = line.split()
                    mass = float(mass)
                    charge = float(charge)
                    spectrum.append((mass, charge))
        self.raw_spectrum = numpy.array(spectrum)

    @property
    def spectrum(self):
        if self.filter is None:
            return self.raw_spectrum
        else:
            if self.output_spectrum is None:
                self.output_spectrum = self.filter(self.raw_spectrum)
            return self.output_spectrum



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
    kernel_matrix_diff = numpy.zeros((ker_size, ker_size))
    kernel_matrix_ppkr = numpy.zeros((ker_size, ker_size))

    j_ms = MSSpectrum()
    i_ms = MSSpectrum()

    cnt = 0
    for i in range(len(iokrdata.spectra)):
        i_name = iokrdata.spectra[i][0]
        i_ms.load(ms_path + os.sep + i_name + '.ms')
        for j in range(i + 1):
            cnt += 1

            j_name = iokrdata.spectra[j][0]
            j_ms.load(ms_path + os.sep + j_name + '.ms')

            print('%s vs %s' % (len(i_ms.spectrum), len(j_ms.spectrum)))
            
            ij_peaks = ppk(i_ms.spectrum, j_ms.spectrum, sigma_mass, sigma_int)
            kernel_matrix_peaks[i, j] = ij_peaks
            kernel_matrix_peaks[j, i] = ij_peaks

            ij_nloss = ppk_nloss(i_ms.spectrum, j_ms.spectrum, i_ms.parentmass, j_ms.parentmass, sigma_mass, sigma_int)
            kernel_matrix_nloss[i, j] = ij_nloss
            kernel_matrix_nloss[j, i] = ij_nloss

            ij_diff = ppk_diff(i_ms.spectrum, j_ms.spectrum, sigma_mass, sigma_int)
            kernel_matrix_diff[i, j] = ij_diff
            kernel_matrix_diff[j, i] = ij_diff

            kernel_matrix_ppkr[i, j] = ij_peaks + ij_nloss + ij_diff
            kernel_matrix_ppkr[j, i] = ij_peaks + ij_nloss + ij_diff

            print('done %s/%s' % (cnt, (ker_size ** 2) / 2))
            
    numpy.savetxt('ppk_peaks.csv', kernel_matrix_peaks, delimiter=',')
    numpy.savetxt('ppk_nloss.csv', kernel_matrix_nloss, delimiter=',')
    numpy.savetxt('ppk_diff.csv', kernel_matrix_diff, delimiter=',')
    numpy.savetxt('ppk_r.csv', kernel_matrix_ppkr, delimiter=',')


def create_ppk_matrix_parallell():
    iokr_data_path = '/home/grimur/iokr/data'
    data_gnps = scipy.io.loadmat("/home/grimur/iokr/data/data_GNPS.mat")
    ms_path = '/home/grimur/iokr/data/SPEC'

    sigma_mass = 0.00001
    sigma_int = 100000.0

    iokrdata = data.IOKRDataServer(iokr_data_path, kernel='PPKr.txt')
    ker_size = len(iokrdata.spectra)
    
    kernel_matrix_peaks = numpy.zeros((ker_size, ker_size))
    kernel_matrix_nloss = numpy.zeros((ker_size, ker_size))
    kernel_matrix_diff = numpy.zeros((ker_size, ker_size))
    kernel_matrix_ppkr = numpy.zeros((ker_size, ker_size))

    j_ms = MSSpectrum()
    i_ms = MSSpectrum()

    p = multiprocessing.Pool(8)
    active_jobs = []

    cnt = 0
    for i in range(len(iokrdata.spectra)):
        i_name = iokrdata.spectra[i][0]
        i_ms.load(ms_path + os.sep + i_name + '.ms')
        for j in range(i + 1):
            cnt += 1

            j_name = iokrdata.spectra[j][0]
            j_ms.load(ms_path + os.sep + j_name + '.ms')

            print('%s vs %s' % (len(i_ms.spectrum), len(j_ms.spectrum)))
            
            args = (i_ms.spectrum, j_ms.spectrum, i_ms.parentmass, j_ms.parentmass, sigma_mass, sigma_int)
            active_jobs.append((i, j, p.apply_async(do_pair, args)))

            if len(active_jobs) > 10:
                active_jobs, results = gather_results(active_jobs)

                for i_res, j_res, res in results:
                    ij_peaks, ij_nloss, ij_diff = res

                    kernel_matrix_peaks[i_res, j_res] = ij_peaks
                    kernel_matrix_peaks[j_res, i_res] = ij_peaks

                    kernel_matrix_nloss[i_res, j_res] = ij_nloss
                    kernel_matrix_nloss[j_res, i_res] = ij_nloss

                    kernel_matrix_diff[i_res, j_res] = ij_diff
                    kernel_matrix_diff[j_res, i_res] = ij_diff

            print('done %s/%s' % (cnt, (ker_size ** 2) / 2))
            
    numpy.savetxt('ppk_peaks.csv', kernel_matrix_peaks, delimiter=',')
    numpy.savetxt('ppk_nloss.csv', kernel_matrix_nloss, delimiter=',')
    numpy.savetxt('ppk_diff.csv', kernel_matrix_diff, delimiter=',')

    kernel_matrix_ppkr = kernel_matrix_peaks + kernel_matrix_nloss + kernel_matrix_diff

    numpy.savetxt('ppk_r.csv', kernel_matrix_ppkr, delimiter=',')


def gather_results(active_jobs):
    while len(active_jobs) > 10:
        done_jobs = []
        remaining_jobs = []
        for i, j, job in active_jobs:
            if job.ready():
                res = job.get()
                done_jobs.append((i, j, res))
            else:
                remaining_jobs.append((i, j, job))
        active_jobs = remaining_jobs
        time.sleep(1)
    return active_jobs, done_jobs


def do_pair(i_spectrum, j_spectrum, i_parentmass, j_parentmass, sigma_mass, sigma_int):
    ij_peaks = ppk(i_spectrum, j_spectrum, sigma_mass, sigma_int)
    ij_nloss = ppk_nloss(i_spectrum, j_spectrum, i_parentmass, j_parentmass, sigma_mass, sigma_int)
    ij_diff = ppk_diff(i_spectrum, j_spectrum, sigma_mass, sigma_int)
    return ij_peaks, ij_nloss, ij_diff


if __name__ == '__main__':
    create_ppk_matrix_parallell()
