import numpy
import os
import iokr_opt
import iokrdata as iokrdataserver
import mk_fprints as fingerprint
import spectrum


_IOKR = None


class IOKRWrapper(object):
    """
    Wrapper around the IOKR server.
    Takes care of format conversion, fingerprint calculations, etc.
    Should also eventually take over the hardcoded stuff curently in get_iokr_server.
    """
    def __init__(self):
        self.fingerprint_type = None
        self.fingerprint_kernel = None
        self.ms_kernel = None

        self.iokr_server = None

    def fingerprint(self, smiles):
        """
        Calculate molecular fingerprint for a SMILES string
        """
        return fingerprint.fingerprint_from_smiles(smiles, self.fingerprint_type)

    def rank_smiles(self, ms, candidate_smiles):
        """
        Rank a spectrum against a candidate set of SMILES strings
        """
        print('Calculate candidate FPs')
        candidate_fps = numpy.array([self.fingerprint(x) for x in candidate_smiles])
        print('Extract latent basis')
        latent, latent_basis, gamma = self.iokr_server.get_data_for_novel_candidate_ranking()

        print('Get kernel vector for input sample')
        ms_kernel_vector = numpy.array(self.iokr_server.get_kernel_vector_for_sample(ms))
        print('Rank candidate set')
        ranking, _ = iokr_opt.rank_candidates_opt(0, candidate_fps, latent, ms_kernel_vector, latent_basis, gamma)

        return ranking


def get_iokr_server():
    datapath = "/home/grimur/iokr/data"
    kernel_files = [datapath + os.sep + 'input_kernels_gh/ppk_all_shifted_normalised_nloss.npy',
                    datapath + os.sep + 'input_kernels_gh/ppk_all_shifted_normalised_peaks.npy']
    fingerprint_type = "klekota-roth"
    iokr_wrapper = IOKRWrapper()

    iokr_wrapper.fingerprint_type = fingerprint_type
    iokr_wrapper.fingerprint_kernel = None  # function

    print('Init IOKR data server')
    iokrdata = iokrdataserver.IOKRDataServer(datapath, kernel=None)

    # When the kernel is initialised from matrix, we don't have guarantee
    # that the kernel_file and iokrdata.calculate_kernel match!
    print('Init kernel values')
    # Want to be able to set this to novel kernels
    kernel_matrix = iokrdataserver.load_kernels(kernel_files)
    iokrdata.kernel = kernel_matrix

    print('Load MS files')
    ms_path = '/home/grimur/iokr/data/SPEC'
    iokrdata.load_ms_files(ms_path)

    def ppk_wrapper(ms_i, ms_j):
        sigma_mass = 0.00001
        sigma_int = 1000000.0
        ppk = spectrum.ppk(ms_i.spectrum, ms_j.spectrum, sigma_mass, sigma_int)
        return ppk

    # The function should accept two MSSpectrum objects and return a value
    print('Calculate kernel')
    iokrdata.calculate_kernel = ppk_wrapper
    # TODO: This is super slow.
    # iokrdata.build_kernel_matrix()

    iokrdata.set_fingerprint(fingerprint_type)

    all_indices = iokrdata.get_all_indices()

    iokr = iokr_opt.InputOutputKernelRegression(iokrdata)
    iokr.set_training_indices(all_indices, _lambda=0.001)
    iokr.fit()

    iokr_wrapper.iokr_server = iokr

    return iokr_wrapper


def test():
    from pyteomics import mgf
    d = mgf.read('/home/grimur/iokr/data/mibig/matched_mibig_gnps_2.0.mgf')
    # Wrap the MGF entry in a MSSpectrum object
    test_spectrum = spectrum.MSSpectrum(d.next())

    # Candidate set
    SMILES = ["C\C=C\C=C\C(=O)C1=C(O)C(C)=C(O)C(C)=C1",
              "CC1=CC2=C(C(O)=C1)C(=O)C3=C(C=C(O)C=C3O)C2=O",
              "CC1=CC2=C(C(O)=C1)C(=O)C3=C(C=C(O)C=C3O)C2=O",
              "CC1=C2C(OC(=O)C3=C2C=C(O)C=C3O)=CC(O)=C1",
              "CC1=C2C(=O)C3=C(OC2=CC(O)=C1)C=C(O)C=C3O",
              "CC1CC(C)C(=O)C(C1)C(O)CC2CC(=O)NC(=O)C2"
              ]

    iokr = get_iokr_server()
    print('done init')

    print('rank')
    rank = iokr.rank_smiles(test_spectrum, SMILES)
    print(rank)


if __name__ == '__main__':
    test()
