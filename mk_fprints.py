import scipy.io
from cdk_pywrapper.cdk_pywrapper import Compound
import pickle
import numpy

def main():
    datapath = '/home/grimur/iokr/data'

    data = scipy.io.loadmat(datapath + '/data_GNPS.mat')

    fp_size = 1024

    new_fps = []

    for i in data['inchi']:
        print('Processing sample %s' % i)
        inchi = i[0][0]
        print(inchi)
        c = Compound(compound_string=inchi, identifier_type='inchi')
        fp = c.get_fingerprint()
        fp_array = numpy.zeros(fp_size)
        for fp_bit in range(fp_size):
            fp_array[fp_bit] = fp.get(fp_bit)
        new_fps.append(fp_array)

    with open(datapath + '/cdk_fingerprints.bin', 'wb') as f:
        pickle.dump(new_fps, f)


def default_fingerprint_from_inchi(inchi):
    c = Compound(compound_string=inchi, identifier_type='inchi')
    # this should accommodate different types!
    fp = c.get_fingerprint()
    fp_size = 1024
    fp_array = numpy.zeros(fp_size)
    for fp_bit in range(fp_size):
        fp_array[fp_bit] = fp.get(fp_bit)
    return fp_array


def fingerprint_from_inchi(inchi, fingerprint_type=None):
    c = Compound(compound_string=inchi, identifier_type='inchi')

    if fingerprint_type is None:
        fingerprinter = c.cdk.fingerprint.Fingerprinter()
    elif fingerprint_type is 'default':
        fingerprinter = c.cdk.fingerprint.Fingerprinter()
    elif fingerprint_type is 'substructure':
        fingerprinter = c.cdk.fingerprint.SubstructureFingerprinter()
    elif fingerprint_type is 'klekota-roth':
        fingerprinter = c.cdk.fingerprint.KlekotaRothFingerprinter()

    fp = fingerprinter.getBitFingerprint(c.mol_container)
    fp_size = fp.size()
    fp_array = numpy.zeros(fp_size)
    for fp_bit in range(fp_size):
        fp_array[fp_bit] = fp.get(fp_bit)
    return fp_array


if __name__ == '__main__':
    main()
