Implementation of
https://doi.org/10.1093/bioinformatics/btw246

Test data downloaded from
https://doi.org/10.5281/zenodo.804241

Requirements:
- cdk_pywrapper (substructure fingerprinting)
- pyteomics (MGF file parser)
- numba
- numpy
- scipy

The files, especially build_input_matrix.py, build_test_matrix.py, run_iokr.py, run_iokr_novel.py and nplinker_iokr.py, include a bunch of hardcoded data paths and also make strong assumptions about the structure of the data.

Note that the DAG- and fragmentation tree filters on the MS spectra assume the presence of the corresponding data, which is not included in the data download mentioned above.
