# inversedipole
A numerical implementation to solve the dipole scattering amplitude from inclusive deep inelastic scattering cross sections as an inverse problem.
This approach leverages the observation that the inference problem is an integral transform inversion problem.
Derivation and a proof-of-principole demonstration of this approach is published in [arXiv:2509.05005 [hep-ph]](https://arxiv.org/abs/2509.05005v1) in collaboration with A. Kykkänen and H. Schlüter., and the corresponding source code is tagged with the release [arXiv-2509.05005](https://github.com/hhannine/inversedipole/releases/tag/arXiv-2509.05005).

# Overview of key components

- `deepinelasticscattering.py` implements the leading order DIS cross sections in the dipole picture.
- `discretize_forward_operator_masses.py` implements the compution and export of the forward operators.
- `reconstruct_dipole_matlab_gaussian_errors.m` implements the reconstruction algorithm.
