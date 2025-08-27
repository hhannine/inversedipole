# Predefined mass schemes to use in the discretization of the forward operator.
# Pole mass values from Ref. https://arxiv.org/pdf/hep-ph/9712201

# masses and fractional charge squared is recorded under the key for each quark as a tuple.

mass_scheme_standard = {
    "u": (0.14, 4/9),
    "d": (0.14, 1/9),
    "s": (0.14, 1/9),
    "c": (1.4, 4/9),
    "b": (4.9, 1/9),
}

mass_scheme_pole = {
    "u": (0.501, 4/9),
    "d": (0.517, 1/9),
    "s": (0.687, 1/9),
    "c": (1.59, 4/9),
    "b": (4.89, 1/9),
}

mass_scheme_mq_Mpole = {
    "u": (0.0307, 4/9),
    "d": (0.0445, 1/9),
    "s": (0.283, 1/9),
    "c": (1.213, 4/9),
    "b": (4.248, 1/9),
}

mass_scheme_mq_mq = {
    "u": (0.436, 4/9),
    "d": (0.448, 1/9),
    "s": (0.553, 1/9),
    "c": (1.302, 4/9),
    "b": (4.339, 1/9),
}

mass_scheme_mcharm = {
    "u": (0.00418, 4/9),
    "d": (0.00840, 1/9),
    "s": (0.1672, 1/9),
    "c": (1.302, 4/9),
    "b": (6.12, 1/9),
}

mass_scheme_mbottom = {
    "u": (0.00317, 4/9),
    "d": (0.00637, 1/9),
    "s": (0.1268, 1/9),
    "c": (0.949, 4/9),
    "b": (4.34, 1/9),
}

mass_scheme_mW = {
    "u": (0.00235, 4/9),
    "d": (0.00473, 1/9),
    "s": (0.0942, 1/9),
    "c": (0.684, 4/9),
    "b": (3.03, 1/9),
}

mass_scheme_heracc_charm_only = {
    "c": (1.302, 4/9),
}
