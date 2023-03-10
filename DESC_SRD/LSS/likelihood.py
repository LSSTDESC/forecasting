import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
import firecrown.likelihood.gauss_family.statistic.source.number_counts as nc
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap

import sacc
import pyccl as ccl

def build_likelihood(_):
    """Create a firecrown likelihood for an SRD-like galaxy clustering analysis.
    No systematics yet apart from linear bias. 
    Based on DES 3x2pt example in firecrown"""
    
    sources: Dict[str, nc.NumberCounts] = {}
        
    for i in range(5):
        sources[f"lens{i}"] = nc.NumberCounts(
            sacc_tracer=f"lens{i}"
        )

    # Now that we have all sources we can instantiate all the two-point
    # functions. For each one we create a new two-point function object.
    stats = {}
    for i in range(5):
        stats[f"ell_cl_lens{i}_lens{i}"] = TwoPoint(
            source0=sources[f"lens{i}"],
            source1=sources[f"lens{i}"],
            sacc_data_type="galaxy_density_cl",
        )

    # Here we instantiate the actual likelihood. The statistics argument carry
    # the order of the data/theory vector.
    modeling_tools = ModelingTools()
    likelihood = ConstGaussian(statistics=list(stats.values()))

    # We load the correct SACC file.
    saccfile = "lsst_y1_desc_srd_sacc.fits"
    sacc_data = sacc.Sacc.load_fits(saccfile)
    
    ellmax_lin = [250, 394, 522, 635,  736] #corresponding to kmax=0.3h at z bin centers
    for i, t in enumerate(sacc_data.tracers):
        sacc_data.remove_selection(tracers=(t, t), ell__gt=ellmax_lin[i])

    likelihood.read(sacc_data)
    
    print(
        "Using parameters:", list(likelihood.required_parameters().get_params_names())
    )

    # This script will be loaded by the appropriated connector. The framework
    # will call the factory function that should return a Likelihood instance.
    return likelihood, modeling_tools

