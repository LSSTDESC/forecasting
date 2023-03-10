import pyccl as ccl
import numpy as np
from scipy.ndimage import gaussian_filter
import sacc

s = sacc.Sacc()

year = 'Y1'
if year=='Y1' or year==1 or year=='y1':
    nbin_z = 5
    z0 = 0.26
    alpha = 0.94
    sig_z = 0.03
    Ngal = 18 # Normalisation, galaxies/arcmin^2
    linear_bias = np.array([1.562362, 1.732963, 1.913252, 2.100644, 2.293210])
else:
    print('only Y1 implemented currently')

h=0.6727
cosmo = ccl.Cosmology(
    Omega_c=0.2664315,
    Omega_b=0.0491685,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
    sigma8 = 0.831,
    n_s=0.9645,
    h=h,
)

bin_edges = np.linspace(0.2, 1.2, nbin_z+1)
bin_low_zs = bin_edges[:-1]
bin_high_zs = bin_edges[1:]
zmids = (bin_low_zs+bin_high_zs)/2

nbin_ell = 20 
lmin = 20
lmax = 15000
ell_edges = np.geomspace(lmin, lmax, nbin_ell+1)
ell_centers = [lmin**(1-(i+0.5)/nbin_ell)*lmax**((i+0.5)/nbin_ell) for i in range(nbin_ell)]

chis = ccl.comoving_radial_distance(cosmo, 1/(1+zmids)) # Mpc
kmax = 0.3*h
lmax = kmax*chis-0.5

z= np.linspace(0., 3., 1001)
#redshift distribution normalized to galaxy number density -- I don't think the normalization matters
pz = (z / z0)**2. * np.exp(-(z / z0)**alpha) # Redshift distribution, p(z)
norm = Ngal/np.trapz(pz, z)
nz = norm * pz # Number density distribution

tracers = []
for i, zmin in enumerate(bin_low_zs):
    zmax = bin_high_zs[i]
    zmid = zmids[i]
    nz_bin = np.zeros_like(z)
    nz_bin[(z>zmin) & (z<zmax)] = nz[(z>zmin) & (z<zmax)]
    dz = z[1]-z[0]
    nz_bin = gaussian_filter(nz_bin, sig_z*(1+zmid)/dz)
    
    # galaxy clustering CCL tracer for computing mock datavector
    galaxy_tracer_bin = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z,nz_bin), bias=(zmids,linear_bias), mag_bias=None)
    tracers.append(galaxy_tracer_bin)
    
    # galaxy clustering SACC tracer for saving to file later
    s.add_tracer('NZ', "lens%d" % (i),  # Name
             quantity='galaxy_density',  # Quantity
             spin=0,  # Spin
             z=z,  # z
             nz=nz_bin)  # nz
    
ell_unbinned = np.arange(20, 15002)
n_ell_unbinned = len(ell_unbinned)
# window functions in ell: assume tophat
ell_windows_tophat = np.zeros([n_ell_unbinned, nbin_ell])
for i in range(nbin_ell):
    ell_windows_tophat[:,i] = (ell_unbinned>=ell_edges[i]) & (ell_unbinned<=ell_edges[i+1])
    ell_windows_tophat[:,i]/=np.sum(ell_windows_tophat[:,i])
    
# Create a SACC bandpower window object
wins = sacc.BandpowerWindow(ell_unbinned, ell_windows_tophat)

for i in range(nbin_z):
    cl_zbin_i = ccl.angular_cl(cosmo, tracers[i], tracers[i], ell_unbinned)
    cl_binned = [np.mean(cl_zbin_i[(ell_unbinned>=ell_edges[j]) & (ell_unbinned<=ell_edges[j+1])]) for j in range(nbin_ell)]
    cl_binned_win = [np.sum(cl_zbin_i*ell_windows_tophat[:,j])for j in range(nbin_ell)]
    # cosmolike convention is that for scales that we are going to cut, the cls are set to 0
    # what is best practice for implementing scale cuts in sacc format?
    
    # GC-GC
    s.add_ell_cl('galaxy_density_cl',  # Data type
     "lens%d" % (i),  # 1st tracer's name
     "lens%d" % (i),  # 2nd tracer's name
     ell_centers,  # Effective multipole
     cl_binned,  # Power spectrum values
     window=wins,  # Bandpower windows
    )

# read in covariance matrix from DESC SRD CosmoLike, select galaxy clustering bits
# covmat
cov = np.loadtxt('/Users/heatherp/Documents/DESC/cosmolike_inputs_for_3x2_used_for_srd/outputs/cov/Y1_pos_pos_cov_unmasked.txt')
#mask = np.abs(cov)>1e-15

s.add_covariance(cov) # deal with mask?
s.save_fits("lsst_y1_desc_srd_sacc.fits", overwrite=True)
print("saved in sacc format as lsst_y1_desc_srd_sacc.fits")