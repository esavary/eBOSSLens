import numpy as np
import pyfits as pf
import math
from scipy.optimize import leastsq
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy import special as sp
from scipy import interpolate
from matplotlib import rcParams
import os
import errno

### Constants:
H0 = 72e3 #m s-1 Mpc-1
c = 299792458 #m s-1
OmegaM = 0.258
OmegaL = 0.742


#-------------------------------- Function definitions ----------------------------------------------------
# Lorentzian 
def lorentz(x,x_0,g,A):
	return A*(g/np.pi)/((x-x_0)**2 + g**2)
# chi2 Lorentzian
def chi2Lorenz(params,xdata,ydata,ivar):
	return np.sum(ivar*(ydata - lorentz(x=xdata, x_0=params[0], g=params[1], A=params[2]))**2)/(len(xdata)-len(params)-1)

# Generate a Gaussian around x_0 with amplitude A and variance var
def gauss(x,x_0,A,var):
	y = A*np.exp( (-(x-x_0)**2) / (2*var) )
	return y
# Generate doublet
def gauss2(x,x1,x2,A1,A2,var):
	return gauss(x,x1,A1,var) + gauss(x,x2,A2,var)
#Skew normal profile
def skew(x,A,w,a,eps):
	phi = 0.5*(1+sp.erf(a*(x-eps)/(w*np.sqrt(2))))
	return A*2*gauss(x,eps,1/np.sqrt(2*np.pi),w**2)*phi/w
# Skew normal doublet profile
def skew2(x,A1,w1,a1,eps1,A2,w2,a2,eps2):
	return skew(x,A1,w1,a1,eps1) + skew(x,A2,a2,w2,eps2)

#Reduced Chi square for one gaussian
def chi2g(params, xdata, ydata, ivar):
	return np.sum(ivar*(ydata - gauss(x=xdata, x_0=params[0], A=params[1], var=params[2]))**2)/(len(xdata)-len(params)-1)
#Reduced Chi square for Doublet
def chi2D(params, xdata, ydata, ivar):
	return np.sum(ivar*(ydata - gauss(x=xdata, x_0=params[3], A=params[0], var=params[1])-gauss(x=xdata, x_0=params[4], A=params[2], var=params[1]))**2)/(len(xdata)-len(params) -1)
#Reduced Chi square for skew profile
def chi2skew(params, xdata, ydata, ivar):
	return np.sum(ivar*(ydata - skew(x=xdata,A = params[0], w=params[1], a=params[2], eps = params[3]))**2)/(len(xdata)-len(params)-1)
#Reduced Chi square for  double skew profile
def chi2skew2(params, xdata, ydata, ivar):
	return np.sum(ivar*(ydata - skew(x=xdata,A = params[0], w=params[1], a=params[2], eps = params[3]) - skew(x=xdata, A = params[4], w = params[5], a=params[6], eps=params[7]))**2)/(len(xdata)-len(params)-1)

# Check if x0 is near any emission line redshifted by z
def nearline(x0, zline, fiberid, z, mjd, plate):
	match1 = np.logical_and(abs(zline['linewave']*(1+z) -x0) < 10, zline['lineew']/zline['lineew_err'] > 6)
	match2 = np.logical_and(zline['fiberid']==fiberid,zline['mjd']==int(mjd))
	match3 = np.logical_and(zline['plate']==int(plate), zline['lineew_err']>0)
	match4 = np.logical_and(match1,np.logical_and(match2,match3))
	if (np.sum(match4)>0):
		return True
	else:
		return False

#Gaussian kernel used in first feature search (Bolton et al.,2004 method)	
def kernel(j,width,NormGauss,length):
	ker = np.zeros(length)
	ker[j-int(width*0.5):j+int(width*0.5)] = NormGauss
	return ker
	
# Estimated Einstein Radius from Single Isothermal Sphere (SIS) model
def radEinstein(z1,z2,vdisp):
	#compute ang. diam. distances
	Ds = ((c/H0)*quad(x12,0.0,z2)[0])/(1+z2)
	Dls = ((c/H0)*quad(x12,z1,z2)[0])/(1+z2)
	### return value in arcsec
	coeff = 3600*(180/np.pi)
	return coeff*4.0*np.pi*vdisp*vdisp*Dls/(c*c*Ds)

#Function needed for numerical computation of angular diameter distance
def x12(z):
	return 1.0/np.sqrt((1-OmegaM-OmegaL)*(1+z)*(1+z) + OmegaM*(1+z)**3 + OmegaL)
	
#Convert wavelength to bin number
def wave2bin(x,c0,c1,Nmax):
	b = int((np.log10(x)/c1)-c0/c1)
	if b <= 0:
		return 0
	elif b >= Nmax:
		return Nmax
	else:
		return b
#Give BOSS approximated resolution as a function of wavelength
def resolution(x):
	if 4000<x<5800:
		a = (2000-1400)/(5800-4000)
		b = 1400-a*4000
		return a*x+b
	elif 5800<x<6200:
		a = (1900-2000)/(6200-5800)
		b = 2000-a*5800
		return a*x+b
	elif 6200<x<9400:
		a = (2600-1900)/(9400-6200)
		b = 2600-a*9400
		return a*x+b
	else:
		return 2500

#Prepare the flux in the BOSS bins starting from MC template/any datapoints array
def template_stretch(template_x, template_y, xdata, x0,A,B,eps):
	if A < 0:
		A = -A
		template_y = template_y[::-1]
	k = max(1,int(len(template_x)/B))
	step = (template_x[-1]- template_x[0])/(len(template_x)-1)
	temp_x = np.linspace(template_x[0]-k*step, template_x[-1]+k*step,len(template_x)+2*k)
	temp_y = temp_x*0 + 0.5*(template_y[0]+template_y[-1])
	temp_y[k:-k] = template_y
	template_x, template_y = temp_x, temp_y
		
	m = np.mean(template_x) 
	template_x = B*(template_x -m) + m + eps 
	sigma = x0/resolution(x0)
	gaussian_kernel = gauss(template_x,x_0=x0+eps,A=1/np.sqrt(sigma*2*np.pi),var=sigma**2)
	template_y = np.convolve(template_y*A, gaussian_kernel, mode = 'same')
	interpol = interpolate.interp1d(template_x,template_y, kind ='linear')
	
	return interpol(xdata)
# Compute the chi2 any template template
def chi2template(params,xdata,ydata, template_x, template_y, x0, ivar):
	y_fit = template_stretch(template_x, template_y, xdata, x0, params[0],params[1],params[2])
	return np.sum(ivar*(ydata - y_fit)**2)/(len(xdata)-len(params)-1)

#Transform RA DEC to SDSS name
def SDSSname(RA,DEC):
	sign = np.sign(DEC)
	DEC = np.abs(DEC)
	HH = math.trunc(RA//15)
	MM = math.trunc((RA-HH*15.)*60./15.)
	SS = round((RA-HH*15.-MM*15./60.)*3600./15,4)
	SS = math.trunc(SS*100.)/100.
	DD = math.trunc(DEC)
	MM_dec = math.trunc((DEC-DD)*60.)
	SS_dec = (DEC - DD - MM_dec/60.)*3600
	SS_dec = math.trunc(SS_dec*10.)/10.
	if sign < 0:
		return'SDSS J'+'{:02}'.format(HH)+'{:02}'.format(MM)+'{:05.2f}'.format(SS)+'-'+'{:02}'.format(DD)+'{:02}'.format(MM_dec)+'{:04.1f}'.format(SS_dec)
	else:
		return 'SDSS J'+'{:02}'.format(HH)+'{:02}'.format(MM)+'{:05.2f}'.format(SS)+'+'+'{:02}'.format(DD)+'{:02}'.format(MM_dec)+'{:04.1f}'.format(SS_dec)

# Check if a path exists, if not make it
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
        
        
        
        
 
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
#-----------------------------------------------------------------------------------------------------

def load_data(mjd, plate, BOSS = True, eBOSS = False, logdir = '../../../../../SCRATCH/' ):
	if BOSS == True:
		spfile = '../../../../../../SCRATCH/BOSS/data/v5_7_0/'+ str(plate) + '/spPlate-' + str(plate) + '-' + str(mjd) + '.fits'
		zbfile = '../../../../../../SCRATCH/BOSS/data/v5_7_0/' + str(plate) + '/v5_7_0/' + 'spZbest-'+ str(plate) + '-' + str(mjd) + '.fits'
		zlfile = '../../../../../../SCRATCH/BOSS/data/v5_7_0/' + str(plate) + '/v5_7_0/' + 'spZline-'+ str(plate) + '-' + str(mjd) + '.fits'
	elif eBOSS == False:
		spfile = '../../../../../../SCRATCH/eBOSS/data/v5_10_0/'+ str(plate) + '/spPlate-' + str(plate) + '-' + str(mjd) + '.fits'
		zbfile = '../../../../../../SCRATCH/eBOSS/data/v5_10_0/' + str(plate) + '/v5_10_0/' + 'spZbest-'+ str(plate) + '-' + str(mjd) + '.fits'
		zlfile = '../../../../../../SCRATCH/eBOSS/data/v5_10_0/' + str(plate) + '/v5_10_0/' + 'spZline-'+ str(plate) + '-' + str(mjd) + '.fits'
	
	hdulist = pf.open(spfile)
	c0 = hdulist[0].header['coeff0']
	c1 = hdulist[0].header['coeff1']
	npix = hdulist[0].header['naxis1']
	wave = 10.**(c0 + c1 * np.arange(npix))
	#bunit = hdulist[0].header['bunit']
	flux = hdulist[0].data	
	ivar = hdulist[1].data
	#ivar_copy = copy.deepcopy(ivar)
	hdulist.close()
	hdulist = 0
	hdulist = pf.open(zbfile)
	vdisp = hdulist[1].data.field('VDISP')
	vdisp_err = hdulist[1].data.field('VDISP_ERR')
	synflux = hdulist[2].data
	fiberid = hdulist[1].data.field('FIBERID')
	RA = hdulist[1].data.field('PLUG_RA')
	DEC = hdulist[1].data.field('PLUG_DEC')
	obj_id = hdulist[1].data.field('OBJID')
	obj_class = hdulist[1].data.field('CLASS')
	obj_type = hdulist[1].data.field('OBJTYPE')
	z = hdulist[1].data.field('Z')
	zwarning = hdulist[1].data.field('ZWARNING')
	z_err = hdulist[1].data.field('Z_ERR')
	spectroflux = hdulist[1].data.field('SPECTROFLUX')
	rchi2 = hdulist[1].data.field('RCHI2')
	rchi2diff = hdulist[1].data.field('RCHI2DIFF')
	hdulist.close()
	hdulist = 0
	hdulist = pf.open(zlfile)
	zline = hdulist[1].data
	hdulist.close()
	hdulist = 0
	return c0,c1,wave,flux,ivar,vdisp, synflux,fiberid, RA, DEC, obj_id, obj_class, obj_type, z, zwarning, z_err, spectroflux, rchi2, rchi2diff, zline,npix
	
def load_data2(mjd, plate, BOSS = True, eBOSS = False, logdir = '../../../../../SCRATCH/' ):
	if BOSS == True:
		spfile = '../../../../../../SCRATCH/BOSS/data/v5_7_0/'+ str(plate) + '/spPlate-' + str(plate) + '-' + str(mjd) + '.fits'
		zbfile = '../../../../../../SCRATCH/BOSS/data/v5_7_0/' + str(plate) + '/v5_7_0/' + 'spZbest-'+ str(plate) + '-' + str(mjd) + '.fits'
		zlfile = '../../../../../../SCRATCH/BOSS/data/v5_7_0/' + str(plate) + '/v5_7_0/' + 'spZline-'+ str(plate) + '-' + str(mjd) + '.fits'
	elif eBOSS == False:
		spfile = '../../../../../../SCRATCH/eBOSS/data/v5_10_0/'+ str(plate) + '/spPlate-' + str(plate) + '-' + str(mjd) + '.fits'
		zbfile = '../../../../../../SCRATCH/eBOSS/data/v5_10_0/' + str(plate) + '/v5_10_0/' + 'spZbest-'+ str(plate) + '-' + str(mjd) + '.fits'
		zlfile = '../../../../../../SCRATCH/eBOSS/data/v5_10_0/' + str(plate) + '/v5_10_0/' + 'spZline-'+ str(plate) + '-' + str(mjd) + '.fits'
	
	hdulist = pf.open(spfile)
	npix = hdulist[0].header['naxis1']
	
	#bunit = hdulist[0].header['bunit']
	flux = hdulist[0].data	
	ivar = hdulist[1].data
	#ivar_copy = copy.deepcopy(ivar)
	hdulist.close()
	hdulist = 0
	hdulist = pf.open(zbfile)
	
	c0 = hdulist[0].header['coeff0']
	c1 = hdulist[0].header['coeff1']
	
	wave = 10.**(c0 + c1 * np.arange(npix))
	
	vdisp = hdulist[1].data.field('VDISP')
	vdisp_err = hdulist[1].data.field('VDISP_ERR')
	synflux = hdulist[2].data
	fiberid = hdulist[1].data.field('FIBERID')
	RA = hdulist[1].data.field('PLUG_RA')
	DEC = hdulist[1].data.field('PLUG_DEC')
	obj_id = hdulist[1].data.field('OBJID')
	obj_class = hdulist[1].data.field('CLASS')
	obj_type = hdulist[1].data.field('OBJTYPE')
	z = hdulist[1].data.field('Z')
	zwarning = hdulist[1].data.field('ZWARNING')
	z_err = hdulist[1].data.field('Z_ERR')
	spectroflux = hdulist[1].data.field('SPECTROFLUX')
	rchi2 = hdulist[1].data.field('RCHI2')
	rchi2diff = hdulist[1].data.field('RCHI2DIFF')
	hdulist.close()
	hdulist = 0
	hdulist = pf.open(zlfile)
	zline = hdulist[1].data
	hdulist.close()
	hdulist = 0
	return c0,c1,wave,flux,ivar,vdisp, synflux,fiberid, RA, DEC, obj_id, obj_class, obj_type, z, zwarning, z_err, spectroflux, rchi2, rchi2diff, zline,npix

