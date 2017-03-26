import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
from utils import *

def DR12Q_extractor(path = './Superset_DR12Q.fits'):
	hdulist = pf.open(path)
	z_vi_DR12Q = hdulist[1].data.field('Z_VI')
	z_PCA_DR12Q = hdulist[1].data.field('Z_PIPE')
	plate_DR12Q =  hdulist[1].data.field('PLATE')
	mjd_DR12Q = hdulist[1].data.field('MJD')
	fiber_DR12Q = hdulist[1].data.field('FIBERID')
	return np.transpose(np.vstack((plate_DR12Q,mjd_DR12Q,fiber_DR12Q,z_PCA_DR12Q,z_vi_DR12Q)))

def mask_QSO(ivar,z):
	# Masking Lya, NV, SiIV, CIV, etc...
	l_NV = 1240
	l_SiIV= 1400.0
	l_CIV = 1549.0
	l_HeII = 1640.0
	l_CIII = 1909.0
	l_CII = 2326.0
	l_FeII_a = 2382.765
	l_FeII_b = 2600.173
	l_MgII = 2798.0
	l_NeV = 3426.0
	l_OII = 3727
	l_NeIII = 3869
	l_Hd = 4101
	l_Hg = 4340
	l_Hb = 4861
	l_OIII_a = 4959
	l_OIII_b = 5007
	l_Ha = 6562.81
	
	#Mask the above emission lines of QSO's
	ivar[wave2bin((1+z)*(l_LyA -2.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_LyA +2.5*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_NV -0.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_NV +0.5*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_SiIV -1.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_SiIV +1.5*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_CIV -2*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_CIV +2*l_width),c0,c1,Nmax)] = 0 
	ivar[wave2bin((1+z)*(l_HeII -0.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_HeII +1.5*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_CIII -l_width),c0,c1,Nmax):wave2bin((1+z)*(l_CIII +l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_CII -0.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_CII +0.5*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_FeII_a -l_width),c0,c1,Nmax):wave2bin((1+z)*(l_FeII_a +l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_FeII_b -l_width),c0,c1,Nmax):wave2bin((1+z)*(l_FeII_b +l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_MgII -2.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_MgII +2.5*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_NeV -0.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_NeV +0.5*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_OII -0.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_OII +1.5*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_NeIII -0.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_NeIII +0.5*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_Hd -0.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_Hd +0.5*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_Hg - 1.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_Hg + 1.5*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_Hb - l_width),c0,c1,Nmax):wave2bin((1+z)*(l_Hb + l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_OIII_a -2*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_OIII_a +2*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_OIII_b -2*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_OIII_b +2*l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(l_Ha -2*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_Ha +3*l_width),c0,c1,Nmax)] = 0
	#additional Lines added after 1st results 
	# Ne IV
	ivar[wave2bin((1+z)*(2427 -l_width),c0,c1,Nmax):wave2bin((1+z)*(2427 +l_width),c0,c1,Nmax)] = 0
	# Ne V
	ivar[wave2bin((1+z)*(3350 - 0.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(3350 + 0.5*l_width),c0,c1,Nmax)] = 0
	# NI
	ivar[wave2bin((1+z)*(5200 -l_width),c0,c1,Nmax):wave2bin((1+z)*(5200 +l_width),c0,c1,Nmax)] = 0
	# [Fe VII]	
	ivar[wave2bin((1+z)*(5721 - 0.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(5721 + 0.5*l_width),c0,c1,Nmax)] = 0
	#[Fe VII]	
	ivar[wave2bin((1+z)*(6087 - 0.5*l_width),c0,c1,Nmax):wave2bin((1+z)*(6087 + 0.5*l_width),c0,c1,Nmax)] = 0
	# night sky
	ivar[wave2bin((1+z)*(6376 - l_width),c0,c1,Nmax):wave2bin((1+z)*(6376 + l_width),c0,c1,Nmax)] = 0
	# night sky
	ivar[wave2bin((1+z)*(6307 -l_width),c0,c1,Nmax):wave2bin((1+z)*(6307 +l_width),c0,c1,Nmax)] = 0
	# SII
	ivar[wave2bin((1+z)*(6734 - l_width),c0,c1,Nmax):wave2bin((1+z)*(6734 + l_width),c0,c1,Nmax)] = 0
	# SII
	ivar[wave2bin((1+z)*(6716 - l_width),c0,c1,Nmax):wave2bin((1+z)*(6716 + l_width),c0,c1,Nmax)] = 0
	
	# Fe ?
	ivar[wave2bin((1+z)*(5317 -l_width),c0,c1,Nmax):wave2bin((1+z)*(5317 +l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(5691 -l_width),c0,c1,Nmax):wave2bin((1+z)*(5691 +l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(6504 - l_width),c0,c1,Nmax):wave2bin((1+z)*(6504 + l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(4490 - l_width),c0,c1,Nmax):wave2bin((1+z)*(4490 + l_width),c0,c1,Nmax)] = 0
	ivar[wave2bin((1+z)*(5080 -l_width),c0,c1,Nmax):wave2bin((1+z)*(5080 +l_width),c0,c1,Nmax)] = 0
	
	return ivar

def QSO_compute_FWHM(ivar,flux,wave,c0,c1,Nmax,z):
	### Constants:
	H0 = 72e3 #m s-1 Mpc-1
	c = 299792458 #m s-1

	if z<1: 
		#H_beta, need to mask OIII
		l_OIII_a = 4959
		l_OIII_b = 5007
		l_Hb = 4861
		ivar[wave2bin((1+z)*(l_OIII_a -2*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_OIII_a +2*l_width),c0,c1,Nmax)] = 0
		ivar[wave2bin((1+z)*(l_OIII_b -2*l_width),c0,c1,Nmax):wave2bin((1+z)*(l_OIII_b +2*l_width),c0,c1,Nmax)] = 0
		HB_flux = flux[wave2bin(4612*(1+z),c0,c1,Nmax):wave2bin(5112*(1+z),c0,c1,Nmax)]
		HB_wave = wave[wave2bin(4612*(1+z),c0,c1,Nmax):wave2bin(5112*(1+z),c0,c1,Nmax)]
		HB_weight = np.sqrt(ivar[wave2bin(4612*(1+z),c0,c1,Nmax):wave2bin(5112*(1+z),c0,c1,Nmax)])
		### fit a line to continuum on unmasked points
		line_coeff = np.polyfit(x = HB_wave, y = HB_flux, deg = 1, w=HB_weight)
		HB_flux_r = flux[wave2bin(4812*(1+z),c0,c1,Nmax):wave2bin(4912*(1+z),c0,c1,Nmax)]
		HB_wave_r = wave[wave2bin(4812*(1+z),c0,c1,Nmax):wave2bin(4912*(1+z),c0,c1,Nmax)]
		HB_weight_r = np.sqrt(ivar[wave2bin(4812*(1+z),c0,c1,Nmax):wave2bin(4912*(1+z),c0,c1,Nmax)])
		res =  minimize(chi2Lorenz,[4862*(1+z),10,30],args=(HB_wave_r, HB_flux_r-line_coeff[0]*HB_wave_r -line_coeff[1],HB_weight_r), method='SLSQP', bounds = [(4862*(1+z)-5,4862*(1+z)+5),(1,100),(1,10000)])
		params_beta = res.x
		FWHM = (c/1000)*2*params_beta[1]/((1+z)*l_Hb) # km s-1
		average_flux = np.mean(flux[wave2bin(5100-40,c0,c1,Nmax):wave2bin(5100+40,c0,c1,Nmax)])
		l_times_luminosity = 5100*(1e-17)*average_flux*4*np.pi*(100*parsec*1e6*(c/H0)*quad(x12,0.0,z)[0]*(1+z))**2
	elif 6.2>z>1.5:
		HB_wave = None
		#CIV
		l_CIV = 1549.0
		CIV_flux = flux[wave2bin(1300*(1+z),c0,c1,Nmax):wave2bin(1800*(1+z),c0,c1,Nmax)]
		CIV_wave = wave[wave2bin(1300*(1+z),c0,c1,Nmax):wave2bin(1800*(1+z),c0,c1,Nmax)]
		CIV_weight = np.sqrt(ivar[wave2bin(1300*(1+z),c0,c1,Nmax):wave2bin(1800*(1+z),c0,c1,Nmax)])
		### fit a line to continuum on unmasked points
		line_coeff = np.polyfit(x = CIV_wave, y = CIV_flux, deg = 1, w=CIV_weight)
		CIV_flux_r = flux[wave2bin(1500*(1+z),c0,c1,Nmax):wave2bin(1600*(1+z),c0,c1,Nmax)]
		CIV_wave_r = wave[wave2bin(1500*(1+z),c0,c1,Nmax):wave2bin(1600*(1+z),c0,c1,Nmax)]
		CIV_weight_r = np.sqrt(ivar[wave2bin(1500*(1+z),c0,c1,Nmax):wave2bin(1600*(1+z),c0,c1,Nmax)])
		res =  minimize(chi2Lorenz,[1549*(1+z),10,10],args=(CIV_wave_r, CIV_flux_r-line_coeff[0]*CIV_wave_r -line_coeff[1],CIV_weight_r), method='SLSQP', bounds = [(1549*(1+z)-5,1549*(1+z)+5),(1,100),(1,10000)])
		params_CIV = res.x
		average_flux = 1350*np.mean(flux[wave2bin(1350-40,c0,c1,Nmax):wave2bin(1350+40,c0,c1,Nmax)])
		FWHM =  (c/1000)*2*params_CIV[1]/((1+z)*l_CIV) #km s-1
		l_times_luminosity = 1350*(1e-17)*average_flux*4*np.pi*(100*parsec*1e6*(c/H0)*quad(x12,0.0,z)[0]*(1+z))**2
	else:
		FWHM = 0.0
		l_times_luminosity = 0.0
		
	return FWHM, l_times_luminosity, HB_wave

def plot_QSOLAE(RA,DEC,z,flux,wave,synflux,x0,ivar, reduced_flux,window,peak,params,params_skew, topdir, savedir, show = False, paper=True, QSOlens = True):
	if show ==False:
		mpl.use('Agg')
	# Create and save graph
	fontP = FontProperties()
	fontP.set_size('medium')	
	plt.figure(figsize=(12,3))
	plt.suptitle(SDSSname(RA,DEC)+'\n'+'RA='+str(RA)+', Dec='+str(DEC) +', $z_{QSO}='+'{:03.3}'.format(z)+ '$')
	plt.ylabel('$f_{\lambda}\, (10^{-17} erg\, s^{-1} cm^{-2}  \AA^{-1}$')
	
	if paper:
		gs = gridspec.GridSpec(1,3)
		
		smoothed_flux = np.array([np.mean(flux[ii-2:ii+3]) for ii in range(len(flux[0,:])) if (ii>4 and ii<len(flux[0,:])-4)])
		
		p1 = plt.subplot(gs[0,:2])
		#p1.plot(wave,  flux[:], 'k', label = 'BOSS Flux', drawstyle='steps-mid')
		p1.plot(wave[5:-4], smoothed_flux, 'k', label = 'eBOSS Flux', drawstyle='steps-mid')
		p1.plot(wave, synflux[:], 'r', label = 'PCA fit', drawstyle='steps-mid')
		#p1.fill_between(wave,np.min(synflux[:])-10,np.max(synflux[:])+10,where=(ivar[:]<0.001),facecolor='k', alpha=0.2)
		p1.set_ylim(np.min(synflux[:])-3, np.max(synflux[:])+3)
		p1.vlines(x = x0,ymin= -100,ymax= 100,colors= 'g',linestyles='dashed')
		box = p1.get_position()
		p1.set_position([box.x0,box.y0+0.06,box.width,box.height*0.85])
		plt.ylabel('Flux [$10^{-17} erg\, s^{-1} cm^{-2}  \AA^{-1}$]')
		plt.xlabel('Observed wavelength [$\AA$]')
		p2 = plt.subplot(gs[0,2:3])
		p2.plot(wave,  flux[:], 'k', label = 'eBOSS Flux', drawstyle='steps-mid')
		p2.plot(wave, synflux[:], 'r', label = 'PCA fit', drawstyle='steps-mid')
		p2.set_ylim(np.min(flux[window]), np.max(flux[window])+0.5)
		p2.legend(loc='upper right', bbox_to_anchor = (1.3,1.1), ncol = 1, prop=fontP)
		box = p2.get_position()
		p2.set_position([box.x0,box.y0+0.06,box.width*0.9,box.height*0.85])
		x1 = int(x0/10.)*10
		plt.xticks([x1-10,x1,x1+10,x1+20])
		p2.set_xlim(x0-15,x0+25)
		plt.xlabel('Observed wavelength [$\AA$]')
		
	else:
		
		gs = gridspec.GridSpec(2,2)
		p1 = plt.subplot(gs[0,:2])
		p1.plot(wave,  flux[:], 'k', label = 'BOSS Flux', drawstyle='steps-mid')
		p1.plot(wave, synflux[:], 'r', label = 'PCA fit', drawstyle='steps-mid')
		#p1.plot(wave,ivar[:]-15, 'g', label = 'var$^{-1}-15$')
		p1.fill_between(wave,np.min(synflux[:])-10,np.max(synflux[:])+10,where=(ivar[:]<0.000001),facecolor='k', alpha=0.2)
		p1.legend(loc='upper right', bbox_to_anchor = (1.2,1), ncol = 1, prop=fontP)
		box = p1.get_position()
		p1.set_position([box.x0,box.y0,box.width*0.9,box.height])
		p1.set_ylim(np.min(synflux[:])-3, np.max(synflux[:])+3)
		plt.ylabel('$f_{\lambda}\, [10^{-17} erg\, s^{-1} cm^{-2}  \AA^{-1}]$')
		if QSOlens == False:
			p1.set_xlim(3600,6000)
		
		
		window = np.linspace(wave2bin(x0,c0,c1,Nmax)-40,wave2bin(x0,c0,c1,Nmax)+40,81,dtype = np.int16)	
		if QSOlens:
			p3 = plt.subplot(gs[1,:1])
			p3.plot(wave,  flux[:], 'k', label = 'BOSS Flux', drawstyle='steps-mid')
			p3.plot(wave, synflux[:], 'r', label = 'PCA fit', drawstyle='steps-mid')
			p3.set_xlim(np.min(wave[window]),np.max(wave[window]))
			p3.set_ylim(np.min(synflux[window])-1, np.max(flux[window])+1)
			box = p3.get_position()
			p3.set_position([box.x0,box.y0,box.width*0.8,box.height])
			plt.ylabel('$f_{\lambda}\, [10^{-17} erg\, s^{-1} cm^{-2}  \AA^{-1}]$')
			p3.legend(loc='upper right', bbox_to_anchor = (1.4,1), ncol = 1,prop=fontP)
			p3.locator_params(axis='x',nbins=6)
			
			median_local = np.median(reduced_flux[window])
			fit_QSO = np.poly1d(np.polyfit(x=wave[window],y=reduced_flux[window],deg=3,w=(np.abs(reduced_flux[window]-median_local)<5)*np.sqrt(ivar[window])) )
			p4 = plt.subplot(gs[1,1:2])
			p4.plot(wave[window], fit_QSO(wave[window]), '-m',label = 'Order 3 fit')
			box = p4.get_position()
			p4.set_xlim(np.min(wave[window]),np.max(wave[window]))
			p4.set_position([box.x0,box.y0,box.width*0.8,box.height])
			p4.plot(wave[window], reduced_flux[window],'k', label = 'Reduced flux', drawstyle='steps-mid')
			p4.legend(loc='upper right', bbox_to_anchor = (1.4,1), ncol = 1,prop=fontP)
			p4.locator_params(axis='x',nbins=6)
		else: 
			p3 = plt.subplot(gs[1,:2])
			p3.plot(wave,  flux[:], 'k', label = 'BOSS Flux', drawstyle='steps-mid')
			p3.plot(wave, synflux[:], 'r', label = 'PCA fit', drawstyle='steps-mid')
			p3.legend(prop=fontP)
			p3.set_xlim(peak[0]-50,peak[0]+60)
			p3.set_ylim(np.min(synflux[bounds])-2, np.max(flux[bounds])+3)
			plt.ylabel('$f_{\lambda}\, [10^{-17} erg\, s^{-1} cm^{-2}  \AA^{-1}]$', fontsize=18)	
			
			
		p2 = plt.subplot(gs[2,:2])
		if QSOlens:
			p2.plot(wave[window], reduced_flux[window]-fit_QSO(wave[window]),'k', label = 'Reduced flux', drawstyle='steps-mid')
		else:
			p2.plot(wave[window], reduced_flux[window],'k', label = 'Reduced flux')
		if 0.0<peak[16]<peak[15]:
			p2.plot(wave,gauss2(x=wave,x1=params[0],x2=params[1],A1=params[2],A2=params[3],var=params[4]),'g', label = r'$\chi_D^2 = $' + '{:.4}'.format(peak[16]))
		else:
			p2.plot(wave,gauss(x=wave, x_0=params[0], A=params[1], var=params[2]),'r', label = r'$\chi_G^2 = $' + '{:.4}'.format(peak[15]) )
		if 0.0<peak[17]<peak[18]:
			p2.plot(wave,skew(x=wave,A = params_skew[0], w=params_skew[1], a=params_skew[2], eps=params_skew[3]), 'b', label =r'$\chi_S^2 = $' + '{:.4}'.format(peak[17]))
		else:
			p2.plot(wave,skew2(x=wave,A1 = params_skew[0], w1=params_skew[1], a1=params_skew[2], eps1 = params_skew[3], A2 = params_skew[4], w2=params_skew[5], a2=params_skew[6], eps2=params_skew[7]), 'c',label= r'$\chi_{S2}^2 = $' + '{:.4}'.format(peak[18]))
		box = p2.get_position()
		p2.set_position([box.x0,box.y0,box.width*0.9,box.height])
		p2.legend(loc='upper right', bbox_to_anchor = (1.2,1), ncol = 1,prop=fontP)
		plt.xlabel('$Wavelength\, [\AA]$',fontsize = 18)
		p2.set_xlim(np.min(wave[window]),np.max(wave[window]))
		if QSOlens:
			p2.set_ylim(-1, np.max(reduced_flux[window]-fit_QSO(wave[window])+1))

	make_sure_path_exists(topdir + savedir +'/plots/')
	plt.savefig(topdir + savedir +'/plots/'+SDSSname(RA,DEC)+ '-' + str(plate) + '-' + str(mjd) + '-' + str(fiberid[i]) + '-' + str(n_peak)+ '.eps', format = 'eps', dpi = 2000)
	if show:
		plt.show()
	else:
		plt.close()
		
	
def plot_QSOGal(RA,DEC,z, z_backgal,flux,wave,synflux,ivar, reduced_flux,show = False, HB_wave = 0.0, params_beta=[0,0,0], line_coeff = [0,0]):
	if show ==False:
		mpl.use('Agg')
	fontP = FontProperties()
	fontP.set_size('medium')	
	plt.suptitle(SDSSname(RA,DEC)+'\n'+'RA='+str(RA)+', Dec='+str(DEC) +', $z_{QSO}='+'{:03.3}'.format(z[i])+ '$')
	
	gs = gridspec.GridSpec(2,4)
	p1 = plt.subplot(gs[0,:4])
	
	smoothed_flux = n.array([n.mean(flux[ii-2:ii+3]) for ii in range(len(flux[:])) if (ii>4 and ii<len(flux[:])-4)])
			
	p1.plot(wave[5:-4], smoothed_flux, 'k', label = 'BOSS Flux', drawstyle='steps-mid')
	#p1.plot(wave,  flux[:], 'k', label = 'BOSS Flux')
	p1.plot(wave, synflux[:], 'r', label = 'PCA fit')
	if z[i]<1:
		p1.plot(HB_wave, lorentz(HB_wave, params_beta[0],params_beta[1],params_beta[2]) + HB_wave*line_coeff[0] + line_coeff[1], '--g')
	box = p1.get_position()

	p1.set_position([box.x0,box.y0+0.02,box.width*0.9,box.height])
	p1.set_ylim(n.min(synflux[:])-3, n.max(synflux[:])+3)
	p1.vlines(x = em_lines*(1+z_backgal),ymin= -100,ymax= 100,colors= 'g',linestyles='dashed')
	p1.legend(loc='upper right', bbox_to_anchor = (1.2,1), ncol = 1, prop=fontP)
	p1.set_xlim(3500,10500)
	plt.ylabel('Flux [$10^{-17} erg\, s^{-1} cm^{-2}  \AA^{-1}]$')
	
	p2 = plt.subplot(gs[1,:1])
	p2.vlines(x = em_lines*(1+z_backgal),ymin= -100,ymax= 100,colors= 'g',linestyles='dashed')
	loc_flux = flux[wave2bin((1+z_backgal)*(3727-10),c0,c1,Nmax) :wave2bin((1+z_backgal)*(3727+10),c0,c1,Nmax)]
	p2.plot(wave[wave2bin((1+z_backgal)*(3727-10),c0,c1,Nmax) :wave2bin((1+z_backgal)*(3727+10),c0,c1,Nmax)],loc_flux,'k', label = 'OII', drawstyle='steps-mid')
	p2.plot(wave[wave2bin((1+z_backgal)*(3727-10),c0,c1,Nmax) :wave2bin((1+z_backgal)*(3727+10),c0,c1,Nmax)],synflux[wave2bin((1+z_backgal)*(3727-10),c0,c1,Nmax) :wave2bin((1+z_backgal)*(3727+10),c0,c1,Nmax)],'r', label = 'OII', drawstyle='steps-mid')
	if loc_flux != []:
		p2.set_ylim(n.min(loc_flux)-1,n.max(loc_flux)+1)set
	plt.title('[OII] 3727')
	p2.set_xlim((1+z_backgal)*(3727-10),(1+z_backgal)*(3727+10))
	x1 = int((1+z_backgal)*3727)
	plt.xticks([x1-15,x1,x1+15])
	plt.ylabel('Flux [$10^{-17} erg\, s^{-1} cm^{-2}  \AA^{-1}]$')
	
	#If Ha is below 9500 A, show it
	if z>0.44:
		p3 = plt.subplot(gs[1,1:4])
	else: 
		p3 = plt.subplot(gs[1,1:3])
	p3.vlines(x = em_lines*(1+z_backgal),ymin= -100,ymax= 100,colors= 'g',linestyles='dashed')
	loc_flux = flux[wave2bin((1+z_backgal)*(4861-10),c0,c1,Nmax) :wave2bin((1+z_backgal)*(5007+10),c0,c1,Nmax)]
	p3.plot(wave[wave2bin((1+z_backgal)*(4861-10),c0,c1,Nmax) :wave2bin((1+z_backgal)*(5007+10),c0,c1,Nmax)],loc_flux,'k', label = 'OIII, Hb', drawstyle='steps-mid')
	p3.plot(wave[wave2bin((1+z_backgal)*(4861-10),c0,c1,Nmax) :wave2bin((1+z_backgal)*(5007+10),c0,c1,Nmax)],synflux[wave2bin((1+z_backgal)*(4861-10),c0,c1,Nmax) :wave2bin((1+z_backgal)*(5007+10),c0,c1,Nmax)],'r', label = 'OIII, Hb', drawstyle='steps-mid')
	if loc_flux != []:
		p3.set_ylim(n.min(loc_flux)-1,n.max(loc_flux)+1)
	plt.title(r'H$\beta$,[OIII] 4959, [OIII] 5007')
	plt.xlabel(r'Observed wavelength [$\AA$]')
	p3.set_xlim((1+z_backgal)*(4861-10),(1+z_backgal)*(5007+10))
	x1 = int((1+z_backgal)*4862/10.)*10
	plt.xticks([x1,x1+40,x1+80,x1+120, x1+160,x1+200])
	box = p3.get_position()
	p3.set_position([box.x0+0.02,box.y0,box.width*0.9,box.height])
	
	if z<0.44:
		p4 = plt.subplot(gs[1,3:4])
		p4.vlines(x = em_lines*(1+z_backgal),ymin= -100,ymax= 100,colors= 'g',linestyles='dashed')
		loc_flux = flux[wave2bin((1+z_backgal)*(6562-10),c0,c1,Nmax) :wave2bin((1+z_backgal)*(6562+10),c0,c1,Nmax)]
		p4.plot(wave[wave2bin((1+z_backgal)*(6562-10),c0,c1,Nmax) :wave2bin((1+z_backgal)*(6562+10),c0,c1,Nmax)],loc_flux,'k', label = 'Ha', drawstyle='steps-mid')
		p4.plot(wave[wave2bin((1+z_backgal)*(6562-10),c0,c1,Nmax) :wave2bin((1+z_backgal)*(6562+10),c0,c1,Nmax)],synflux[wave2bin((1+z_backgal)*(6562-10),c0,c1,Nmax) :wave2bin((1+z_backgal)*(6562+10),c0,c1,Nmax)],'r', label = 'Ha', drawstyle='steps-mid')
		if loc_flux != []:
			p4.set_ylim(n.min(loc_flux)-1,n.max(loc_flux)+1)
		plt.title(r'H$\alpha$')
		p4.set_xlim((1+z_backgal)*(6562-10),(1+z_backgal)*(6562+10))
		x1 = int((1+z_backgal)*6562)
		plt.xticks([x1-10,x1,x1+10])
	
	
	make_sure_path_exists(topdir + savedir +'/plots/')
	plt.savefig(topdir + savedir +'/plots/'+SDSSname(RA,DEC)+ '-' + str(plate) + '-' + str(mjd) + '-' + str(fiberid[i]) + '-'+str(k+1) +'.png')
	if show:
		plt.show()
	else:
		plt.close()

				
