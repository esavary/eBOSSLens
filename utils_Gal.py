import matplotlib as mpl

mpl.use('Agg')

from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
from utils import *

def plot_GalaxyLens(doublet,RA,DEC,plate,fiberid,mjd,z,z_err,obj_class,wave, reduced_flux, zline,fit,topdir,savedir, peak_candidates, doublet_index, c0,c1,Nmax, show = False):
	#if show==False:
		#mpl.use('Agg')
		
	if doublet != True: 
		ax = plt.subplot(1,1,1)
		plt.title('RA='+str(RA)+', Dec='+str(DEC)+', Plate='+str(plate)+', Fiber='+str(fiberid)+', MJD='+str(mjd)+'\n$z='+str(z)+' \pm'+str(z_err)+'$, Class='+str(obj_class))
		ax.plot(wave, reduced_flux[:],'k')
		plt.xlabel('$Wavelength\, (Angstroms)$')
		plt.ylabel('$f_{\lambda}\, (10^{-17} erg\, s^{-1} cm^{-2} Ang^{-1}$')
		ax.plot(wave,fit,'r')
		make_sure_path_exists(topdir + savedir +'/plots/')
		plt.savefig(topdir + savedir +'/plots/' + str(plate) + '-' + str(mjd) + '-' + str(fiberid) + '.png')
		if show:
			plt.show()
		else:
			plt.close()
	
	# If doublet, plot in two different windows
	if doublet==True:
		x_doublet = 0.5*(peak_candidates[doublet_index][9]+ peak_candidates[doublet_index][10])
		bounds = np.linspace(wave2bin(x_doublet,c0,c1,Nmax)-10,wave2bin(x_doublet,c0,c1,Nmax)+10,21,dtype = np.int16)
		f = open(topdir + savedir +'/doublet_ML.txt','a')
		f.write('\n' +  str(plate) + ' ' + str(mjd) + ' ' + str(fiberid) + ' ' + str(reduced_flux[bounds]))
		f.close()
		
		plt.figure(figsize=(14,6))
		ax1 = plt.subplot2grid((1,3), (0,0), colspan=2)
		plt.suptitle('RA='+str(RA)+',
		 Dec='+str(DEC)+', Plate='+str(plate)+', Fiber='+str(fiberid)+', MJD='+str(mjd)+'\n$z='+str(z)+' \pm'+str(z_err)+'$, Class='+str(obj_class))
		ax2 = plt.subplot2grid((1,3), (0,2))
		ax1.plot(wave[10:-10], reduced_flux[10:-10],'k')
		ax1.plot(wave,fit,'r')
		ax1.set_xlabel('$\lambda \, [\AA]$ ')
		ax1.set_ylabel('$f_{\lambda}\, (10^{-17} erg\, s^{-1} cm^{-2} Ang^{-1}$')
		ax2.set_xlabel('$\lambda \, [\AA]$ ')
		ax2.locator_params(tight=True)
		
		ax2.set_xlim([peak_candidates[doublet_index][0]-30,  peak_candidates[doublet_index][0]+30])
		ax2.plot(wave, reduced_flux[:],'k')
		ax2.plot(wave,fit,'r')
		ax2.set_ylim([-5,10])
		
		ax2.vlines(x = zline['linewave']*(1+z),ymin= -10,ymax= 10,colors= 'g',linestyles='dashed')
		#ax1.vlines(x = zline['linewave']*(1+z),ymin= -10,ymax= 10,colors= 'g',linestyles='dashed')
		ax1.set_xlim([np.min(wave),np.max(wave)])
		
		make_sure_path_exists(topdir + savedir +'/plots/')
		plt.savefig(topdir + savedir +'/plots/' + str(plate) + '-' + str(mjd) + '-' + str(fiberid) + '.png')
		if show:
			plt.show()
		else:
			plt.close()
	return
