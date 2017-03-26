  # Romain A. Meyer, 2016, LASTRO, EPFL
# Spectroscopic lensing systems detection tool 
# Detects OII doublet for galaxy-galaxy lensing systems
# Searches for Lyman alpha emission for galaxy-LAE or QSO-LAE systems
# Switch between the cases using booleans below (lines 30-32)
# 'Paper' mode removes intermediate steps plots useful for analysis but irrelevant for publication
# To change list of plate to analyse: line 34





import sys



#Get the plate list name , save directory , and option plots if passed as arguments:

if len(sys.argv)==4:
	listname='/'+ sys.argv[1]
	dossname='/'+ sys.argv[2]
	plotarg= sys.argv[3]
else :
	 listname ='list.txt'
	 dossname = ''
	 plotarg = 'no'
	

# Imports
import numpy as n
import pyfits as pf
import matplotlib as mpl

if plotarg != 'yes':
	mpl.use('Agg')


from matplotlib import pyplot as plt
from matplotlib import gridspec
import math
from scipy.integrate import quad
from scipy import special as sp
from scipy import interpolate
import datetime
import copy
from matplotlib import rcParams
import os
import errno
import itertools as it




# ------------------
from utils import *
from utils_QSO import *
from utils_Gal import *



#-----------------------------------------------------------------------------------------------------







# Operation mode
searchLyA = False
QSOlens = False
paper = True
Jackpot = False
# BOSS or eBOSS data?
BOSS = True
eBOSS = False
# show graphs or not? (do not use show on PC clusters!!!)

if plotarg=='yes' :
	
	plot_show = True
	
else:
	
	plot_show = False
	


#Set topdir,savedir:
topdir = '..'
savedir = '/QSO-LAE/BOSS/paper_selec'  + dossname
#-------- give file in [plate mjd] format of plates you want to inspect -----------------
plates_list = listname
#-------- give file in [plate mjd fiber] format of specific objects you want to inspect -----------------
inspect_candidates = False
#candidates = n.loadtxt('../QSO-Gal/QSO-Gal_selec2/best/selec.txt')
#----------------------------------------------------------------------------------------------------
#------------------------------------ INITIALIZATION ------------------------------------------------
plate_mjd = [line.strip().split() for line in open(topdir + savedir + plates_list)]
plate = 0
fiberid = [0]
if searchLyA == True and QSOlens == True:
	f = open(topdir + savedir  + dossname + '/candidates_QSO_LyA.txt','w+')
	f.close()
elif searchLyA == True and QSOlens == False:
	f = open(topdir + savedir  +'/candidates_LAE.txt','w+')
	f.close()
elif searchLyA == False and QSOlens == True:
	f = open(topdir + savedir  + '/candidates_QSO.txt','w+')
	f.close()
elif Jackpot:
	f = open(topdir + savedir + '/candidates_Jackpot.txt','w+')
	f.close()
elif searchLyA == False and QSOlens == False:
	f = open(topdir + savedir + '/candidates_doublet.txt','a')
	f.close() 
	f = open(topdir + savedir + '/candidates_DM.txt','a')
	f.close()
	f = open(topdir + savedir + '/candidates_multi.txt','a')
	f.close()

#Set of emission lines used for lensed galaxy detection: OII, Hb, OIII, OIII, Ha
em_lines = n.array([3726.5,4861.325,4958.911,5006.843,6562.801])
# Constants:
l_LyA = 1215.668 #Angstroms

######---- Needed for QSO lenses (QSO in DR12 with proper classificaiton)
DR12Q = DR12Q_extractor(path = './Superset_DR12Q.fits')

# Counters used to check the numbers of candidates at key step
counter1 = 0
counter2 = 0
counter3 = 0
counter4 = 0

#if plot_show == False:
	#mpl.use('Agg')

#-----------------------------------------------------------------------------------------------------

#Loop over plates
#for j in n.arange(len(plate_mjd)):
for j in n.arange(len(plate_mjd)):
	# Loading the data --------------------------------------------------------------------------------
	flux = 0
	mjd = plate_mjd[j][1]
	plate = plate_mjd[j][0]
	c0,c1,wave,flux,ivar,vdisp, synflux,fiberid, RA, DEC, obj_id, obj_class, \
		obj_type, z, zwarning,z_err, spectroflux, rchi2, rchi2diff, zline,npix = load_data(mjd = mjd, plate=plate, BOSS = True, eBOSS = False, logdir = '../../../../../../SCRATCH/' )
	#-----------------------------------------------------------------------------------------------------
	
	reduced_flux = n.array(flux - synflux)
	sqrtivar=copy.deepcopy(ivar)
	
	Nmax = len(flux[0,:])

	#Mask BOSS spectra glitches
	ivar[:,wave2bin(5570,c0,c1,Nmax): wave2bin(5590,c0,c1,Nmax)] = 0
	ivar[:,wave2bin(5880,c0,c1,Nmax): wave2bin(5905,c0,c1,Nmax)] = 0
	
	#------------- Loop over fibers------------------------------------------------------------------
	for i in  n.arange(len(flux[:,0])):
	#Using above file, allow to look only at certain plate-mjd-fiber 
		#candidates_list = [x for x in candidates if (int(x[0])==int(plate) and int(x[1])==int(mjd) and  int(x[2])== int(fiberid[i]))]
		#if len(candidates_list)== 0 and inspect_candidates : commenter
			#continue
		#print plate,mjd,fiberid[i]
		if QSOlens:
			
			redshift_warning = False
			# Take only QSOs
			if (not(obj_class[i] == 'QSO   ') or obj_type[i] =='SKY             ' or 'SPECTROPHOTO_STD'==obj_type[i]): 
				continue
			
			# Reject badly fitted QSOs
			if (zwarning[i] != 0 or rchi2[i]>10 or z_err[i] < 0):
				continue	
			index = [x for x in DR12Q if (int(x[0])==int(plate) and int(x[1])==int(mjd) and int(x[2])==int(fiberid[i]))]
			
			if len(index)>0:
				if n.abs(index[0][3]-z[i])<0.005 and n.abs(index[0][4]-z[i])>0.1:
					continue
			else:
				redshift_warning = True
			
			### Mask typical width
			l_width = 15
			### Before masking, compute the FWHM of CIV or HBeta depending on redshift:
			FWHM,l_times_luminosity, HB_wave = QSO_compute_FWHM(ivar = ivar[i,:],flux = flux[i,:], wave = wave[i,:],c0=c0,c1=c1,Nmax=Nmax,z =z[i])
			
			M_BH = 10**(6.91 + n.log10(n.sqrt(5100*l_times_luminosity/1e44)*(FWHM/1000)**2)) # Masses solaires
			sigma_host = 200*10**((n.log10(M_BH) - 7.92)/3.93) ### km s-1

			ivar = masks_QSO(ivar=ivar,z=z[i])	
			
		else:
			if (obj_class[i] == 'STAR  ' or obj_class[i] == 'QSO   ' or obj_type[i] =='SKY             ' or 'SPECTROPHOTO_STD'==obj_type[i]): 
				continue
				
		
		peaks = []
		peak_number = len(peaks)
		doublet = None
		
		counter1 = counter1 +1;
		
		sqrtivar[i,:] = n.sqrt(ivar[i,:])
		
		### Bolton 2004: S/N of maximum likelihood estimator of gaussian peaks
		if searchLyA == True:
			width = 30.0
			sig = 2.17
		elif searchLyA == False:
			width = 30.0
			sig = 2.4
		## Prepare normalized gaussian
		NormGauss = gauss(n.linspace(-width*0.5,width*0.5,width),0.0,1.0,sig**2)
		NormGauss = NormGauss/n.sum(NormGauss)
		Cj1 = n.array([n.sum(reduced_flux[i,:]*kernel(j+0.5*width,width,NormGauss,len(wave))*ivar[i,:]) for j in range(int(len(wave)-width))])
		Cj2 = n.array([n.sum(ivar[i,:]*kernel(j+0.5*width,width,NormGauss,len(wave))**2) for j in range(int(len(wave)-width))])
		SN = n.zeros(len(wave))
		SN[width*0.5:len(wave)-width*0.5] = Cj1/n.sqrt(Cj2)
                        

		if searchLyA == True and QSOlens == True:
			peak_candidates = n.array([(x0,0.0,0.0,0.0,0.0,0.0,test,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0) for x0,test in zip(wave,SN) if (test>8.0 and  (l_LyA*(1+z[i])+300)<x0<8000)])
		elif searchLyA == False and QSOlens == True:
			peak_candidates = n.array([(x0,0.0,0.0,0.0,0.0,0.0,test) for x0,test in zip(wave,SN) if (test>6.0 and  (l_LyA*(1+z[i])+300)<x0<9500)])
		elif searchLyA == True and QSOlens == False:
			peak_candidates = n.array([(x0,0.0,0.0,0.0,0.0,0.0,test,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0) for x0,test in zip(wave,SN) if (test>8.0 and  3600<x0<4800)])
		elif Jackpot == True:
			# x0 free free z1 z2 Quad_SN2 SN0->Quad_SN1
			peak_candidates = n.array([(x0,0.0,0.0,0.0,0.0,0.0,test) for x0,test in zip(wave,SN) if test>8.0])
		elif searchLyA == False and QSOlens == False:
			peak_candidates = n.array([(x0,0.0,0.0,0.0,test,0.0,0.0,0.0,0.0,0.0,0.0) for x0,test in zip(wave,SN) if test>6.0])
			
		else:
			continue
		#(Legend key:) wavelength(x0) chisq_doublet amp_gauss var_gauss S/N chisq_doublet amp1_doublet amp2_doublet var_doublet x1 x2
		#Keep only center of candidate peaks
		k = 0
			

		if searchLyA == False and QSOlens == False:
			while (k < (len(peak_candidates)-1)):
				if (abs(peak_candidates[k][0] - peak_candidates[k+1][0]) < 10):
					if peak_candidates[k][4] < peak_candidates[k+1][4]:
						peak_candidates = n.delete(peak_candidates,k, axis=0)
						k = k-1
					else:
						peak_candidates = n.delete(peak_candidates,k+1,axis = 0)
						k = k-1					
				k = k+1
				#?
		else:
			while (k < (len(peak_candidates)-1)):
				if (abs(peak_candidates[k][0] - peak_candidates[k+1][0]) < 10):
					if peak_candidates[k][6] < peak_candidates[k+1][6]:
						peak_candidates = n.delete(peak_candidates,k, axis=0)
						k = k-1
					else:
						peak_candidates = n.delete(peak_candidates,k+1,axis = 0)
						k = k-1					
				k = k+1
		chisq = 10000
		chisq2 = 10000
		chisq_skew = 10000
		chi2_width = 10000	
	
		#### check hits are not from foreground galaxy or badly fitted QSO
		if QSOlens and searchLyA:
			foreground_z = False
			compare = it.combinations(em_lines,len(peak_candidates))
			for group in compare:
				for k in range(len(peak_candidates)):
					for j in range(k+1,len(peak_candidates)):
						if ( n.abs(peak_candidates[k][0]/group[k] - peak_candidates[j][0]/group[j]) < 0.01 and 0.0<peak_candidates[k][0]/group[k]-1.0 < (z[i] - 0.05) ):
							foreground_z = True
			if foreground_z:
				continue


		#Search for suitable peak candidates
		for peak in peak_candidates:
			x0 = peak[0]
			if nearline(x0, zline, fiberid[i], z[i], int(mjd), int(plate)):
				continue

 			bounds = n.linspace(wave2bin(x0,c0,c1,Nmax)-15,wave2bin(x0,c0,c1,Nmax)+15,31,dtype = n.int16)
			# Fit QSO continuum and check if signal is reduced or not (i.e. check if line detection is produced by large features)
			if QSOlens:
				window = n.linspace(wave2bin(x0,c0,c1,Nmax)-40,wave2bin(x0,c0,c1,Nmax)+40,81,dtype = n.int16)
				median_local = n.median(reduced_flux[i,window])
				fit_QSO = n.poly1d(n.polyfit(x=wave[window],y=reduced_flux[i,window],deg=3,w=(n.abs(reduced_flux[i,window]-median_local)<5)*n.sqrt(ivar[i,window])) )
				new_flux = reduced_flux[i,window] - fit_QSO(wave[window])

				cj1_new = n.sum(new_flux*kernel(int(len(window)/2),width,NormGauss,len(new_flux))*ivar[i,window])
				cj2_new = n.sum(ivar[i,window]*kernel(int(len(window)/2),width,NormGauss,len(window))**2)
				SN_fitted = cj1_new/n.sqrt(cj2_new)
				if  searchLyA and (SN_fitted < 0):
					continue
				elif searchLyA and SN_fitted > 0: 
					peak[19] = SN_fitted 
					reduced_flux[i,window]=new_flux
				elif searchLyA == False and SN_fitted < 4:
					continue
				elif searchLyA == False and SN_fitted > 4:
					peak[3] = SN_fitted 
					reduced_flux[i,window]=new_flux

		
			#### Special case: QSOlens with background galaxies	
			if searchLyA == False and QSOlens==True and Jackpot == False:
				for l in em_lines:
					test_z = peak[0]/l - 1.0 
					if test_z > z[i]:
						quad_SN = 0.0
						for w in em_lines:
							center_bin = wave2bin(w*(1+test_z),c0,c1,Nmax)
							SN_line = n.array(SN[center_bin-2:center_bin+2])
							quad_SN += max(SN_line*(SN_line>0))**2
						quad_SN = n.sqrt(quad_SN)
						
						if quad_SN > peak[5]:
							peak[5] = quad_SN
							peak[4] = test_z 	
				continue
			### Special case: Jackpot lenses 
			if Jackpot == True:

				first_lens = False
				peak[5] = peak[6]
				peak[4] = peak[6]
				for l in em_lines:
					test_z = peak[0]/l -1.0
					if test_z > z[i]+0.05:
						quad_SN_1 = 0.0
						for w in em_lines:
							center_bin = wave2bin(w*(1+test_z),c0,c1,Nmax)
							SN_line = n.array(SN[center_bin-2:center_bin+2])*(not(nearline(w*(1+test_z), zline, fiberid[i], z[i], int(mjd), int(plate))))
							quad_SN_1 += max(SN_line*(SN_line>0))**2
						quad_SN_1 = n.sqrt(quad_SN_1)
						if quad_SN_1 > peak[6] + 6:
							peak[5] = quad_SN_1
							peak[2] = test_z 
							first_lens = True
				if first_lens:
					for peak2 in peak_candidates:
						if n.abs(peak2[0]- peak[0])> 5:
							for l in em_lines:
								test_z_2 = peak2[0]/l -1.0
								if test_z_2 > z[i]+ 0.05 and abs(test_z-test_z_2)> 0.05:
									quad_SN_2 = 0.0
									for w in em_lines:
										center_bin = wave2bin(w*(1+test_z_2),c0,c1,Nmax)
										SN_line = n.array(SN[center_bin-2:center_bin+2])*(not(nearline(w*(1+test_z_2), zline, fiberid[i], z[i], int(mjd), int(plate))))
										quad_SN_2 += max(SN_line*(SN_line>0))**2
									quad_SN_2 = n.sqrt(quad_SN_2)
									if quad_SN_2 > peak[5] + 6:
										peak[4] = quad_SN_2
										peak[3] = test_z_2
				continue
				

			#Single Line: Gaussian fit around x_0
			if searchLyA == True:
				init = [x0,4,6]
				res =  minimize(chi2g,init,args=(wave[bounds], reduced_flux[i,bounds],ivar[i,bounds]), method='SLSQP', bounds = [(x0-2,x0+2),(1,100),(1,15)])
			elif searchLyA == False and QSOlens ==False:
				init = [x0,1,2]
				#res =  minimize(chi2g,init,args=(wave[bounds], reduced_flux[i,bounds],ivar[i,bounds]), method='SLSQP', bounds = [(x0-2,x0+2)(0.1,5),(1,8)])
				res =  minimize(chi2g,init,args=(wave[bounds], reduced_flux[i,bounds],ivar[i,bounds]), method='SLSQP', bounds = [(x0-2,x0+2),(0.1,5),(1,8)])
			params = res.x
			chisq = res.fun

			#Check for not too high chi square and save
			if (not(chisq > 2.0) and searchLyA == False and QSOlens == False):
				peak[1] = chisq
				peak[2] = params[1]
				peak[3] = params[2]
				peak[0] = params[0]
			elif searchLyA:
				peak[1] = params[0]
				peak[2] = params[1]
				peak[3] = params[2]
				#eq_Width = quad(gauss,x0-200,x0+200,args=(params[0],params[1],params[2]))
				chi2_width = chisq
				peak[15] = chisq
				
			#Doublet OII: Gaussian fit around x_0
			if (x0 > 3727.0*(1+z[i]) or searchLyA and QSOlens == False): 

				res2 = minimize(chi2D,[1.0,5,1.0,x0-1.5,x0+1.5],args=(wave[bounds], reduced_flux[i,bounds],ivar[i,bounds]), method='SLSQP', bounds = [(0.1,5),(1,8),(0.1,5),(x0-7,x0),(x0,x0+7)])
				params2 = res2.x
				chisq2 = res2.fun
				
				if  (searchLyA == False and 0.5*x0/3726.5<abs(params2[3]-params2[4])<2.1*x0/3726.5 and not(chisq2 > 2)):					
					peak[5] = chisq2
					peak[6] = params2[0] #amp1
					peak[7] = params2[2] #amp2
					peak[8] = params2[1] #var
					peak[9] = params2[3] #x1
					peak[10] = params2[4] #x2
				elif searchLyA and chisq2<chisq:
					peak[1] = params2[3] #x1
					peak[2] = params2[4] #x2 
					peak[3] = params2[0] #amp1
					peak[4] = params2[2] #amp2
					peak[5] = params2[1] #var
					chi2_width = chisq2
					peak[16] = chisq2
				elif searchLyA:
					peak[16] = chisq2
					# Delta OII restframe: 	1.3 A  (3725.94 3727.24)
						
			# If looking at LAE, test a skew-normal profile as well 
			if searchLyA:
				# Sharp blue, red tail
				init_skew = [params[1],0.5,2,x0]
				res_skew = minimize(chi2skew,init_skew,args=(wave[bounds], reduced_flux[i,bounds],ivar[i,bounds]), method='SLSQP', bounds = [(2,50),(0.0,10),(1,10),(x0-4,x0+4)])
				params_skew_a = res_skew.x
				chisq_skew_a = res_skew.fun
				# Double skew symmetric
				init_skew = [params[1],0.5,-2,x0, params[1]/2,0.5,2,x0+8]
				res_skew = minimize(chi2skew2,init_skew,args=(wave[bounds], reduced_flux[i,bounds],ivar[i,bounds]), method='SLSQP', bounds = [(2,50),(0.0,10),(-10,-1),(x0-6,x0+6), (2,50),(0.0,10),(1,10),(x0-15,x0+15)])
				params_skew_c = res_skew.x
				chisq_skew_c = res_skew.fun
				
				# Sharp red, blue tail
				init_skew = [params[1],0.5,-2,x0]
				res_skew = minimize(chi2skew,init_skew,args=(wave[bounds], reduced_flux[i,bounds],ivar[i,bounds]), method='SLSQP', bounds = [(2,50),(0.0,10),(-10,-1),(x0-4,x0+4)])
				params_skew_b = res_skew.x
				chisq_skew_b = res_skew.fun
				
				if chisq_skew_b < chisq_skew_a: 
					params_skew = params_skew_b
					chisq_skew = chisq_skew_b
				elif chisq_skew_a < chisq_skew_b: 
					params_skew = params_skew_a
					chisq_skew = chisq_skew_a
				if chisq_skew < chisq_skew_c:
					peak[7] = params_skew[0] #A
					peak[8] = params_skew[1] #w
					peak[9] = params_skew[2] #a
					peak[10] = params_skew[3] #eps
					if chisq_skew < chi2_width:
						chi2_width = chisq_skew
				else:
					peak[7] = params_skew_c[0] #A1
					peak[8] = params_skew_c[1] #w1
					peak[9] = params_skew_c[2] #a1
					peak[10] = params_skew_c[3] #eps1
					peak[11] = params_skew_c[4] #A2
					peak[12] = params_skew_c[5] #w2
					peak[13] = params_skew_c[6] #a2
					peak[14] = params_skew_c[7] #eps2
					if chisq_skew_c < chi2_width:
						chi2_width = chisq_skew_c
						
				peak[17] = chisq_skew	
				peak[18] = chisq_skew_c

				#put back reduced flux by adding again 3rd order fit (plotting purpose)
				if QSOlens:
					reduced_flux[i,window]= new_flux + fit_QSO(wave[window])
				
		counter2 = counter2 + 1;					

		if searchLyA == False and QSOlens == False and Jackpot == False:
		#Finding peak with lowest chi square for doublet and see if it is better fitted by single line or not
			doublet_index = 0
			chi2saved = 1000.0
		# Find the doublet index
			for k in range(len(peak_candidates)):
				peak = peak_candidates[k]
				if (peak[1]>peak[5]>0 and peak[5]< chi2saved and peak[0]<9200) :
					peak[1] = peak[5]
					chi2saved = peak[5]
					doublet = True
					doublet_index = k
		
			#Removing candidates that were not fitted : params still 0
			peak_candidates = n.array([peak for peak in peak_candidates if (peak[2]!=0 or peak[5]==peak[1]!=0)])
			if len(peak_candidates) == 0:
				continue
			
			counter3 = counter3+1;
			#Sorting candidates by chi square
		
			peak_candidates = sorted(peak_candidates, key=lambda peak: peak[1])
			
			# Keeping only 5 most likely candidates
			if len(peak_candidates) > 5:
				peak_candidates = peak_candidates[0:5]
			#find again doublet index
			found = False
			for k in range(len(peak_candidates)):
				if (peak_candidates[k][5] == chi2saved):
					doublet_index = k
					found = True
			if found==False:
				doublet = False

		elif searchLyA == False and QSOlens == True and Jackpot == False:
			peak_candidates = n.array([peak for peak in peak_candidates if peak[5]>(1.5+peak[6])])
			if len(peak_candidates) == 0:
				continue
			peak_candidates = sorted(peak_candidates, key=lambda peak: peak[5])
			if len(peak_candidates) > 3:
				peak_candidates = peak_candidates[0:3]
		elif Jackpot:
			peak_candidates = n.array([peak for peak in peak_candidates if peak[3]>0.0])
			if len(peak_candidates) == 0:
					continue
			peak_candidates = sorted(peak_candidates, key=lambda peak: peak[5])
			if len(peak_candidates) > 3:
				peak_candidates = peak_candidates[0:3]
		
		if len(peak_candidates) == 0:
			print('length peak_candidate =0')
			continue	
		
		
		# Check that at least 1 candidate is below 9200 Angstrom cut, if not, go to next fiber
		below_9200 = False	
		for peak in peak_candidates:
			if peak[0] < 9200:
				below_9200 = True
		if below_9200 == False:
			continue	
		
		counter4 = counter4+1;
		#Try to infer background redshift
		detection = False
		score = 0.0

		if (doublet == True and searchLyA == False and QSOlens==False and Jackpot == False):
			fileD = open(topdir + savedir +  '/candidates_doublet.txt','a')
			z_s = peak_candidates[doublet_index][0]/3727.24 - 1.0
			if (z_s > z[i]+0.05):
				detection = True
				score += peak_candidates[doublet_index][5]
				fileD.write('\n' +str([radEinstein(z[i],z_s,vdisp[i]*1000), score,z_s, RA[i], DEC[i], int(plate), int(mjd), fiberid[i],peak_candidates[doublet_index][9]]))
				fileD.close()
			if len(peak_candidates):	
				fileDM = open(topdir + savedir  + '/candidates_DM.txt','a')
				confirmed_lines = []
				#Generating all combinations of lines from above list to compare with candidates
				temp = [peak for peak in peak_candidates if peak[1]!=peak[5]]
				compare = em_lines[1:5]
				if z_s > z[i]+0.05 :
					for peak in temp:
						for line in compare:
							if (abs(peak[0]/line -1 - z_s) < 0.01):
								detection = True
								confirmed_lines.append(line)
								score+= peak[5]
					if (confirmed_lines != []):
						fileDM.write('\n'+str([radEinstein(z[i],z_s,vdisp[i]*1000), score,z_s, RA[i], DEC[i], int(plate), int(mjd), fiberid[i],confirmed_lines]))
					fileDM.close()
		elif (doublet != True and len(peak_candidates) > 1 and searchLyA == False and QSOlens==False and Jackpot == False):
			compare = it.combinations(em_lines,len(peak_candidates))
			confirmed_lines = []
			fileM = open(topdir + savedir  +'/candidates_multi.txt','a')
			for group in compare:
				for k in range(len(peak_candidates)):
					for j in range(k+1,len(peak_candidates)):
						if ( abs(peak_candidates[k][0]/group[k] - peak_candidates[j][0]/group[j]) < 0.01 and peak_candidates[k][0]/group[k]-1.0 > (z[i] + 0.05) ):
							detection = True
							z_s = peak_candidates[k][0]/group[k]-1.0
							confirmed_lines.append([group, peak_candidates[k][0]/group[k]-1.0])
							score+= peak_candidates[j][4]**2+peak_candidates[k][4]**2
			if (confirmed_lines != []):
				fileM.write('\n'+str([radEinstein(z[i],z_s,vdisp[i]*1000), score, z_s, RA[i], DEC[i], int(plate), int(mjd), fiberid[i],confirmed_lines]))
			fileM.close()
		elif searchLyA == False and QSOlens == True and Jackpot==False:
			for k in range(len(peak_candidates)):				
				
				z_backgal = peak_candidates[k][4]
				peak = peak_candidates[k]
				# compute OII,OIII flux (of lensed galaxy)
				temp_fluxes_OII = n.zeros(5)
				temp_fluxes_OIII = n.zeros(5)
				dwave = n.array([wave[gen_i+1]-wave[gen_i] if gen_i<len(wave)-1 else 0 for gen_i in range(len(wave))])
				if z_backgal < 1:
					for j in range(4,9):
						temp_bounds = n.linspace(wave2bin((1+z_backgal)*3727,c0,c1,Nmax)-j,wave2bin((1+z_backgal)*3727,c0,c1,Nmax)+j,2*j+1,dtype = n.int16)
						temp_fluxes_OII[j-4] = n.sum((flux[i,temp_bounds]-synflux[i,temp_bounds])*dwave[temp_bounds])
						temp_bounds = n.linspace(wave2bin((1+z_backgal)*5007,c0,c1,Nmax)-j,wave2bin((1+z_backgal)*5007,c0,c1,Nmax)+j,2*j+1,dtype = n.int16)
						temp_fluxes_OIII[j-4] = n.sum((flux[i,temp_bounds]-synflux[i,temp_bounds])*dwave[temp_bounds])
					#print dwave
				OII_flux = n.median(temp_fluxes_OII)
				OIII_flux = n.median(temp_fluxes_OIII)
				
				
				fileQSO = open(topdir + savedir +  '/candidates_QSO.txt','a')
				fileQSO.write('\n' + str([RA[i], DEC[i], int(plate), int(mjd), fiberid[i], z[i], peak[0],peak[4],peak[5],peak[6],spectroflux[i,1], spectroflux[i,3], OII_flux, OIII_flux ]))
				fileQSO.close()
				
				plot_QSOGal(RA = RA[i],DEC= DEC[i],z=z[i], z_backgal= z_backgal,flux=flux[i,:],wave=wave,synflux=synflux[i,:],ivar= ivar[i,:], \
					reduced_flux = reduced_flux[i,:],show = plot_show, HB_wave = HB_wave , params_beta=params_beta, line_coeff =line_coeff)

		elif Jackpot == True:
			for peak in peak_candidates:				
				
				fileJ = open(topdir + savedir +  '/candidates_Jackpot.txt','a')
				fileJ.write('\n' + str([RA[i], DEC[i], int(plate), int(mjd), fiberid[i], z[i], peak[2],peak[3],peak[4],peak[5],peak[6],spectroflux[i,1], spectroflux[i,3]]))
				fileJ.close()
				fontP = FontProperties()
				fontP.set_size('medium')	
				plt.suptitle(SDSSname(RA[i],DEC[i])+'\n'+'RA='+str(RA[i])+', Dec='+str(DEC[i]) +', $z_{QSO}='+'{:03.3}'.format(z[i])+ '$')
				
				gs = gridspec.GridSpec(1,4)
				p1 = plt.subplot(gs[0,:4])
				
				smoothed_flux = n.array([n.mean(flux[i,ii-2:ii+3]) for ii in range(len(flux[0,:])) if (ii>4 and ii<len(flux[0,:])-4)])
		
				p1.plot(wave[5:-4], smoothed_flux, 'k', label = 'BOSS Flux', drawstyle='steps-mid')
				#p1.plot(wave,  flux[i,:], 'k', label = 'BOSS Flux')
				p1.plot(wave, synflux[i,:], 'r', label = 'PCA fit')
				box = p1.get_position()
				p1.set_position([box.x0,box.y0+0.02,box.width*0.9,box.height])
				p1.set_ylim(n.min(synflux[i,:])-3, n.max(synflux[i,:])+3)
				p1.vlines(x = em_lines*(1+peak[2]),ymin= -100,ymax= 100,colors= 'g',linestyles='dashed')
				p1.vlines(x = em_lines*(1+peak[3]),ymin= -100,ymax= 100,colors= 'b',linestyles='dashed')
				p1.legend(loc='upper right', bbox_to_anchor = (1.2,1), ncol = 1, prop=fontP)
				p1.set_xlim(3500,10500)
				plt.ylabel('Flux [$10^{-17} erg\, s^{-1} cm^{-2}  \AA^{-1}]$')
				
				make_sure_path_exists(topdir + savedir +'/plots/')
				#plt.show()
				plt.savefig(topdir + savedir +'/plots/'+SDSSname(RA[i],DEC[i])+ '-' + str(plate) + '-' + str(mjd) + '-' + str(fiberid[i]) + '-'+str(k+1) +'.png')
				plt.close()

	
		elif searchLyA==True and QSOlens==True and Jackpot == False:
			if QSOlens:
				fileLyA = open(topdir + savedir +  '/candidates_QSO_LyA.txt','a')
			else:
				fileLyA = open(topdir + savedir  + '/candidates_LAE.txt','a')
			n_peak = 1
			for peak in peak_candidates:
				x0 = peak[0]
				z_O2 = x0/3727.24 - 1.0
				# compute SN of Ha, Hb, OIII to conclude if LyA candidate is or not OII at low-redshift
				SNlines = 0
				em_lines = n.array([4861.325,5006.843,6562.801])
				for l in em_lines:
					center_bin = wave2bin(l*(1+z_O2),c0,c1,Nmax)
					SNlines += max(SN[center_bin-2:center_bin+2])**2
				SNlines = n.sqrt(SNlines)
				
				if SNlines < 100:
					if n.abs(peak[9]) > n.abs(peak[13]):
						skewness = peak[9]
					else: 
						skewness = peak[13]
					params = peak[1:6]
					params_skew = peak[7:15]
					if not(n.any(params) and n.any(params_skew)):
						continue
					
					bounds = n.linspace(wave2bin(x0,c0,c1,Nmax)-15,wave2bin(x0,c0,c1,Nmax)+15,31,dtype = n.int16)
 					
		 			#compute equivalent width before manipulating the spectra
					dwave = n.array([wave[gen_i+1]-wave[gen_i] if gen_i<len(wave)-1 else 0 for gen_i in range(len(wave))])
					eq_Width = n.sum((flux[i,bounds]/synflux[i,bounds]-1)*dwave[bounds])
					# compute LyA flux
					temp_fluxes = n.zeros(5)
					for j in range(4,9):
						temp_bounds = n.linspace(wave2bin(x0,c0,c1,Nmax)-j,wave2bin(x0,c0,c1,Nmax)+j,2*j+1,dtype = n.int16)
						temp_fluxes[j-4] = n.sum((flux[i,temp_bounds]-synflux[i,temp_bounds])*dwave[temp_bounds])
					lyA_flux = n.median(temp_fluxes)
					
					#compute skewness indicator (on reduced flux but without 3rd order fit (which is meant for bad fit QSO and not present in final candidates)
					I = n.sum(reduced_flux[i,bounds])
					xmean = n.sum(reduced_flux[i,bounds]*wave[bounds])/I
					sigma2 = n.sum(reduced_flux[i,bounds]*(wave[bounds] - xmean)**2)/I
					S = n.sum(reduced_flux[i,bounds]*(wave[bounds] - xmean)**3)/(I*n.sign(sigma2)*n.abs(sigma2)**1.5)
		
					local_wave = wave[bounds]
					local_flux = reduced_flux[i,bounds]
					peak_index = local_flux.argmax()
					F = 0.1*local_flux[peak_index]
					#Find blue and red 10% peak flux
					f = 10*F
					k = peak_index
					while f > F and 0<k:
						k = k-1
						f = local_flux[k]
					a = (local_flux[k+1]-local_flux[k])/(local_wave[k+1]-local_wave[k]) 
					b = local_flux[k] - a*local_wave[k] 
					l_blue_10 = (F-b)/a
					k = peak_index
					f = 10*F
					while f > F and k<len(local_flux)-1:
						k = k+1
						f = local_flux[k]
					a = (local_flux[k]-local_flux[k-1])/(local_wave[k]-local_wave[k-1]) 
					b = local_flux[k-1] - a*local_wave[k-1]
					l_red_10 = (F-b)/a
					Sw = S*(l_red_10-l_blue_10)
					
					a_lambda = (l_red_10-local_wave[peak_index])/(local_wave[peak_index]-l_blue_10)
					# save the parameters
					fileLyA.write('\n' + str([peak[0],peak[6],z[i], SNlines ,RA[i], DEC[i], int(plate), int(mjd), fiberid[i],params[0],params[1],params[2],params[3],params[4],
									params_skew[0],params_skew[1],params_skew[2],params_skew[3],params_skew[4],params_skew[5],params_skew[6],params_skew[7],peak[15],peak[16],peak[17],peak[18], skewness, S, Sw, a_lambda,rchi2[i],peak[19],spectroflux[i,1], spectroflux[i,3],lyA_flux]))
					# Make the graph
					plot_QSOLAE(RA= RA[i],DEC = DEC[i],z=z[i],flux=flux[i,:],wave=wave,synflux=synflux[i,:],x0= x0, ivar = ivar[i,:], reduced_flux = reduced_flux[i,:],window=window,peak =peak,
						params = params,params_skew=params_skew, topdir = topdir, savedir = savedir, show = plot_show, paper = paper, QSOlens = QSOlens)
					
				n_peak = n_peak +1
			fileLyA.close()
		# Save surviving candidates (Galaxy-Galaxy case)
		if searchLyA == False and QSOlens == False and Jackpot == False:
			for k in range(len(peak_candidates)):
				peak = peak_candidates[k]
				if (k == doublet_index and doublet ==True):
					# Save doublet gaussians
					peaks.append([peak[9], peak[6], peak[8]])
					peaks.append([peak[10], peak[7], peak[8]])
				else:
					# Save emission line 
					peaks.append([peak[0], peak[2], peak[3]])
		peak_number = len(peak_candidates)	
		
		#Graphs OII doublet
		if ((peak_number>1 or doublet==True) and below_9200 and detection and searchLyA==False and QSOlens==False and Jackpot == False):
			#Computing total fit of all peaks
			fit=0
			for k in n.arange(len(peaks)):
				fit = fit + gauss(wave, x_0 = peaks[k][0] , A=peaks[k][1], var=peaks[k][2])

			plot_GalaxyLens(doublet = doublet,RA = RA[i],DEC = DEC[i],plate = plate,fiberid=fiberid[i],mjd=mjd,z=z[i],z_err =z_err[i], \
				obj_class = obj_class[i],wave = wave, reduced_flux=reduced_flux[i,:], zline = zline ,fit = fit,topdir = topdir, savedir = savedir, peak_candidates =peak_candidates, \
				doublet_index = doublet_index, c0 = c0,c1 = c1,Nmax = Nmax, show = plot_show)
