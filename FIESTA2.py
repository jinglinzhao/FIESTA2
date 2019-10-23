# FIESTA #

import numpy as np
import os
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from functions import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


N_section 	= 7											# equally divide power spectrum into #N_section
freq_HN 	= 0.1563									# higher limit of frequency range

# jitter 		= np.loadtxt('/Volumes/DataSSD/SOAP_2/outputs/02.01/RV.dat')
# jitter 		= (jitter - np.mean(jitter))

FILE 		= sorted(glob.glob('./fits/*.fits'))
N_file 		= len(FILE)
hdulist     = fits.open(FILE[0])
CCF_tpl     = 1 - hdulist[0].data 						# flip the line profile
V 			= (np.arange(401)-200)/10					# CCF Velocity grid
idx_ccf		= (abs(V) <= 10)
v 			= V[idx_ccf]
ccf_tpl 	= CCF_tpl[idx_ccf]
power_tpl, phase_tpl, freq = ft(ccf_tpl, 0.1)

idx 		= (freq <= freq_HN)

n_idx 		= len(freq[idx])
power_int 	= np.zeros(n_idx) 							# integrated power
for i in range(n_idx):
	power_int[i] = np.trapz(power_tpl[:i+1], x=freq[:i+1])

per_portion = power_int[n_idx-1]/N_section
freq_LH 	= np.zeros(N_section+1)

# Section 1: [0, freq_LH[1]; section 2: [freq_LH[1], freq_LH[2]]...
for i in range(N_section):
	freq_LH[i] = max(freq[idx][power_int<=per_portion*i])
freq_LH[N_section] = freq_HN


#------------------#
# Line deformation #
#------------------#

RV_gauss 	= np.zeros(N_file)							# RV derived from a Gaussian fit
RV_FT 		= np.zeros((N_file, N_section))
delta_RV 	= np.zeros((N_file, N_section))

# plt.rcParams.update({'font.size': 12})
# fig, axes 	= plt.subplots(figsize=(12, 12))

for n in range(N_file):
	hdulist     = fits.open(FILE[n])
	CCF         = 1 - hdulist[0].data 					# flip the line profile
	ccf 		= CCF[idx_ccf]
	popt, pcov 	= curve_fit(gaussian, v, ccf)
	RV_gauss[n] = popt[1]

	power, phase, freq = ft(ccf, 0.1)
	for i in range(N_section):
		RV_FT[n, i] = rv_ft(freq_LH[i], freq_LH[i+1], freq, phase-phase_tpl, power_tpl)
		delta_RV[n, i] = RV_FT[n, i] - RV_gauss[n] * 1000

mean_Gaussian = np.mean(RV_gauss)
mean_delta_RV = np.mean(delta_RV)
RV_gauss = (RV_gauss - np.mean(RV_gauss))*1000
delta_RV = delta_RV - np.mean(delta_RV)

#---------------------------#
# Multiple Regression Model # 
#---------------------------#

from sklearn import linear_model
regr = linear_model.LinearRegression()

msk = np.random.rand(len(RV_gauss)) < 0.2
# msk = (np.arange(100) < 50)
train_x = delta_RV[msk, :]
train_y = RV_gauss[msk]
test_x = delta_RV[~msk, :]
test_y = RV_gauss[~msk]


regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ', regr.intercept_)

#------------#
# Prediction # 
#------------#


TEST 		= sorted(glob.glob('./test/*.fits'))
RV_gauss 	= np.zeros(len(TEST))
RV_FT 		= np.zeros((len(TEST), N_section))
delta_RV 	= np.zeros((len(TEST), N_section))


for n in range(len(TEST)):
	hdulist     = fits.open(TEST[n])
	CCF         = 1 - hdulist[0].data 					# flip the line profile
	ccf 		= CCF[idx_ccf]
	popt, pcov 	= curve_fit(gaussian, v, ccf)
	RV_gauss[n] = popt[1]
	power, phase, freq = ft(ccf, 0.1)
	for i in range(N_section):
		RV_FT[n, i] = rv_ft(freq_LH[i], freq_LH[i+1], freq, phase-phase_tpl, power_tpl)
		delta_RV[n, i] = RV_FT[n, i] - RV_gauss[n] * 1000
RV_gauss = (RV_gauss - mean_Gaussian)*1000
delta_RV = delta_RV - mean_delta_RV
test_x = delta_RV
test_y = RV_gauss

y_hat= regr.predict(delta_RV)
rss = np.mean((y_hat - test_y) ** 2)
print('rms: %.2f' % rss**0.5)




# #############################################################################
# Compute paths

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# #############################################################################



# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(test_x, test_y))

plt.plot(test_y, y_hat, '.')
plt.show()

plt.plot(test_y, y_hat-test_y, '.')
plt.show()

















# plt.savefig('./outputs/Overview.png')

if 0:
	# ---------------------------- #
	# Radial velocity correlations #
	# ---------------------------- #
	RV_gauss 	= (RV_gauss - RV_gauss[0]) * 1000 			# all radial velocities are relative to the first ccf
	fig, axes 	= plt.subplots(figsize=(15, 5))
	plt.subplots_adjust(wspace=wspace)
	plot_correlation(RV_gauss, RV, RV_L, RV_H)
	plt.savefig('./outputs/RV_FT.png')
	plt.close('all')


# plt.plot(RV_FT); plt.show()


# plt.plot(jitter, RV_FT, '.'); plt.show()