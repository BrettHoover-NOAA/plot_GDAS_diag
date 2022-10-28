# Set of functions for conversions, list integrations, and plotting with GDAS diagnostic data
#
# PyGSI compliant
#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors 
from netCDF4 import Dataset
import scipy.stats as scistats
from glob import glob
import copy
#
# General Purpose Functions
#   uwdvwd_to_spddir   Convert from (u,v) to (spd,dir)
#   spddir_to_uwdvwd   Convert from (spd,dir) to (u,v)
#   calc_tstat_mean    Compute t-test significance on means between 2 samples
#   calc_fstat_mse     Compute f-test significance on mean squared error between 2 samples
#   listIntersection   Returns the list containing shared elements from 2 lists (i.e. intersection of lists)
#   read_errtable      Reads prepobs_errtable.global, returns vectors of pressure and error-value for chosen ob-type, variable
#
# Plotting Functions
#   OmF_histogram_plot Plots histograms of 2 OmF samples, provides statistics on t-test significance of
#                      differences in mean and f-test significance of differences in (r)mse
#   OmF_profile_plot   Plots vertical profiles of bias, rms(e) from 2 OmF samples, provides statistics on
#                      t-test significance of differences in bias and f-test significance of differences in
#                      (r)mse
#
def uwdvwd_to_spddir(uwd,vwd):
    spd=np.sqrt(uwd**2.+vwd**2.)
    ang=(270.-np.arctan2(vwd,uwd)*(180./np.pi))%(360.)
    return spd, ang

def spddir_to_uwdvwd(spd,ang):
    uwd=-spd*np.sin(ang*(np.pi/180.))
    vwd=-spd*np.cos(ang*(np.pi/180.))
    return uwd, vwd

def calc_tstat_mean(x,y):
    #######################################################################################
    # Computes the t-statistic to compute statistical significance
    # of the difference between two sample means.
    # ASSUMES: Gaussian distribution of samples, independence of samples
    #
    # INPUTS
    #  x: sample set 1 (numpy matrix any dimensions, will flatten)
    #  y: sample set 2 (numpy matrix any dimensions, will flatten)
    # OUTPUTS
    #  t: t-statistic of samples
    #  p: p-value of t-statistic
    # dm: difference in means
    #
    # The two samples can be considered to contain statistically significant
    # variances if p>threshold (often 0.05)
    #
    # DEPENDENCIES (PyGSI environment compliant)
    #  numpy, scipy.stats.ttest_ind (ttest_rel?)
    import numpy as np
    import scipy.stats as scistats
    #######################################################################################
    # Initialize outputs
    f = None
    p = None
    dm = None
    # Convert x, y to flattened arrays
    xf = np.ndarray.flatten(x)
    yf = np.ndarray.flatten(y)
    # Compute difference in means (y-x)
    dm = np.mean(yf)-np.mean(xf)
    # Calculate t-test statistic and p-value, presume unequal variances
    (t,p)=scistats.ttest_ind(xf,yf,equal_var=False)
    # return
    return t, p, dm

def calc_fstat_mse(x,y):
    #######################################################################################
    # Computes the F-statistic to compute statistical significance
    # of the difference between two sample mean squared errors (MSE).
    # ASSUMES: Gaussian distribution of samples, independence of samples
    # Sources:
    # https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/f-test/
    # https://www.statology.org/f-test-python/
    #
    # Sources describe the f-test for comparing variances, but it is recognized that the
    # ratio of MSE is also F-distributed.
    #
    # INPUTS
    #  x: sample set 1 (numpy matrix any dimensions, will flatten)
    #  y: sample set 2 (numpy matrix any dimensions, will flatten)
    # OUTPUTS
    #  f: f-statistic of samples
    #  p: p-value of f-statistic
    # dr: difference in root-mean-square error (sqrt(MSE))
    #
    # The two samples can be considered to contain statistically significant
    # variances if p>threshold (often 0.05)
    #
    # DEPENDENCIES (PyGSI environment compliant)
    #  numpy, scipy.stats.f.cdf
    import numpy as np
    import scipy.stats as scistats
    #######################################################################################
    # Initialize outputs
    f = None
    p = None
    dr = None
    # Convert x, y to flattened arrays, compute dof as size-1
    xf = np.ndarray.flatten(x)
    yf = np.ndarray.flatten(y)
    dofx = np.size(xf)-1
    dofy = np.size(yf)-1
    # Calculate mean squared error of each sample
    msex = np.mean(xf**2)
    msey = np.mean(yf**2)
    # Calculate difference in root mean squared error between samples (y-x)
    dr = np.sqrt(msey)-np.sqrt(msex)
    # Calculate f-test statistic: According to source, putting the higher MSE in the
    # numerator is preferred in order to force a right-tailed test, rather than doing a
    # two-tailed test. I will follow this advice here. Define dof of numerator and
    # denominator accordingly.
    if (msex > msey):
        f = msex/msey
        dofn = dofx
        dofd = dofy
    else:
        f = msey/msex
        dofn = dofy
        dofd = dofx
    # Find p-value of f-statistic test
    p = 1. - scistats.f.cdf(f,dofn,dofd)
    # return
    return f, p, dr

def listIntersection(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def read_errtable(tblfile,obtype,var):
    # Given a prepobs_errtable.global file, an ob-type, and a variable, returns vectors of
    # pressure level and assigned errors.
    #  INPUTS
    #    tblfile: filename for prepobs_errtable.global file
    #    obtype: integer observation type
    #    var: 'w' == wind (in m/s)
    #         't' == temperature (in K)
    #         'rh' == moisture (in RH)
    #         'pw' == precipitable water (in kg/m^2, or mm)
    #         'ps' == surface pressure (in hPa)
    #  OUTPUTS
    #    pvec: vector of pressures for each level of table
    #    evec: vector of error-vals for each level of table
    #
    #  NOTES:
    #   ' q' requires conversion of table from tens of percent RH to percent RH
    #    Assumes standard 33 level table from prepobs_errtable.global
    #    
    #  DEPENDENCIES: numPy
    #
    nlev=33
    # Initialize outputs as NaN arrays
    pvec=np.nan*np.ones((nlev,))
    evec=np.nan*np.ones((nlev,))
    # Identify var and assert table column for read, if not of acceptable type return NaN
    # arrays and warn
    # From Table 4.10 in:
    # https://dtcenter.org/sites/default/files/community-code/gsi/docs/users-guide/GSIUserGuide_v3.5.pdf
    if var=='t':
        ecol=1
    elif var=='rh':
        ecol=2
    elif var=='w':
        ecol=3
    elif var=='ps':
        ecol=4
    elif var=='pw':
        ecol=5
    else:
        print('var '+var+' not acceptable type [t,rh,w,ps,pw], exiting')
        return pvec, evec
    #
    with open(tblfile,'r') as tblhdl:
        try:
            # Dump entire content of file to string
            tbls=tblhdl.read()
            # Define search-string for obtype
            srch_str=str(obtype)+' OBSERVATION TYPE'
            # Find header-line based on srch_str up to newline
            hdr_beg=tbls.find(srch_str,0,len(tbls))
            hdr_end=tbls.find('\n',hdr_beg)
            # Initialize last_end as end of header
            last_end=hdr_end
            # Loop over nlev
            for i in range(nlev):
                # Define table-line from character following last_end to newline
                tbl_beg=last_end+1
                tbl_end=tbls.find('\n',tbl_beg)
                tbl_line=tbls[tbl_beg:tbl_end]
                # Extract pvec from 0th element and evec from ecol-th element
                pvec[i]=float(tbl_line.split()[0])
                evec[i]=float(tbl_line.split()[ecol])
                # If ecol==2, this is in RH/10 space, convert to RH space
                if ecol==2:
                    evec[i]=10.*evec[i]
                # Assign end of table-line to last_end
                last_end=tbl_end
            return pvec, evec
        except:
            print('error in table-read, exiting')
            return pvec, evec

def OmF_histogram_plot(OmF1,OmF2,name1='SET1',name2='SET2',titleStr='TITLE',figax=None):
    # Inputs:
    #   OmF1: vector of OmF values for set-1
    #   OmF2: vector of OmF values for set-2
    #   name1: string for set-1 name on plot
    #   name2: string for set-2 name on plot
    #   titleStr: string for figure (panel) title
    #   figax: figure axis, or None for no specified axis (generate a figure, axis)
    # Outputs:
    #   figreturn: returned axis, if figax specified, otherwise returned figure
    #
    #
    # Example use: No figax provided, returns figure handle
    #
    # fighdl=OmF_histogram_plot(omf_old,omf_new,name1='old data',name2='new data',titleStr='OMF (old vs new)')
    # fighdl.savefig('old_vs_new.png')
    #
    # Example use: figax provided for each of 2-panels, returns axes
    #
    # fighdl,axhdls=plt.subplots(nrows=1,ncols=2,figsize=(20,9))
    # leftAx=OmF_histogram_plot(old_umg,new_umg,name1='old data',name2='new data',titleStr='OmF U',figax=axhdls[0])
    # rightAx=OmF_histogram_plot(old_vmg,new_vmg,name1='old data',name2='new data',titleStr='OmF V',figax=axhdls[1])
    # fighdl.savefig('old_vs_new_2panel.png')
    #
    # Compute ob-count for each set
    nObs1=np.size(OmF1)
    nObs2=np.size(OmF2)
    # Compute appropriate OmF range for plot
    OmFmin=np.min([np.min(OmF1),np.min(OmF2)])
    OmFmax=np.max([np.max(OmF1),np.min(OmF2)])
    OmFrng=np.arange(OmFmin,OmFmax,(OmFmax-OmFmin)/100.)
    # Compute means, difference in means, and t-test values
    OmF_tt_val, OmF_tt_p, OmF_mean_dif = calc_tstat_mean(OmF1,OmF2)
    OmF1_mean = np.mean(OmF1)
    OmF2_mean = np.mean(OmF2)
    # Compute rmses, difference in rmses, and f-test values (on mses)
    OmF_ft_val, OmF_ft_p, OmF_rmse_dif = calc_fstat_mse(OmF1,OmF2)
    OmF1_rmse = np.sqrt(np.mean(OmF1**2.))
    OmF2_rmse = np.sqrt(np.mean(OmF2**2.))
    # Generate plot axes
    if figax==None:
        pax=plt.figure(figsize=(9,9))
        figax=pax.add_axes([0.,0.,1.,1.])
        figreturn=pax
    else:
        ax=figax
        figreturn=figax
    # Plot histogram of OmF1 and OmF2 across OmFrng
    figax.hist(OmF1,OmFrng,color='blue',alpha=0.65,density=True)
    figax.hist(OmF2,OmFrng,color='orange',alpha=0.65,density=True)
    figax.set_title(titleStr)
    figax.text(
         0.05 , -0.1  ,
         name1+' RMSE: {:.2f}'.format(OmF1_rmse) ,
         transform=figax.transAxes ,
         fontsize=18
        )
    figax.text(
         0.60 , -0.1  ,
         name1+' MEAN: {:.2f}'.format(OmF1_mean) ,
         transform=figax.transAxes ,
         fontsize=18
        )
    if OmF_rmse_dif>0:
        sym='+'
    else:
        sym=''
    figax.text(
         0.05 , -0.15 ,
         name2+' RMSE: {:.2f} ('.format(OmF2_rmse)+sym+'{:.4f})'.format(OmF_rmse_dif) ,
         transform=figax.transAxes ,
         fontsize=18
        )
    if OmF_mean_dif>0:
        sym='+'
    else:
        sym=''
    figax.text(
         0.60 , -0.15 ,
         name2+' MEAN: {:.2f} ('.format(OmF2_mean)+sym+'{:.4f})'.format(OmF_mean_dif) ,
         transform=figax.transAxes ,
         fontsize=18
        )
    figax.text(
         0.05 , -0.2  ,
         'P-VAL MSE: {:.2f}'.format(OmF_ft_p) ,
         transform=figax.transAxes ,
         fontsize=18
         )
    figax.text(
         0.60 , -0.2  ,
         'P-VAL MEAN: {:.2f}'.format(OmF_tt_p) ,
         transform=figax.transAxes ,
         fontsize=18
         )
    figax.legend([name1+' ({:d} obs)'.format(nObs1),name2+' ({:d} obs)'.format(nObs2)])
    
    return figreturn

def OmF_profile_plot(OmF1,pre1,OmF2,pre2,name1='SET1',name2='SET2',titleStr='TITLE',figax=None):
    # Inputs:
    #   OmF1: vector of OmF values for set-1
    #   pre1: vector of pressure values for set-1
    #   OmF2: vector of OmF values for set-2
    #   pre2: vector of pressure values for set-2
    #   name1: string for set-1 name on plot
    #   name2: string for set-2 name on plot
    #   titleStr: string for profile figure (panel) title
    #   figax: profile figure axis, or None for no specified axis (generate a figure, axis)
    # Outputs:
    #   figreturn: returned axis, if figax specified, otherwise returned figure
    #
    #
    # Example use: No figax provided, returns figure handle
    #
    # fighdl=OmF_profile_plot(old_umg,old_pre,new_umg,new_pre,name1='old data',name2='new data',
    #                         titleStr='OmF U')
    # fighdl.savefig('old_vs_new.png')
    #
    # Example use: figax provided for 1 of 2-panels, returns axis (second axis is blank in this ex)
    #
    # fighdl,axhdls=plt.subplots(nrows=1,ncols=2,figsize=(20,9))
    # filledaxs=OmF_profile_plot(old_umg,old_pre,new_umg,new_pre,name1='old data',name2='new data',
    #                            titleStr='OmF U',figax=axs[0])
    # fighdl.savefig('old_vs_new_2panel.png')
    #
    # Define pressure layers: 50 hPa layers from 1025–125 hPa
    pre_edge=np.arange(1025.,125.1,-50.)
    pre_maxs=pre_edge[0:-1]
    pre_mins=pre_edge[1:]
    pre_mids=0.5*(pre_maxs+pre_mins)
    n_plevs=np.size(pre_mids)
    # Initialize OmF rmse and bias profiles, and ob-counts, for OmF1 and OmF2
    OmF1_rmse_prof=np.nan*np.zeros((n_plevs,))
    OmF1_bias_prof=np.nan*np.zeros((n_plevs,))
    OmF1_nobs_prof=np.nan*np.zeros((n_plevs,))
    OmF2_rmse_prof=np.nan*np.zeros((n_plevs,))
    OmF2_bias_prof=np.nan*np.zeros((n_plevs,))
    OmF2_nobs_prof=np.nan*np.zeros((n_plevs,))
    # Initialize rmse and bias significance-test profiles
    rmse_ft_prof=np.nan*np.zeros((n_plevs,))
    bias_tt_prof=np.nan*np.zeros((n_plevs,))
    # Loop over pressure levels
    for i in range(n_plevs):
        # Define minimum/maximum pressure on level, collect observations
        # within pressure interval for OmF1 and OmF2
        pmin=pre_mins[i]
        pmax=pre_maxs[i]
        idx1=np.where((pre1<pmax)&(pre1>=pmin))[0]
        idx2=np.where((pre2<pmax)&(pre2>=pmin))[0]
        # Compute nobs for each profile at level i
        OmF1_nobs_prof[i]=np.size(idx1)
        OmF2_nobs_prof[i]=np.size(idx2)
        # If at least 3 observations, produce rmse and bias for OmF1 on level i
        if np.size(idx1)>3:
            OmF1_rmse_prof[i]=np.sqrt(np.mean(OmF1[idx1]**2.))
            OmF1_bias_prof[i]=np.mean(OmF1[idx1])
        # If at least 3 observations, produce rmse and bias for OmF2 on level i
        if np.size(idx2)>3:
            OmF2_rmse_prof[i]=np.sqrt(np.mean(OmF2[idx2]**2.))
            OmF2_bias_prof[i]=np.mean(OmF2[idx2])
        # If both sets have at least 3 observations, produce significance-tests on level i
        if (np.size(idx1)>3)&(np.size(idx2)>3):
            OmF_ft_val, OmF_ft_p, OmF_rmse_diff = calc_fstat_mse(OmF1[idx1],OmF2[idx2])
            OmF_tt_val, OmF_tt_p, OmF_bias_diff = calc_tstat_mean(OmF1[idx1],OmF2[idx2])
            rmse_ft_prof[i]=OmF_ft_p
            bias_tt_prof[i]=OmF_tt_p
    # Generate plot axes
    if (figax==None):
        pax=plt.figure(figsize=(16,9))
        ax1=pax.add_axes([0.04,0.,0.45,1.])
        figreturn=pax
    else:
        ax1=figax
        figreturn=ax1
    # ax1: rmse, bias, and significance profiles
    ax1.plot(OmF1_rmse_prof,pre_mids,color='blue',linewidth=3.)
    ax1.plot(OmF1_bias_prof,pre_mids,color='blue',linewidth=3.,linestyle='--')
    ax1.plot(OmF2_rmse_prof,pre_mids,color='orange',linewidth=3.)
    sig_idx=np.where(rmse_ft_prof<0.05)[0]
    if (np.size(sig_idx)>0):
        ax1.plot(OmF2_rmse_prof[sig_idx],pre_mids[sig_idx],marker='*',color='red',markersize=10.,linestyle='none',label='_nolegend_')
    ax1.plot(OmF2_bias_prof,pre_mids,color='orange',linewidth=3.,linestyle='--')
    sig_idx=np.where(bias_tt_prof<0.05)[0]
    if (np.size(sig_idx)>0):
        ax1.plot(OmF2_bias_prof[sig_idx],pre_mids[sig_idx],marker='*',color='red',markersize=10.,linestyle='none',label='_nolegend_')
    ax1.plot(np.zeros((n_plevs,)),pre_mids,color='black',linewidth=1.,label='_nolegend_')
    ax1.set_yticks(ticks=pre_mids[::2])
    ax1.invert_yaxis()
    ax1.set_title(titleStr)
    ax1.legend([name1+' rmse',name1+' bias',name2+' rmse',name2+' bias'])
    return figreturn

def count_profile_plot(pre1,pre2,name1='SET1',name2='SET2',titleStr='Ob Count (thousands)',figax=None):
    # Inputs:
    #   pre1: vector of pressure values for set-1
    #   pre2: vector of pressure values for set-2
    #   name1: string for set-1 name on plot
    #   name2: string for set-2 name on plot
    #   titleStr: string for profile figure (panel) title
    #   figax: profile figure axis, or None for no specified axis (generate a figure, axis)
    # Outputs:
    #   figreturn: returned axis, if figax specified, otherwise returned figure
    #
    #
    # Example use: No figax provided, returns figure handle
    #
    # fighdl=count_profile_plot(old_pre,new_pre,name1='old data',name2='new data',
    #                         titleStr='Counts (thousands): U')
    # fighdl.savefig('old_vs_new.png')
    #
    # Example use: figax provided for 1 of 2-panels, returns axis
    #
    # fighdl,axhdls=plt.subplots(nrows=1,ncols=2,figsize=(20,9))
    # filledaxs=OmF_profile_plot(old_pre,new_pre,name1='old data',name2='new data',
    #                            titleStr='Counts (thousands): U',figax=axs[1])
    # fighdl.savefig('old_vs_new_2panel.png')
    #
    # Define pressure layers: 50 hPa layers from 1025–125 hPa
    pre_edge=np.arange(1025.,125.1,-50.)
    pre_maxs=pre_edge[0:-1]
    pre_mins=pre_edge[1:]
    pre_mids=0.5*(pre_maxs+pre_mins)
    n_plevs=np.size(pre_mids)
    # Initialize OmF rmse and bias profiles, and ob-counts, for OmF1 and OmF2
    Nobs1_prof=np.nan*np.zeros((n_plevs,))
    Nobs2_prof=np.nan*np.zeros((n_plevs,))
    # Loop over pressure levels
    for i in range(n_plevs):
        # Define minimum/maximum pressure on level, collect observations
        # within pressure interval for OmF1 and OmF2
        pmin=pre_mins[i]
        pmax=pre_maxs[i]
        idx1=np.where((pre1<pmax)&(pre1>=pmin))[0]
        idx2=np.where((pre2<pmax)&(pre2>=pmin))[0]
        # Compute nobs for each profile at level i
        Nobs1_prof[i]=np.size(idx1)
        Nobs2_prof[i]=np.size(idx2)
    # Generate plot axes
    if (figax==None):
        pax=plt.figure(figsize=(16,9))
        ax2=pax.add_axes([0.54,0.,0.45,1.])
        figreturn=pax
    else:
        ax2=figax
        figreturn=ax2
    # ax2: Ob-count by layer
    ax2.barh(y=pre_mids-5,width=0.001*Nobs1_prof,height=10.0,facecolor='blue')
    ax2.barh(y=pre_mids+5,width=0.001*Nobs2_prof,height=10.0,facecolor='orange')
    ax2.set_yticks(ticks=pre_mids[::2])
    ax2.set_title(titleStr)
    ax2.legend([name1,name2])
    ax2.invert_yaxis()
    return figreturn

