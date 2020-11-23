import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style("ticks")
import pickle
import arviz as az
import pymc3 as pm

from matplotlib.colors import to_rgb

import scipy.stats as stats 

from IPython.display import display

import matplotlib as mpl

from collections import defaultdict 

def doScatterPlotAlpha(pred,mu,sigma,colorstring,label,alphaMin,alphaMax,sizeMin,sizeMax):
    # convert to flat arrays
    x_flat, y_flat, alpha_arr = predictiveSamples2_X_Y_Alpha_1D(alphaMin,alphaMax,pred)
    x_flat, y_flat, size_arr = predictiveSamples2_X_Y_Alpha_1D(sizeMin,sizeMax,pred)
    
    # convert to original space
    y_plot = mu + sigma*y_flat
    
    # compute color and alpha 
    r, g, b = to_rgb(colorstring)
    colors = [(r, g, b, alpha) for alpha in alpha_arr]
    
    # do scatter plot
    plt.scatter(x_flat,y_plot,c=colors,s=size_arr,label=label);

def doScatterPlotAlphaLines(pred,mu,sigma,colorstring,label,alphaMin,alphaMax):
    # convert to original space
    y = mu + sigma*pred
    
    # compute median and percentiles
    l = np.percentile(y, 2.5, axis=0)
    m = np.percentile(y, 50, axis=0)
    u = np.percentile(y, 92.5, axis=0)
        
    # plot median 
    plt.plot(m,label=label,color=colorstring,alpha=alphaMax)
    
    # plot boundaries 
    #alphaBound = 0.5*(alphaMin+alphaMax)
    #plt.plot(l,color=colorstring,alpha=alphaBound,ls='--')
    #plt.plot(u,color=colorstring,alpha=alphaBound,ls='--')
    
    # fill
    x = np.arange(len(m))
    alphaFill = alphaMin
    plt.fill_between(x,l,m,alpha=alphaFill,color=colorstring)
    plt.fill_between(x,m,u,alpha=alphaFill,color=colorstring)

def plotPriorPosteriorB(widthInch,heigthIch,dpi,sizes,writeOut,path,dictMeanStd,pm_data,yname):
        
    SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE = sizes
    
    mu,sigma = dictMeanStd[yname]
    trans = lambda x: sigma*x
    
    var_names = ['{}_b1'.format(yname)]
    
    for var in var_names:
        axes = az.plot_dist_comparison(pm_data,var_names=var,transform=trans,figsize=(widthInch,heigthIch),textsize=SMALL_SIZE);
        
        if writeOut:
            plt.savefig(path + "prior_posterior_{}.pdf".format(var),dpi=dpi)

def plotPosterior(widthInch,heigthInch,dpi,writeOut,path,dictMeanStd,pm_data,yname):
    # get list of varnames to plot as intersection of targeted and existing ones 
    target_var_names=['{}_b0'.format(yname),'{}_b1'.format(yname),'{}_b2'.format(yname)]
    varnames_trace = list(pm_data.posterior.data_vars.keys())
    intersec = list(set(target_var_names) & set(varnames_trace))  
    
    mu,sigma = dictMeanStd[yname]
    trans = lambda x: sigma*x+mu
    
    az.plot_posterior(pm_data,var_names=intersec,transform=trans,figsize=(2*widthInch,3*heigthInch),hdi_prob=0.95);
    
    if writeOut:
        plt.savefig(path + "posterior_b_{}.pdf".format(yname),dpi=dpi)

def plotTracesB(widthInch,heigthIch,dpi,writeOut,path,trace,yname):
    # get list of varnames to plot as intersection of targeted and existing ones 
    target_var_names=['{}_b0'.format(yname),'{}_b1'.format(yname),'{}_b2'.format(yname)]
    varnames_trace = trace.varnames
    intersec = list(set(target_var_names) & set(varnames_trace))
    
    pm.traceplot(trace,var_names=intersec,figsize=(widthInch,heigthIch));
    if writeOut:
        plt.savefig(path + "trace_{}.pdf".format(yname),dpi=dpi)
    
        
def plotOverlayDataInformation(fig,dfData):
    # ==== Get boundaries of datasets
    # get dataset names
    names = dfData.Dataset.unique()
    
    # get indicies
    dictNameIndices = dict()
    for name in names:
        indices = dfData[dfData.Dataset == name].index.values
        dictNameIndices[name] = (indices[0],indices[-1]) 
    
    # ==== Get boundaries of treatments
    treat = dfData.Treatment.unique()
   
    dictTreatIndices = dict()
    for t in treat:
        indices = dfData[dfData.Treatment == t].index.values
        dictTreatIndices[t] = (indices[0],indices[-1])
            
    (ymin,ymax) = plt.gca().get_ylim()
    
    
    for (name,(a,b)) in dictNameIndices.items():
        plt.axvline(x=a,ymin=0.0,ymax=0.14,color='k',ls='-',lw=0.5,alpha=0.5)        
        plt.text(0.5*(a+b), 0.07*(ymax-ymin)+ymin, name, horizontalalignment='center',verticalalignment='center',)

        # draw last line
        if name == list(dictNameIndices.keys())[-1]:
            plt.axvline(x=b,ymin=0.0,ymax=0.14,color='k',ls='-',lw=0.5,alpha=0.5)
    
            
        
    for (t,(a,b)) in dictTreatIndices.items():                
        plt.axvline(x=a,ymin=0.8,ymax=1.0,color='k',ls='-',lw=0.5,alpha=0.5)        
        plt.text(0.5*(a+b), 0.8*(ymax-ymin)+ymin, t,rotation=90, horizontalalignment='center',verticalalignment='bottom',size=8)
        
        # draw last line
        if t == list(dictTreatIndices.keys())[-1]:
            plt.axvline(x=b,ymin=0.8,ymax=1.0,color='k',ls='-',lw=0.5,alpha=0.5)

def plotPriorPredictive(widthInch,heigthIch,dpi,writeOut,path,df,dictMeanStd,prior_pred,y,yname):
        
    fig = plt.figure(figsize=(widthInch,heigthIch),dpi=dpi, facecolor='w');

    mu,sigma = dictMeanStd[yname]
    
    #ymin = np.min(y) - 1* np.abs(np.min(y))
    #ymax = np.max(y) + 1* np.abs(np.max(y))
    #plt.ylim([sigma*ymin+mu,sigma*ymax+mu])
    
    #plt.xlabel("Sample number");
    plt.ylabel("{}".format(yname));

    plt.plot(sigma*y+mu,'k.',label='Data', fillstyle='none');
    doScatterPlotAlphaLines(prior_pred["{}_y".format(yname)],mu,sigma,'lightblue','Prior predicitive',0.4,0.99)
    
    plotOverlayDataInformation(fig,df)
    
    plt.legend(bbox_to_anchor=(0.78, -0.05),ncol=3)
    
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    
    if writeOut:
        plt.savefig(path + "prior_predicitive_{}.pdf".format(yname),dpi=dpi)

def plotPriorPosteriorPredictive(widthInch,heigthIch,dpi,writeOut,path,df,dictMeanStd,prior_pred,post_pred,y,yname):
    
    fig = plt.figure(figsize=(widthInch,heigthIch), dpi= dpi, facecolor='w');

    mu,sigma = dictMeanStd[yname]
    
    #ymin = np.min(y) - 1* np.abs(np.min(y))
    #ymax = np.max(y) + 1* np.abs(np.max(y))
    #plt.ylim([sigma*ymin+mu,sigma*ymax+mu])
    
    #plt.xlabel("Sample number");
    plt.ylabel("{}".format(yname));

    
    doScatterPlotAlphaLines(prior_pred["{}_y".format(yname)],mu,sigma,'lightblue','Prior predictive',0.4,0.99)
    plt.plot(sigma*y+mu,'k.',label='Data', fillstyle='none');
    doScatterPlotAlphaLines(post_pred["{}_y".format(yname)],mu,sigma,'lightgreen','Posterior predictive',0.4,0.99)
    
    plotOverlayDataInformation(fig,df)
    
    plt.legend(bbox_to_anchor=(0.95, -0.05),ncol=3)
    
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    
    if writeOut:
        plt.savefig(path + "prior_posterior_predicitive_{}.pdf".format(yname),dpi=dpi)

def plotContrast(widthInch,heigthIch,dpi,writeOut,path,dictMeanStd,x1contrast_dict,trace,yname):
    
    mu_Val,sig_Val = dictMeanStd[yname]
    b1_sample = sig_Val*trace['{}_b1'.format(yname)]
    
    for key, value in x1contrast_dict.items():
        contrast = np.dot(b1_sample, value)        
        pm.plot_posterior(contrast, ref_val=0.0, bins=50,hdi_prob=0.95,figsize=(widthInch,heigthIch));
        plt.title('Contrast {} on {}'.format(key,yname));
        
        if writeOut:
            plt.savefig(path + "contrast_{}_{}.pdf".format(key,yname),dpi=dpi)

def plotDiagnostics(widthInch,heigthInch,dpi,writeOut,path,trace,dataTrace,yname):
    # Report divergences 
    #diverging = trace['diverging']
    #print('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
    #diverging_pct = diverging.nonzero()[0].size / len(trace) * 100
    #print('Percentage of Divergent Chains: {:.1f}'.format(diverging_pct))
    
    # Report issues with r_hat
    #print("\n \n \nVariables with r_hat > 1.05")
    #display(pm.summary(dataTrace).query("r_hat > 1.05").round(2))
    
    # look for correlations in posterior
    #print("\n \n \nCorrelations in posterior")
    az.plot_pair(dataTrace,var_names=['b1'],filter_vars='like',figsize=(2.0*widthInch,2.0*heigthInch),divergences=True,marginals=True);
    if writeOut:
        plt.savefig(path + "posterior_pair_b1_{}.pdf".format(yname),dpi=dpi)
        
    az.plot_pair(dataTrace,var_names=['b1','b2'],filter_vars='like',figsize=(2.0*widthInch,2.0*heigthInch),divergences=True,marginals=True);
    if writeOut:
        plt.savefig(path + "posterior_pair_b2_{}.pdf".format(yname),dpi=dpi)
    
    # look for correlations in posterior
    #print("\n \n \nDivergences in posterior")
    az.plot_parallel(dataTrace,var_names=['b0','b1','b2'],filter_vars='like',figsize=(2.0*widthInch,heigthInch), norm_method='normal');
    if writeOut:
        plt.savefig(path + "posterior_parallel_{}.pdf".format(yname),dpi=dpi)    
    
    # forest plot
    #print("\n \n \nForest plot of posterior")
    az.plot_forest(dataTrace,var_names=['b0','b1','b3','M'],filter_vars='like',figsize=(widthInch,5*heigthInch),hdi_prob=0.95,ess=True,r_hat=True);
    if writeOut:
        plt.savefig(path + "posterior_forest_{}.pdf".format(yname),dpi=dpi)
        
def getPercentiles(y,lower=2.5,middle=50.,upper=92.5,axis=0):
    l = np.percentile(y, lower, axis=axis)
    m = np.percentile(y, middle, axis=axis)
    u = np.percentile(y, upper, axis=axis)

    return l,m,u

def plotLevels(widthInch,heigthInch,dpi,sizes,writeOut,path,dictMeanStd,dictTreatment,dictSoftware,trace,yname,x1,x2):
    fig = plt.figure(figsize=(widthInch,heigthInch), dpi= dpi)
    
    SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE = sizes
    
    mu_Val,sig_Val = dictMeanStd[yname]
    
    b1l,b1m,b1u = getPercentiles(sig_Val*trace['{}_b1'.format(yname)])
    b2l,b2m,b2u = getPercentiles(sig_Val*trace['{}_b2'.format(yname)])
    M12l,M12m,M12u = getPercentiles(sig_Val*trace['{}_M12'.format(yname)])
     
    widthUpDown = 0.1
    widthLevel = 0.4
    
    # prepare color dict:
    # use groups of 4 colors, as in tab20c
    colorIndex = dict({5:0,6:1,7:2,0:4,1:5,4:6,10:7,2:8,3:9,8:10,9:11})
    
    for lvl2 in np.arange(len(np.unique(x2))):
        # compute color
        color = mpl.cm.tab20c(colorIndex[lvl2])
        
        # ===  treatment
        # up / down
        x0,x1 = 0,widthUpDown
        plt.plot([x0,x1],[0,b2m[lvl2]],color=color);
        
        # horizontal
        x1,x2 = widthUpDown,widthUpDown+widthLevel
        x2l,x2r = x1+0.3*widthLevel, x2-0.3*widthLevel
        
        plt.plot([x1,x2l],[b2m[lvl2],b2m[lvl2]],color=color);     
        plt.plot([x2r,x2],[b2m[lvl2],b2m[lvl2]],color=color);
        
        # label
        y1 = b2m[lvl2]
        plt.text(0.5*(x1+x2), y1, dictTreatment[lvl2],
                 rotation=0,
                 horizontalalignment='center',
                 verticalalignment='center',
                size=SMALL_SIZE)
    
        # ===  treatment & software
        # up / down
        x2,x3 = widthUpDown+widthLevel,widthUpDown+widthUpDown+widthLevel
        y2,y31,y32 = b2m[lvl2],b2m[lvl2]+b1m[0]+M12m[0,lvl2],b2m[lvl2]+b1m[1]+M12m[1,lvl2]
        plt.plot([x2,x3],[y2,y31],color=color,ls='dotted');
        plt.plot([x2,x3],[y2,y32],color=color,ls='--');
        
        # horizontal
        x4 = x3 + widthLevel
        plt.plot([x3,x4],[y31,y31],color=color,ls='dotted');
        plt.plot([x3,x4],[y32,y32],color=color,ls='--');
              
       
        # ===  software
        # up / down
        x5 = x4 + widthUpDown
        y41,y42 = b1m[0],b1m[1]
        plt.plot([x4,x5],[y31,y41],color=color,ls='dotted');
        plt.plot([x4,x5],[y32,y42],color=color,ls='--');

    # horizontal
    x6 = x5 + widthLevel
    x5l,x5r = x5+0.3*widthLevel, x6-0.3*widthLevel
    plt.plot([x5,x5l],[y41,y41],color='k',ls='dotted');
    plt.plot([x5r,x6],[y41,y41],color='k',ls='dotted');
    
    plt.plot([x5,x5l],[y42,y42],color='k',ls='--');
    plt.plot([x5r,x6],[y42,y42],color='k',ls='--');

    plt.text(0.5*(x5+x6), y41, dictSoftware[0],
             rotation=0,
             horizontalalignment='center',
             verticalalignment='center',
            size=SMALL_SIZE)
    plt.text(0.5*(x5+x6), y42, dictSoftware[1],
             rotation=0,
             horizontalalignment='center',
             verticalalignment='center',
            size=SMALL_SIZE)

    # up / down
    x7 = x6 + widthUpDown        
    plt.plot([x6,x7],[y41,0*y41],color='k',ls='dotted');
    plt.plot([x6,x7],[y42,0*y42],color='k',ls='--');
    
    # ==== Captions                
    # get minimum
    ymin = plt.ylim()[0]
    
    plt.text(0.5*(x1+x2), ymin, "Treatment",rotation=0,
             horizontalalignment='center',
             verticalalignment='bottom',
            size=MEDIUM_SIZE)
    plt.text(0.5*(x3+x4), ymin, "Interaction",rotation=0,
             horizontalalignment='center',
             verticalalignment='bottom',
            size=MEDIUM_SIZE)
    plt.text(0.5*(x5+x6), ymin, "Software",rotation=0,
             horizontalalignment='center',
             verticalalignment='bottom',
            size=MEDIUM_SIZE)
       
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
        
    plt.ylabel("Delta_{}".format(yname))
    plt.show()
    
def plotLevelsStd(widthInch,heigthInch,dpi,sizes,writeOut,path,dictMeanStd,dictTreatment,dictSoftware,trace,yname,x1,x2):
    
    
    SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE = sizes
    
    mu_Val,sig_Val = dictMeanStd[yname]
    
    b1l,b1m,b1u = getPercentiles(sig_Val*trace['{}_b1'.format(yname)])
    b2l,b2m,b2u = getPercentiles(sig_Val*trace['{}_b2'.format(yname)])
    inter0 = sig_Val*(trace['{}_M12'.format(yname)][:,0,:]+trace['{}_b2'.format(yname)])
    inter1 = sig_Val*(trace['{}_M12'.format(yname)][:,1,:]+trace['{}_b2'.format(yname)])
    inter0l,inter0m,inter0u = getPercentiles(inter0)
    inter1l,inter1m,inter1u = getPercentiles(inter1)
    
    a = np.array([inter0l,inter0m,inter0u,inter1l,inter1m,inter1u,b1l,b2l],dtype=object)
    
    ymin_global = np.min([np.min(x) for x in a])
        
    widthUpDown = 0.1
    widthLevel = 0.4
    
    # prepare color dict
    # use groups of 4 colors, as in tab20c
    colorIndex = dict({5:0,6:1,7:2,0:4,1:5,4:6,10:7,2:8,3:9,8:10,9:11})
    
    # prepare dataset dict    
    dictDataset = dict({5:0,6:0,7:0,0:1,1:1,4:1,10:1,2:2,3:2,8:2,9:2})
    
    # inverse dict
    inv_dictDataset = defaultdict(list)                                                                  
  
    # using loop to perform reverse mapping 
    for keys, vals in dictDataset.items():  
        for val in [vals]:  
            inv_dictDataset[val].append(keys) 
    
    
    numDatasets = len(np.unique(list(dictDataset.values())))
        
    fig,ax = plt.subplots(1, numDatasets, figsize=(widthInch,heigthInch), dpi=dpi,sharey=True)
    
    for indexDataset in np.arange(numDatasets):
        # set subplot
        curr_ax = ax[indexDataset]
        
        for treatmentNum,lvl2 in enumerate(inv_dictDataset[indexDataset]):
            # get number of treatments per dataset
            numTreatments = len(inv_dictDataset[indexDataset])
            
            # compute color
            color = mpl.cm.tab20c(colorIndex[lvl2])

            # ===  treatment
            # up / down
            x0,x1 = 0,widthUpDown
            #curr_ax.plot([x0,x1],[0,b2m[lvl2]],color=color);

            # horizontal
            x1,x2 = widthUpDown,widthUpDown+widthLevel            
            x_start = x1 + (x2-x1)*float(treatmentNum)/float(numTreatments)
            x_stop = x_start + (x2-x1)/float(numTreatments)
            
            curr_ax.plot([x_start,x_stop],[b2m[lvl2],b2m[lvl2]],color=color);     
                        
            # error bar
            curr_ax.fill([x_start,x_start,x_stop,x_stop],[b2l[lvl2],b2u[lvl2],b2u[lvl2],b2l[lvl2]],color=color,alpha=0.3);

            # label
            y1 = b2m[lvl2]
            curr_ax.text(0.5*(x_start+x_stop), b2l[lvl2], dictTreatment[lvl2],
                     rotation=90,
                     horizontalalignment='center',
                     verticalalignment='bottom',
                    size=SMALL_SIZE)

            # ===  treatment & software            
            x2,x3 = widthUpDown+widthLevel,widthUpDown+widthUpDown+widthLevel
                        
            # horizontal
            x4 = x3 + 1.5*widthLevel
            x_start = x3 + (x4-x3)*float(treatmentNum)/float(numTreatments)
            x_mid = x_start + 0.5*(x4-x3)/float(numTreatments)
            x_stop = x_start + (x4-x3)/float(numTreatments)
            
            curr_ax.plot([x_start,x_mid],[inter0m[lvl2],inter0m[lvl2]],color=color,ls='dotted');
            curr_ax.fill([x_start,x_start,x_mid,x_mid],[inter0l[lvl2],inter0u[lvl2],inter0u[lvl2],inter0l[lvl2]],color=color,alpha=0.3,ls='dotted');
            curr_ax.plot([x_mid,x_stop],[inter1m[lvl2],inter1m[lvl2]],color=color,ls='--');
            curr_ax.fill([x_mid,x_mid,x_stop,x_stop],[inter1l[lvl2],inter1u[lvl2],inter1u[lvl2],inter1l[lvl2]],color=color,alpha=0.3,ls='--');
            

            # ===  software
            # up / down
            x5 = x4 + widthUpDown
            y41,y42 = b1m[0],b1m[1]
            #curr_ax.plot([x4,x5],[y31,y41],color=color,ls='dotted');
            #curr_ax.plot([x4,x5],[y32,y42],color=color,ls='--');

        # horizontal
        x6 = x5 + 0.5*widthLevel        
        curr_ax.plot([x5,x5+0.5*(x6-x5)],[y41,y41],color='k',ls='dotted');
        curr_ax.fill([x5,x5,x5+0.5*(x6-x5),x5+0.5*(x6-x5)],[b1l[0],b1u[0],b1u[0],b1l[0]],color='gray',alpha=0.3,ls='dotted');

        curr_ax.plot([x5+0.5*(x6-x5),x6],[y42,y42],color='k',ls='--');
        curr_ax.fill([x5+0.5*(x6-x5),x5+0.5*(x6-x5),x6,x6],[b1l[1],b1u[1],b1u[1],b1l[1]],color='gray',alpha=0.3,ls='--');

        curr_ax.text(x5+0.25*(x6-x5), b1l[0], dictSoftware[0],
                 rotation=90,
                 horizontalalignment='center',
                 verticalalignment='bottom',
                size=SMALL_SIZE)
        curr_ax.text(x6-0.25*(x6-x5), b1l[1], dictSoftware[1],
                 rotation=90,
                 horizontalalignment='center',
                 verticalalignment='bottom',
                size=SMALL_SIZE)

        # up / down
        x7 = x6 + widthUpDown        
        #curr_ax.plot([x6,x7],[y41,0*y41],color='k',ls='dotted');
        #curr_ax.plot([x6,x7],[y42,0*y42],color='k',ls='--');

        # ==== Captions                
        # get minimum
        ymin = ymin_global-0.2*np.abs(ymin_global)#curr_ax.get_ylim()[0]


        curr_ax.text(0.5*(x1+x2), ymin, "Treatment",rotation=90,
                 horizontalalignment='center',
                 verticalalignment='top',
                size=BIGGER_SIZE)
        curr_ax.text(0.5*(x3+x4), ymin, "Interaction",rotation=90,
                 horizontalalignment='center',
                 verticalalignment='top',
                size=BIGGER_SIZE)
        curr_ax.text(0.5*(x5+x6), ymin, "Software",rotation=90,
               horizontalalignment='center',
                 verticalalignment='top',
                size=BIGGER_SIZE)

        curr_ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

        if indexDataset == 0:
            curr_ax.set_ylabel("Delta_{}".format(yname))
    plt.show()
    
def plotContrastLevel(widthInch,heigthIch,dpi,writeOut,path,dictMeanStd,x1contrast_dict,trace,yname):
    
    
    
    mu_Val,sig_Val = dictMeanStd[yname]
    b1_sample = sig_Val*trace['{}_b1'.format(yname)]
    
    for key, value in x1contrast_dict.items():
        contrast = np.dot(b1_sample, value)        
        pm.plot_posterior(contrast, ref_val=0.0, bins=50,hdi_prob=0.95,figsize=(widthInch,heigthIch));
        plt.title('Contrast {} on {}'.format(key,yname));
        
        if writeOut:
            plt.savefig(path + "contrast_{}_{}.pdf".format(key,yname),dpi=dpi)

            
def plotTreatmentPosterior(widthInch,heigthInch,dpi,sizes,writeOut,path,dictMeanStd,dictTreatment,dictSoftware,trace,yname,x1,x3):
        
    SMALL_SIZE,MEDIUM_SIZE,BIGGER_SIZE = sizes
    
    mu_Val,sig_Val = dictMeanStd[yname]
    
    # get posterior samples
    b1P = sig_Val*trace['{}_b1'.format(yname)]
    b2P = sig_Val*trace['{}_b2'.format(yname)]
    M12P = sig_Val*trace['{}_M12'.format(yname)]

    # prepare color dict for treatments
    # use groups of 4 colors, as in tab20c
    colorIndex = dict({5:0,6:1,7:2,0:4,1:5,4:6,10:7,2:8,3:9,8:10,9:11})
    
    # prepare dataset dict for treatments   
    dictDataset = dict({5:0,6:0,7:0,0:1,1:1,4:1,10:1,2:2,3:2,8:2,9:2})
    
    # === inverse dict ==== 
    inv_dictDataset = defaultdict(list)                                                                  
  
    # using loop to perform reverse mapping 
    for keys, vals in dictDataset.items():  
        for val in [vals]:  
            inv_dictDataset[val].append(keys) 
    # === 
    
    # get number of datasets    
    numDatasets = len(np.unique(list(dictDataset.values())))
    
    # get number of treatments per dataset
    dictDataset2NumberTreats = dict()
    for numDataset in range(numDatasets):        
        n = len(inv_dictDataset[numDataset])
        dictDataset2NumberTreats[numDataset] = n      
         
    # Get maximum of treatments per dataset
    tmax = np.max(list(dictDataset2NumberTreats.values()))
    
    
    # compute maximal number of pairs 
    maxpair = int(tmax*(tmax-1)/2)
    
    fig = plt.subplots(squeeze=False, figsize=(numDatasets*widthInch,maxpair*heigthInch), dpi=dpi);
    
    # store list for hdi
    hdiList = []

    for indexDataset in np.arange(numDatasets):
        # counter for row
        rowCounter = 0
        
        # first treatment 
        for treatmentNum_i,lvl2_i in enumerate(inv_dictDataset[indexDataset]):
            
            # second treatment 
            for treatmentNum_j,lvl2_j in enumerate(inv_dictDataset[indexDataset]):
                
                if treatmentNum_i > treatmentNum_j:
                                       
                    
                    # set subplot                    
                    curr_ax = plt.subplot2grid((maxpair, numDatasets), (rowCounter,indexDataset))
                    
                    # compute difference between treatments for each software
                    diffS0 = (M12P[:,0,lvl2_i]+b2P[:,lvl2_i]) -(M12P[:,0,lvl2_j]+b2P[:,lvl2_j])
                    diffS1 = (M12P[:,1,lvl2_i]+b2P[:,lvl2_i]) -(M12P[:,1,lvl2_j]+b2P[:,lvl2_j])                 
                    
                    #plot posterior                    
                    sns.kdeplot(diffS0,ax=curr_ax,label=dictSoftware[0],color='gray',alpha=0.3,ls='dotted');
                    sns.kdeplot(diffS1,ax=curr_ax,label=dictSoftware[1],color='gray',alpha=0.3,ls='--');
                    
                    # get hdi
                    hdiS0 = az.hdi(az.convert_to_inference_data(diffS0),hdi_prob=0.95)['x'].values
                    hdiS1 = az.hdi(az.convert_to_inference_data(diffS1),hdi_prob=0.95)['x'].values
                    
                    isSignificant = lambda x: (x[0] > 0.0) or (x[1] < 0.0)
                    
                    # store hdi
                    hdiList.append([dictTreatment[lvl2_i],dictTreatment[lvl2_j],
                                    hdiS0[0],hdiS0[1],isSignificant(hdiS0),
                                   hdiS1[0],hdiS1[1],isSignificant(hdiS1)])
                                        
                    # plot reference value zero
                    curr_ax.axvline(x=0,color="C1")
                    
                    # set title 
                    nameFirst = dictTreatment[lvl2_i]
                    nameSecond = dictTreatment[lvl2_j]
                    title = "{} vs. {}".format(nameFirst,nameSecond)
                    if isSignificant(hdiS0):
                        title += ": Significant on {}".format(dictSoftware[0])
                        if isSignificant(hdiS1):
                            title += " and {}".format(dictSoftware[1])                        
                    else:
                        if isSignificant(hdiS1):
                            title += ": Significant on {}".format(dictSoftware[1])
                            
                    curr_ax.set_title(title)
                    
                    # add legend
                    curr_ax.legend()   
                    
                    # set x label
                    curr_ax.set_xlabel('Delta {}'.format(yname))
                    
                    # remove y label decoration
                    curr_ax.tick_params(left=False)
                    curr_ax.set(yticklabels=[])
                    
                    
                    # increment counter
                    rowCounter += 1
                    
    #plt.suptitle('Estimated differences between treatments on {}'.format(yname))
    
    plt.tight_layout()                
    
    if writeOut:
        plt.savefig(path + "treatment_pairs_{}.pdf".format(yname),dpi=dpi)
    
    plt.show()
    
    # convert hdi to df
    df = pd.DataFrame(hdiList,columns=["Treatment_i","Treatment_j",                                    "hdi_{}_2.5%".format(dictSoftware[0]),"hdi_{}_97.5%".format(dictSoftware[0]),"isSignificant_on_{}".format(dictSoftware[0]),
                                  "hdi_{}_2.5%".format(dictSoftware[1]),"hdi_{}_97.5%".format(dictSoftware[1]),"isSignificant_on_{}".format(dictSoftware[1])])
    return df
# End of script    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    