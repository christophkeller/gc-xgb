import argparse
import numpy as np
import glob
import sys
import random
import glob
import os
from scipy import stats 
import joblib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import dump_svmlight_file
from math import sqrt
from scipy.stats import gaussian_kde, pearsonr
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta
import xgboost as xgb
import pandas as pd
import requests

#parameter
SVMPATH = 'example/svm'
SVMURL = 'https://gmao.gsfc.nasa.gov/gmaoftp/geoscf/gc-xgb/svm'

def main(args):
    print('Looking at species: '+args.varname)
    # set flags based on outtype
    if args.outtype=="tend":
        logconc = False
        tend    = True
    elif args.outtype=="logconc":
        logconc = True
        tend    = False
    elif args.outtype=="conc":
        logconc = False 
        tend    = False
    elif args.outtype=="ratio":
        logconc = False 
        tend    = False
    oper = args.outtype
#---Read data from svm    
    # loading is fast
    if args.load_svm == 1: 
        train = get_svm('gcxgb_example_train.svm')
        valid = get_svm('gcxgb_example_valid.svm')
        Y_train = train.get_label()
        Y_valid = valid.get_label()
#---Read data from pkl    
    else: 
        # read full array of files
        filenames = glob.glob(args.path)
        megaf,megap,fvars,pvars = read_files(filenames,args.varname,tend,args.fraction,args.nbins)
        print('Shape of megaf: ',megaf.shape)
        print('Shape of megap: ',megap.shape)
        # normalize inputs & output
        idx = fvars.index('KPP_AIRDEN')
        if args.varname=='NOratio':
            megap[:,0] = megap[:,0]
        else:
            megap[:,0] = megap[:,0]/megaf[:,idx]*1.0e15
        idx2 = [i for i in fvars if 'KPP_BEFORE_CHEM' in i]
        for ivar in idx2:
            ii = fvars.index(ivar)
            megaf[:,ii] = megaf[:,ii]/megaf[:,idx]*1.0e15
        # drop input concentration from input features
        if args.drop==1:
            idx = fvars.index('KPP_BEFORE_CHEM_'+args.varname)
            megaf[:,idx] = 0.0
        # training / validation data
        X_train, X_valid, Y_train, Y_valid = train_test_split(megaf,megap)
        # arrays for xgb
        train = xgb.DMatrix(X_train,label=Y_train)
        valid = xgb.DMatrix(X_valid,label=Y_valid)
        # this takes forever!
        if args.save_svm == 1:
            print('saving svm files...')
            dump_svmlight_file(X_train,Y_train.ravel(),SVMPATH+'/gcxgb_example_train.svm',zero_based=True)
            dump_svmlight_file(X_valid,Y_valid.ravel(),SVMPATH+'/gcxgb_example_valid.svm',zero_based=True)
#---Train
# train xgboost using input params
    print('training xgb...')
    sys.stdout.flush()
    bst,dt = run_xgb(args,train,valid,Y_train,Y_valid)
    print('training done!')
    print('training time: {0:.3f}'.format(dt)+'\n')
    sys.stdout.flush()
    # write booster file to binary
    ofile = args.boosterfile.replace('%v',args.varname).replace('%t',args.outtype)
    bst.save_model(ofile)
    print('booster object written to '+ofile)
    return


def get_svm(ifile):
    '''Read svm files'''
    locfile = '/'.join([SVMPATH,ifile])
    urlfile = '/'.join([SVMURL,ifile])
    if not os.path.isfile(locfile):
        print('getting file from '+urlfile)
        r = requests.get(urlfile)
        open(locfile, 'wb').write(r.content)
    print('reading '+locfile) 
    dat = xgb.DMatrix(locfile)
    return dat 


def read_files(filenames,ispec,tend,fraction,nbins):
   '''
   Read pkl files with chemistry model output.
   '''
   first = True
   nfiles = len(filenames)
   for filename in filenames:
       print('Reading '+filename)
       sys.stdout.flush()
       [fvars, pvars, farr, parr] = joblib.load(filename)
       # reduce output to species of interest
       if ispec=='NOx' or ispec=='NOratio':
           # get NO and NO2 in kg N
           pidx1 = pvars.index('KPP_AFTER_CHEM_NO')
           pidx2 = pvars.index('KPP_AFTER_CHEM_NO2')
           ptmp1 = np.copy(parr[:,pidx1]) * (14./30.)
           ptmp2 = np.copy(parr[:,pidx2]) * (14./46.)
           # get NOx
           if ispec=='NOx':
               ptmp  = ptmp1 + ptmp2
           if ispec=='NOratio':
               ptmp  = ptmp1 / ( ptmp1 + ptmp2 )
       else:
           pidx = pvars.index('KPP_AFTER_CHEM_'+ispec)
           ptmp = np.copy(parr[:,pidx])
       if tend:
           if ispec=='NOx':
               fidx1 = fvars.index('KPP_BEFORE_CHEM_NO')
               fidx2 = fvars.index('KPP_BEFORE_CHEM_NO2')
               ftmp1 = np.copy(farr[:,fidx1]) * (14./30.)
               ftmp2 = np.copy(farr[:,fidx2]) * (14./46.)
               ftmp  = ftmp1 + ftmp2
           else:
               fidx = fvars.index('KPP_BEFORE_CHEM_'+ispec)
               ftmp = farr[:,fidx]
           ptmp = ( ptmp - ftmp ) / dt_chem
       if first:
           lens = farr.shape[0]
           nrows_per_file = int(lens*fraction)
           nrows_total    = nrows_per_file*nfiles
           nrows_per_bin  = int(nrows_per_file/nbins)
           megaf = np.zeros((nrows_total,farr.shape[1]),dtype='float32')
           megap = np.zeros((nrows_total,1),dtype='float32')
           first=False
           i1 = 0
       # get percentiles
       ranges = [np.percentile(ptmp,int(i)) for i in range(0,101,int(100/nbins))]
       for ibin in range(0,int(nbins)):
           # get all indeces that are within this percentile
           idxs = np.where( (ptmp >= ranges[ibin]) & (ptmp <= ranges[ibin+1]) )[0]
           # randomly pick values
           idx = np.random.choice(idxs,nrows_per_bin,replace=False)
           # pass to master array
           i0 = i1
           i1 = i0 + nrows_per_bin
           megaf[i0:i1,:] = farr[idx,:]
           megap[i0:i1,0] = ptmp[idx]
   # remove entries where NUMDEN is zero
   idx = fvars.index('KPP_AIRDEN')
   msk = np.where(megaf[:,idx]>0.0)[0]
   megaf = megaf[msk,:]
   megap = megap[msk,:]
   return megaf,megap,fvars,pvars


def run_xgb(args,train,valid,Y_train,Y_valid):
    # define XGBoost
    param = {
        'booster' : 'gbtree' ,
    }
    num_round = 20
    if args.gpu==1:
        param['tree_method'] = 'gpu_hist'
    else: 
        param['tree_method'] = 'hist'
        param['nthread'] = args.nthread  
    # train XGBoost 
    start_time = time.perf_counter()
    bst = xgb.train(param,train,num_round)
    end_time = time.perf_counter()
    dt = end_time - start_time
    # Prediction 
    P_train = bst.predict(train)
    P_valid = bst.predict(valid)
    # make plots
    figure = plt.figure(figsize=(12,6))
    ax = plt.subplot(1,2,1)
    title = args.title.replace('%v',args.varname).replace('%t',args.outtype)
    long_title = title+' - Training ({:.4f}'.format(dt)+'s)'
    make_fig(args,ax,Y_train.ravel(),P_train.ravel(),long_title)
    ax = plt.subplot(1,2,2)
    make_fig(args,ax,Y_valid.ravel(),P_valid.ravel(),title+' - Validation')
    # save figure
    ofile = args.scatterfile.replace('%v',args.varname).replace('%t',args.outtype)
    plt.tight_layout()
    plt.savefig(ofile)
    print('scatter plot written to '+ofile)
    plt.close()
    return bst,dt


def make_fig(args,ax,true,pred,title):
    # convert to ppbv
    #true = true * myrf.dt_chem * 1.0e9 * 28.96 / 48.0
    #pred = pred * myrf.dt_chem * 1.0e9 * 28.96 / 48.0
    # statistics
    R2    = r2_score(true,pred)
    nrmse = sqrt(mean_squared_error(true,pred))/np.std(true)
    nmb   = np.sum(pred-true)/np.sum(true)
    slope, intercept, r_value, p_value, std_err = stats.linregress(true,pred)
    # scatter plot
    ax.hexbin(true,pred,cmap=plt.cm.gist_earth_r,bins='log')
    # 1:1 line
    #minval = -3.0 
    #maxval =  3.0
    minval = np.min((np.min(true),np.min(pred)))
    maxval = np.max((np.max(true),np.max(pred)))
    ax.set_xlim(minval,maxval)
    ax.set_ylim(minval,maxval)
    ax.plot((0.95*minval,1.05*maxval),(0.95*minval,1.05*maxval),color='grey',linestyle='dashed')
    # regression line
    ax.plot((0.95*minval,1.05*maxval),(intercept+(0.95*minval*slope),intercept+(1.05*maxval*slope)),color='blue',linestyle='dashed')
    if args.outtype=='tend':
        xlabel='true tendency [scaled]'
        ylabel='predicted tendency [scaled]'
    if args.outtype=='conc':
        xlabel='true concentration [scaled]'
        ylabel='predicted concentration [scaled]'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    str_ncell = '{:,}'.format(pred.shape[0])
    istr = ' '.join(('N =',str_ncell))
    ax.text(0.05,0.95,istr,transform=ax.transAxes)
    str_R2    = '{0:.2f}'.format(R2)
    istr = ' '.join(('R$^{2}$=',str_R2))
    ax.text(0.05,0.90,istr,transform=ax.transAxes)
    str_rmse  = '{0:.2f}'.format(nrmse*100)
    istr = ' '.join(('NRMSE [%] =',str_rmse))
    ax.text(0.05,0.85,istr,transform=ax.transAxes)
    str_nmb  = '{0:.2f}'.format(nmb*100)
    istr = ' '.join(('NMB [%] =',str_nmb))
    ax.set_title(title)
    return


def parse_args():
    p = argparse.ArgumentParser(description='Undef certain variables')
    p.add_argument('-n','--nthread',type=int,help='number of threads',default=8)
    p.add_argument('-g','--gpu',type=int,help='use gpus',default=0)
    p.add_argument('-sf','--scatterfile',type=str,help='file name of scatter file',default='example/png/xgb_scatter_%v_%t.png')
    p.add_argument('-bf','--boosterfile',type=str,help='file name of booster object',default='example/bin/bst_%v_%t.bin')
    p.add_argument('-t','--title',type=str,help='figure title',default='XGBoost %v')
    p.add_argument('-v','--varname',type=str,help='variable name',default="O3")
    p.add_argument('-ot','--outtype',type=str,help='output type',default="tend")
    p.add_argument('-f','--fraction',type=float,help='data fraction',default=0.1)
    p.add_argument('-p','--path',type=str,help='data source path',default="example/pkl/*.pkl")
    p.add_argument('-d','--drop',type=int,help='drop input',default=0)
    p.add_argument('-b','--nbins',type=int,help='number of bins',default=10)
    p.add_argument('-ls','--load-svm',type=int,help='load svm file',default=1)
    p.add_argument('-ss','--save-svm',type=int,help='save svm file',default=0)
    return p.parse_args()

if __name__ == '__main__':
    main(parse_args())

# eof
