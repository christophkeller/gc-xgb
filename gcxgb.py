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
from math import sqrt
from scipy.stats import gaussian_kde, pearsonr
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta
import xgboost as xgb
import pandas as pd
import requests
import logging

#parameter
SVMPATH = 'example/svm'
SVMURL = 'https://gmao.gsfc.nasa.gov/gmaoftp/geoscf/gc-xgb/svm'

def main(args):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    log.addHandler(handler)
    log.info('Looking at species: O3')
#---Read data from svm    
    train = get_svm('gcxgb_example_train.svm')
    valid = get_svm('gcxgb_example_valid.svm')
    Y_train = train.get_label()
    Y_valid = valid.get_label()
#---Train
# train xgboost using input params
    log.info('training xgb...')
    sys.stdout.flush()
    bst,dt = run_xgb(args,train,valid,Y_train,Y_valid)
    log.info('training done!')
    label = 'training time (GPU)' if args.gpu==1 else 'training time (CPU)'
    log.info(label+': {0:.3f}'.format(dt))
    sys.stdout.flush()
    # write booster file to binary
    ofile = args.boosterfile.replace('%v','O3').replace('%t',args.outtype)
    bst.save_model(ofile)
    log.info('booster object written to {}'.format(ofile))
    return


def get_svm(ifile):
    '''Read svm files'''
    log = logging.getLogger(__name__)
    locfile = '/'.join([SVMPATH,ifile])
    urlfile = '/'.join([SVMURL,ifile])
    if not os.path.isfile(locfile):
        log.info('getting file from '+urlfile)
        r = requests.get(urlfile)
        open(locfile, 'wb').write(r.content)
    log.info('reading '+locfile) 
    dat = xgb.DMatrix(locfile)
    return dat 


def run_xgb(args,train,valid,Y_train,Y_valid):
    log = logging.getLogger(__name__)
    # define XGBoost
    param = {
        'booster' : 'gbtree' ,
    }
    num_round = 20
    if args.gpu==1:
        param['tree_method'] = 'gpu_hist'
    else: 
        param['tree_method'] = 'hist'
        if args.nthread > 0:
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
    title = args.title.replace('%v','O3').replace('%t','tend')
    long_title = title+' - Training ({:.4f}'.format(dt)+'s)'
    make_fig(args,ax,Y_train.ravel(),P_train.ravel(),long_title)
    ax = plt.subplot(1,2,2)
    make_fig(args,ax,Y_valid.ravel(),P_valid.ravel(),title+' - Validation')
    # save figure
    ofile = args.scatterfile.replace('%v','O3').replace('%t','tend')
    plt.tight_layout()
    plt.savefig(ofile)
    log.info('scatter plot written to '+ofile)
    plt.close()
    return bst,dt


def make_fig(args,ax,true,pred,title):
    log = logging.getLogger(__name__)
    # statistics
    R2    = r2_score(true,pred)
    nrmse = sqrt(mean_squared_error(true,pred))/np.std(true)
    nmb   = np.sum(pred-true)/np.sum(true)
    slope, intercept, r_value, p_value, std_err = stats.linregress(true,pred)
    # scatter plot
    ax.hexbin(true,pred,cmap=plt.cm.gist_earth_r,bins='log')
    # 1:1 line
    minval = np.min((np.min(true),np.min(pred)))
    maxval = np.max((np.max(true),np.max(pred)))
    ax.set_xlim(minval,maxval)
    ax.set_ylim(minval,maxval)
    ax.plot((0.95*minval,1.05*maxval),(0.95*minval,1.05*maxval),color='grey',linestyle='dashed')
    # regression line
    ax.plot((0.95*minval,1.05*maxval),(intercept+(0.95*minval*slope),intercept+(1.05*maxval*slope)),color='blue',linestyle='dashed')
    ax.set_xlabel('true tendency [scaled]')
    ax.set_ylabel('predicted tendency [scaled]')
    istr = ' '.join(('N =','{:,}'.format(pred.shape[0])))
    ax.text(0.05,0.95,istr,transform=ax.transAxes)
    istr = ' '.join(('R$^{2}$=','{0:.2f}'.format(R2)))
    ax.text(0.05,0.90,istr,transform=ax.transAxes)
    istr = ' '.join(('NRMSE [%] =','{0:.2f}'.format(nrmse*100)))
    ax.text(0.05,0.85,istr,transform=ax.transAxes)
    istr = ' '.join(('NMB [%] =','{0:.2f}'.format(nmb*100)))
    ax.set_title(title)
    return


def parse_args():
    p = argparse.ArgumentParser(description='Undef certain variables')
    p.add_argument('-n','--nthread',type=int,help='number of threads. Only used if on cpus and if value is > 0',default=-1)
    p.add_argument('-g','--gpu',type=int,help='use gpus',default=0)
    p.add_argument('-sf','--scatterfile',type=str,help='file name of scatter file',default='example/png/xgb_scatter_%v_%t.png')
    p.add_argument('-bf','--boosterfile',type=str,help='file name of booster object',default='example/bin/bst_%v_%t.bin')
    p.add_argument('-t','--title',type=str,help='figure title',default='XGBoost %v')
    p.add_argument('-ot','--outtype',type=str,help='output type',default="tend")
    p.add_argument('-b','--nbins',type=int,help='number of bins',default=10)
    return p.parse_args()

if __name__ == '__main__':
    main(parse_args())

# eof
