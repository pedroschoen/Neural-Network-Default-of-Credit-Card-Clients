#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 19:49:48 2017

@author: pedroschoen
"""


    
def KS(x,verbose=True,plot=False):
    import seaborn
    import pandas
    import numpy
    import matplotlib.pyplot as plt

    x['mau']= 1 - x.CLASSE
    
    
    x['bucket'] = pandas.qcut(x.SCORE, 10)
    
    grouped = x.groupby('bucket', as_index = False)
    
    #numpy.savetxt("testeks.csv", x, fmt='%.2f',delimiter=";",)
    
    
    agg1 = grouped.min().SCORE
     
    agg1 = pandas.DataFrame(grouped.min().SCORE, columns = ['min_scr'])
     
    agg1['min_scr'] = grouped.min().SCORE
    
    
    agg1['max_scr'] = grouped.max().SCORE
     
    agg1['bads'] = grouped.sum().mau
    
    agg1['goods'] = grouped.sum().CLASSE
     
    agg1['total'] = agg1.bads + agg1.goods
    
     
    agg2 = (agg1.sort_values(by = 'min_scr')).reset_index(drop = True)
     
    agg2['odds'] = (agg2.goods / agg2.bads).apply('{0:.2f}'.format)
     
    agg2['bad_rate'] = (agg2.bads / agg2.total).apply('{0:.2%}'.format)
     
     
    agg2['ks'] = numpy.round(((agg2.bads / x.mau.sum()).cumsum() - (agg2.goods / x.CLASSE.sum()).cumsum()), 4) * 100
      
    flag = lambda x: '<----' if x == agg2.ks.max() else ''
     
    agg2['max_ks'] = agg2.ks.apply(flag)
         
    if verbose:
        print ()
        print (agg2)

        
    agg2['goodstotal']=agg2['goods'].sum()
    agg2['badtotal']=agg2['bads'].sum()
    agg2['perc_bom']=agg2['goods']/agg2['goodstotal']
    agg2['perc_mau']=agg2['bads']/agg2['badtotal']
    agg2['perc_bom_acum']=agg2['perc_bom'].cumsum()
    agg2['perc_mau_acum']=agg2['perc_mau'].cumsum()
    
    
    if plot:
        
        plt.figure(1)
        plt.subplot(311)
        plt.plot(agg2['perc_bom_acum'])
        plt.plot(agg2['perc_mau_acum'])
        plt.title('%Bom acum x %Mau acum')
    
        
        plt.subplot(312)
        plt.plot(agg2['perc_bom'])
        plt.plot(agg2['perc_mau'])
        plt.title('%Bom x %Mau')
        
        
        plt.subplot(313)
        plt.plot(agg2['ks'])
        plt.title('KS')
        
        plt.tight_layout()
    
        plt.show()    
        
    return agg2.ks.max()
    
    
