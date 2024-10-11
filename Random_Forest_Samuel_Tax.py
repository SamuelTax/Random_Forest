# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:33:58 2024

@author: smtax
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# import dataset with datetime as index
df = pd.read_csv('C:/Users/smtax/MAIO/Urban_NL10418-AQ-METEO.csv',sep=';',parse_dates=[0], date_format='%d/%m/%Y %H:%M')
df.set_index('date', inplace=True)

# create list to select variables
select_variables = ['o3','wd','ws','ff','t','q','hourly_rain','p','n','rh']

# set negative o3 to 0
df_int = df[select_variables]
df_int.loc[df_int['o3']<0,'o3']=0

# convert winddirection and windspeed into u and v components
windspeed = df_int['ff'].values
winddirection = df_int['wd'].values
u = -np.sin(np.deg2rad(winddirection))*windspeed
v = -np.cos(np.deg2rad(winddirection))*windspeed
df_int.insert(2,'u',u)
df_int.insert(3,'v',v)
df_int = df_int.drop(['wd','ff'],axis=1)

# update list of selected variables
select_variables = ['u','v','ws','t','q','hourly_rain','p','n','rh']

# interpolate missing values and delete if more than 3 consecutive hours is missing
df_int = df_int.interpolate(limit=3).dropna()
print('deleted rows: ',len(df)-len(df_int))

# define training and prediction dataframes
df_train = df_int.loc['2015':'2017']
df_pred = df_int.loc['2018']

# define statistics, random forest, and plot functions
def r2(obs,pred):
    res_ss = np.sum((obs-pred)**2)
    tot_ss = np.sum((obs-np.mean(obs))**2)
    return 1-res_ss/tot_ss

def rmse(obs,pred):
    return np.sqrt(np.mean((pred-obs)**2))

def statistics(ts_t, ts_p):
    y_t = ts_t['o3_m'].values
    y_pt = ts_t['o3_p'].values
    y_p = ts_p['o3_m'].values
    y_pp = ts_p['o3_p'].values
    r2_t = r2(y_t,y_pt)
    r2_p = r2(y_p,y_pp)
    cor_t = np.corrcoef(y_t,y_pt)[0,1]
    cor_p = np.corrcoef(y_p,y_pp)[0,1]
    rmse_t = rmse(y_t,y_pt)
    rmse_p = rmse(y_p,y_pp)
    stats = [r2_t,r2_p,cor_t,cor_p,rmse_t,rmse_p]
    return stats

def random_forest(x_t,x_p,y_t,n,m,r):
    # define and perform model
    model = RandomForestRegressor(n_estimators=n, min_samples_leaf=m, random_state=r)
    model.fit(x_t, y_t)
    y_pt = model.predict(x_t)
    y_pp = model.predict(x_p)
    # create dataframes for output timeseries
    ts_t = pd.DataFrame()
    ts_t['o3_m'] = df_train['o3']
    ts_t['o3_p'] = y_pt
    ts_p = pd.DataFrame()
    ts_p['o3_m'] = df_pred['o3']
    ts_p['o3_p'] = y_pp
    stats = statistics(ts_t,ts_p)
    return stats, ts_t, ts_p

def plots(ts_t, ts_p, stats, name='none', save=False):
    # plot daily mean ozone concentration 2015-2017
    plt.plot(ts_t['o3_m'].resample('D').mean(),label='measured')
    plt.plot(ts_t['o3_p'].resample('D').mean(),label='predicted')
    plt.xlabel('date',fontsize=15)
    plt.ylabel('concentration [$\mu$g m$^{-3}$]',fontsize=15)
    plt.legend(loc='upper right',fontsize=15)
    plt.grid()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    if save==True:
        plt.savefig('C:/Users/smtax/MAIO/RF_figs/'+name+'_ts_t.png',dpi=300)
    plt.show()
    
    # create trendline for correlation plot
    z = np.polyfit(ts_t['o3_m'],ts_t['o3_p'],1)
    p = np.poly1d(z)
    vmax = max(max(ts_t['o3_m'].values),max(ts_t['o3_p'].values))
    
    # create correlation plot 2015-2017
    plt.scatter(ts_t['o3_m'],ts_t['o3_p'],s=0.1)
    plt.plot([0,vmax],[0,vmax],c= 'black',ls='dashed')
    plt.plot(ts_t['o3_m'],p(ts_t['o3_m']),c='tab:red')
    plt.xlabel(r'measured concentration [$\mu$g m$^{-3}$]',fontsize=12)
    plt.ylabel(r'predicted concentration [$\mu$g m$^{-3}$]',fontsize=12)
    plt.axis('square')
    plt.text(0.05, 0.95, rf'$R^2$ = {stats[0]:.2f}'+'\n'+rf'$\rho$ = {stats[2]:.2f}'+'\n'+rf'RMSE = {stats[4]:.1f}',transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    if save==True:
        plt.savefig('C:/Users/smtax/MAIO/RF_figs/'+name+'_cor_t.png',dpi=300)
    plt.show()

    # plot daily mean ozone concentration 2018
    plt.plot(ts_p['o3_m'].resample('D').mean(),label='measured')
    plt.plot(ts_p['o3_p'].resample('D').mean(),label='predicted')
    plt.xlabel('date',fontsize=15)
    plt.ylabel('concentration [$\mu$g m$^{-3}$]',fontsize=15)
    plt.legend(loc='upper right',fontsize=15)
    plt.grid()
    if save==True:
        plt.savefig('C:/Users/smtax/MAIO/RF_figs/'+name+'_ts_p.png',dpi=300)
    plt.show()
    
    # create trendline for correlation plot
    z = np.polyfit(ts_p['o3_m'],ts_p['o3_p'],1)
    p = np.poly1d(z)
    vmax = max(max(ts_p['o3_m'].values),max(ts_p['o3_p'].values))

    # create correlation plot 2018
    plt.scatter(ts_p['o3_m'],ts_p['o3_p'],s=0.1)
    plt.plot([0,vmax],[0,vmax],c= 'black',ls='dashed')
    plt.plot(ts_p['o3_m'],p(ts_p['o3_m']),c='tab:red')
    plt.xlabel(r'measured concentration [$\mu$g m$^{-3}$]',fontsize=12)
    plt.ylabel(r'predicted concentration [$\mu$g m$^{-3}$]',fontsize=12)
    plt.axis('square')
    plt.text(0.05, 0.95, rf'$R^2$ = {stats[1]:.2f}'+'\n'+rf'$\rho$ = {stats[3]:.2f}'+'\n'+rf'RMSE = {stats[5]:.1f}',transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    if save==True:
        plt.savefig('C:/Users/smtax/MAIO/RF_figs/'+name+'_cor_p.png',dpi=300)
    plt.show()
    
# create x and y data for training period (2015-2017) and prediction period (2018)
x_t = df_train[select_variables].values
x_p = df_pred[select_variables].values
y_t = df_train['o3'].values
y_p = df_pred['o3'].values

# finding the optimal number of trees
trees = np.arange(5,255,5,dtype=int)
tree_R2_t = []
tree_R2_p = []
tree_r_t = []
tree_r_p = []
tree_rmse_t = []
tree_rmse_p = []

for i in trees:
    stats,ts_t,ts_p= random_forest(x_t,x_p,y_t,n=i,m=5,r=42)
    tree_R2_t.append(stats[0])
    tree_R2_p.append(stats[1])
    tree_r_t.append(stats[2])
    tree_r_p.append(stats[3])
    tree_rmse_t.append(stats[4])
    tree_rmse_p.append(stats[5])

# plot statistics for tree sensitivity
plt.plot(trees,tree_R2_t,label='training')
plt.plot(trees,tree_R2_p,label='testing')
plt.xlabel('number of trees',fontsize=15)
plt.ylabel('$R^2$',fontsize=15)
plt.grid()
plt.legend(fontsize=15)
#plt.savefig('C:/Users/smtax/MAIO/RF_figs/tree_R2.png',dpi=300)
plt.show()

plt.plot(trees,tree_r_t,label='training')
plt.plot(trees,tree_r_p,label='testing')
plt.xlabel('number of trees',fontsize=15)
plt.ylabel(r'$\rho$',fontsize=15)
plt.grid()
plt.legend(fontsize=15)
#plt.savefig('C:/Users/smtax/MAIO/RF_figs/tree_r.png',dpi=300)
plt.show()

plt.plot(trees,tree_rmse_t,label='training')
plt.plot(trees,tree_rmse_p,label='testing')
plt.xlabel('number of trees',fontsize=15)
plt.ylabel('RMSE [$\mu$g m$^{-3}$]',fontsize=15)
plt.grid()
plt.legend(fontsize=15)
#plt.savefig('C:/Users/smtax/MAIO/RF_figs/tree_RMSE.png',dpi=300)
plt.show()

# finding optimal minimum leaf size
leafs = np.arange(1,51,1)
leaf_R2_t = []
leaf_R2_p = []
leaf_r_t = []
leaf_r_p = []
leaf_rmse_t = []
leaf_rmse_p = []

for i in leafs:
    stats,ts_t,ts_p= random_forest(x_t,x_p,y_t,n=100,m=i,r=42)
    leaf_R2_t.append(stats[0])
    leaf_R2_p.append(stats[1])
    leaf_r_t.append(stats[2])
    leaf_r_p.append(stats[3])
    leaf_rmse_t.append(stats[4])
    leaf_rmse_p.append(stats[5])   

# plot statistics for leaf sensitivity
plt.plot(leafs,leaf_R2_t,label='training')
plt.plot(leafs,leaf_R2_p,label='testing')
plt.xlabel('minimum leaf size',fontsize=15)
plt.ylabel('$R^2$',fontsize=15)
plt.grid()
plt.legend(fontsize=15)
#plt.savefig('C:/Users/smtax/MAIO/RF_figs/leaf_R2.png',dpi=300)
plt.show()

plt.plot(leafs,leaf_r_t,label='training')
plt.plot(leafs,leaf_r_p,label='testing')
plt.xlabel('minimum leaf size',fontsize=15)
plt.ylabel(r'$\rho$',fontsize=15)
plt.grid()
plt.legend(fontsize=15)
#plt.savefig('C:/Users/smtax/MAIO/RF_figs/leaf_r.png',dpi=300)
plt.show()

plt.plot(leafs,leaf_rmse_t,label='training')
plt.plot(leafs,leaf_rmse_p,label='testing')
plt.xlabel('minimum leaf size',fontsize=15)
plt.ylabel('RMSE [$\mu$g m$^{-3}$]',fontsize=15)
plt.grid()
plt.legend(fontsize=15)
#plt.savefig('C:/Users/smtax/MAIO/RF_figs/leaf_RMSE.png',dpi=300)
plt.show()

# run random forest with optimal hyperparameters with all meteorological variables and calculate statistics
stats,ts_t,ts_p= random_forest(x_t,x_p,y_t,n=100,m=10,r=42)
plots(ts_t, ts_p, stats)

# define dataframes and lists to store results for importance analysis
training = pd.DataFrame()
training['o3_m'] = df_train['o3']
prediction = pd.DataFrame()
prediction['o3_m'] = df_pred['o3']
names = []
R2_t = []
R2_p = []
r_t = []
r_p = []
rmse_t = []
rmse_p = []

# calculate importance of each meteorological variable
for var in select_variables:
    x_t = df_train[var].values.reshape(-1,1)
    x_p = df_pred[var].values.reshape(-1,1)
    stats,ts_t,ts_p = random_forest(x_t,x_p,y_t,n=100,m=10,r=42)
    training['o3_p_',var] = ts_t['o3_p']
    prediction['o3_p_',var] = ts_p['o3_p']
    names.append(var)
    R2_t.append(stats[0])
    R2_p.append(stats[1])
    r_t.append(stats[2])
    r_p.append(stats[3])
    rmse_t.append(stats[4])
    rmse_p.append(stats[5])

# rename hourly_rain to h_rain
names[5]='h_rain'

# sort names according to R2 of training data    
pairs = sorted(zip(R2_t,R2_p,r_t,r_p,rmse_t,rmse_p,names), reverse=True)
R2_t_s, R2_p_s, r_t_s, r_p_s, rmse_t_s, rmse_p_s, names_s = zip(*pairs)
x_axis = np.arange(len(names_s))

# plot R2 values
plt.bar(x_axis-0.2,R2_t_s,0.4,label='training')
plt.bar(x_axis+0.2,R2_p_s,0.4,label='testing')
plt.xticks(x_axis,names_s)
plt.xlabel('meteorological variable',fontsize=15)
plt.ylabel('$R^2$',fontsize=15)
plt.legend(loc='upper right',fontsize=15)
#plt.savefig('C:/Users/smtax/MAIO/RF_figs/imp_R2.png',dpi=300)
plt.show()

# plot r values
plt.bar(x_axis-0.2,r_t_s,0.4,label='training')
plt.bar(x_axis+0.2,r_p_s,0.4,label='testing')
plt.xticks(x_axis,names_s)
plt.xlabel('meteorological variable',fontsize=15)
plt.ylabel(r'$\rho$',fontsize=15)
plt.legend(loc='upper right',fontsize=15)
#plt.savefig('C:/Users/smtax/MAIO/RF_figs/imp_r.png',dpi=300)
plt.show() 

# plot rmse values
plt.bar(x_axis-0.2,rmse_t_s,0.4,label='training')
plt.bar(x_axis+0.2,rmse_p_s,0.4,label='testing')
plt.xticks(x_axis,names_s)
plt.xlabel('meteorological variable',fontsize=15)
plt.ylabel('RMSE [$\mu$g m$^{-3}$]',fontsize=15)
plt.legend(loc='lower right',fontsize=15,framealpha=1)
#plt.savefig('C:/Users/smtax/MAIO/RF_figs/imp_RMSE.png',dpi=300)
plt.show() 

# print correlation coefficients between variables and ozone concentrations
cormat = np.corrcoef(df_int.values,rowvar=False)
print('correlation between ozone and meteorological variables:')
for i in range(len(select_variables)):
    print(f'{select_variables[i]}:{cormat[0,i+1]:.2f}')
    


