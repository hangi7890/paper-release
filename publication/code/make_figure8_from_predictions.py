"""Create Figure 8 from real case predictions. No mock data or fallback curves.
Usage: python make_figure8_from_predictions.py case_predictions.csv --out ../figures
Input schema is supplied in ../figure8_pending/case_predictions_TEMPLATE.csv.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('csv');ap.add_argument('--out',default='../figures');args=ap.parse_args()
    df=pd.read_csv(args.csv)
    required=['building_id','seed','adaptation_days','forecast_origin','target_timestamp','horizon','y_true','y_pred']
    if set(required)-set(df):raise ValueError('Missing columns: '+str(set(required)-set(df)))
    for c in ['forecast_origin','target_timestamp']:df[c]=pd.to_datetime(df[c],errors='raise')
    for c in ['seed','adaptation_days','horizon','y_true','y_pred']:df[c]=pd.to_numeric(df[c],errors='raise')
    def pair(building,seed,budget):
        parts=[]
        for day in [0,budget]:
            p=df[(df.building_id==building)&(df.seed==seed)&(df.adaptation_days==day)].copy().sort_values('target_timestamp')
            if len(p)!=672 or p.target_timestamp.nunique()!=672:raise ValueError(f'{building}, seed {seed}, budget {day}: require 672 unique hourly targets')
            if p.forecast_origin.nunique()!=28 or not (p.groupby('forecast_origin').size()==24).all():raise ValueError('Require 28 origins × 24 horizons')
            if not all(set(g.horizon)==set(range(1,25)) for _,g in p.groupby('forecast_origin')):raise ValueError('Horizon must be 1–24 for each origin')
            if not p.target_timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all():raise ValueError('Targets must be a contiguous 672-hour sequence')
            if not np.isfinite(p[['y_true','y_pred']].to_numpy()).all():raise ValueError('Nonfinite prediction or truth')
            parts.append(p.reset_index(drop=True))
        s,a=parts
        if not s[['forecast_origin','target_timestamp','horizon']].equals(a[['forecast_origin','target_timestamp','horizon']]) or not np.array_equal(s.y_true.to_numpy(),a.y_true.to_numpy()):raise ValueError('Source/adapted target keys or truth differ')
        mae=lambda p:float(np.abs(p.y_pred-p.y_true).mean())
        rmse=lambda p:float(np.sqrt(((p.y_pred-p.y_true)**2).mean()))
        return s,a,mae(a)-mae(s),rmse(a)-rmse(s)
    r=pair('Hog_office_Roger',20260926,1);g=pair('Hog_office_Gustavo',20260926,1)
    seeds=[20260924,20260925,20260926];bs=[pair('Eagle_education_Brooke',s,14) for s in seeds];deltas=np.array([b[2] for b in bs])
    if r[2]<=1e-12:raise ValueError('Roger case does not reproduce F1')
    if not(g[2]<-1e-12 and abs(g[3]-6.484847373417267)<1e-4):raise ValueError('Gustavo case does not reproduce frozen result')
    if not(deltas.min()<-1e-12 and deltas.max()>1e-12 and abs(np.ptp(deltas)-226.53299509003176)<.002):raise ValueError('Brooke case does not reproduce frozen seed range')
    blue='#0072B2';orange='#D55E00';gray='#6A6A6A';grid='#D9D9D9';ink='#202020'
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Nimbus Sans','Arial','Liberation Sans','DejaVu Sans'],'font.size':9.5,'axes.titlesize':11,'axes.labelsize':9.5,'axes.titleweight':'bold','axes.edgecolor':'#666666','axes.linewidth':.8,'xtick.major.width':.7,'ytick.major.width':.7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    fig,axes=plt.subplots(3,1,figsize=(10,9.2),layout='constrained',gridspec_kw={'height_ratios':[1.25,.85,1.25]})
    for ax,data,label,bld in [(axes[0],r,'(a) Repeated F1','Hog_office_Roger'),(axes[2],g,'(c) MAE improves while RMSE worsens','Hog_office_Gustavo')]:
        s,a,dm,dr=data;x=np.arange(672)/24
        ax.plot(x,s.y_true,color=ink,lw=1.05,label='Observed',zorder=3)
        ax.plot(x,s.y_pred,color=blue,lw=.85,label='Source-only',zorder=2)
        ax.plot(x,a.y_pred,color=orange,lw=.85,label='Adapted (1 day)',zorder=2)
        ax.set(title=f'{label}: {bld} | seed 20260926',xlabel='Day of fixed evaluation period',ylabel='Electricity (original meter units)',xlim=(0,28),xticks=[0,7,14,21,28])
        ax.grid(axis='y',color=grid,lw=.5);ax.set_axisbelow(True);ax.legend(ncol=3,loc='upper right')
        ax.text(.01,.95,f'ΔMAE = {dm:+.4f}; ΔRMSE = {dr:+.4f}',transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',pad=1.5,alpha=.88))
    axes[1].axhline(0,color=gray,ls='--',lw=.9)
    colors=[blue if d<0 else orange for d in deltas];markers=['D' if d<0 else 'o' for d in deltas]
    for i,(d,c,m) in enumerate(zip(deltas,colors,markers)):axes[1].scatter(i,d,c=c,s=48,marker=m,edgecolor='white',linewidth=.5,zorder=3)
    axes[1].set_xticks(range(3),[str(s) for s in seeds]);axes[1].set(title='(b) F4: Eagle_education_Brooke | 14-day adaptation',xlabel='Random seed',ylabel='ΔMAE versus source-only');axes[1].margins(x=.22);axes[1].grid(axis='y',color=grid,lw=.5);axes[1].set_axisbelow(True)
    for i,d in enumerate(deltas):axes[1].annotate(f'{d:+.4f}',(i,d),xytext=(7,4),textcoords='offset points',fontsize=8.5)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg','pdf']:fig.savefig(out/('Figure_8.'+ext),dpi=600,bbox_inches='tight',facecolor='white')
    pd.DataFrame([{'building_id':'Hog_office_Roger','seed':20260926,'delta_mae':r[2],'delta_rmse':r[3]},*({'building_id':'Eagle_education_Brooke','seed':s,'delta_mae':b[2],'delta_rmse':b[3]} for s,b in zip(seeds,bs)),{'building_id':'Hog_office_Gustavo','seed':20260926,'delta_mae':g[2],'delta_rmse':g[3]}]).to_csv(out/'Figure_8_verified_metrics.csv',index=False)
    print('Figure 8 created from validated real predictions. Inspect labels and curves before submission.')
if __name__=='__main__':main()
