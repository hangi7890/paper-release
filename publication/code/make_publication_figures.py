from pathlib import Path
import csv, json, math, html, textwrap, io, xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT=(Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name=='code' else Path(__file__).resolve().parent/'Figure_Table_Package_v1')
for d in ['figures','drawio','data','tables','code','figure8_pending']: (ROOT/d).mkdir(parents=True,exist_ok=True)
# Restrained, colour-vision-deficiency-safe publication palette (Okabe–Ito based).
BLUE='#0072B2'; ORANGE='#D55E00'; GREEN='#009E73'; PURPLE='#CC79A7'
GRAY='#6A6A6A'; LIGHT='#FFFFFF'; INK='#202020'; GRID='#D9D9D9'; PALE='#F5F5F5'
plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Nimbus Sans','Arial','Liberation Sans','DejaVu Sans'],
    'font.size':9.5,'axes.titlesize':11,'axes.labelsize':9.5,'axes.titleweight':'bold',
    'axes.edgecolor':'#666666','axes.linewidth':0.8,'axes.spines.top':False,
    'axes.spines.right':False,'xtick.color':'#333333','ytick.color':'#333333',
    'xtick.major.width':0.7,'ytick.major.width':0.7,'legend.frameon':False,
    'lines.linewidth':1.35,'lines.markersize':5,'svg.fonttype':'none','pdf.fonttype':42,
    'savefig.facecolor':'white','figure.facecolor':'white'
})
DAYS=[0,1,3,7,14,28]; WINDOWS=[0,1,49,145,313,649]
MAE=[40.6500,38.4238,34.9867,29.6476,27.4135,24.1671]
RMSE=[54.6597,52.1976,48.2985,42.1623,39.3044,35.2364]
DMAE=[0,-2.2262,-5.6633,-11.0024,-13.2365,-16.4829]
DRMSE=[0,-2.4621,-6.3612,-12.4975,-15.3554,-19.4233]
A=[None,39.68,69.23,89.83,91.92,94.84]; B=[6.82,2.67,7.56,15.87,19.93,26.85]
SITES=['Bobcat','Bull','Eagle','Gator','Hog','Moose','Swan']; NS=[22,111,94,63,135,10,18]
H1=[-2.11086,-12.0117,-12.9714,-9.58665,-19.751,-56.3351,-2.61338]
F1_1=[60.61,52.25,60.64,45.50,69.14,43.33,90.74]; F1_28=[12.12,.60,12.77,4.23,6.42,0,0]
ASITE=[87.88,99.40,87.23,95.77,93.58,100,100]; BSITE=[36.36,33.93,24.47,6.88,26.67,13.33,46.30]
COUNTS={'F1':[813,377,134,104,80],'F2':[792,350,101,87,69],'F3-PD':[1211,1097,921,852,748],'F3-W':[1119,1000,892,803,691],'F3-any':[1316,1256,1161,1092,1003],'F3-both':[1014,841,652,563,436],'F4':[164,149,61,44,30]}
ASSOC=[(1,'Daily repeatability',389,-.4063),(1,'24-h context scale gap',453,-.3221),(1,'Available-data scale gap',453,-.3194),(3,'Available-data scale gap',453,-.4128),(3,'24-h context scale gap',453,-.3891),(3,'Available-data CV',453,.2979),(7,'Available-data CV',453,.3185),(7,'Available-data scale gap',453,-.2990),(7,'24-h context scale gap',453,-.2837),(14,'24-h context mean',453,-.3225),(14,'Source-mean percentile',453,-.3186),(14,'Available-data mean',453,-.3153),(28,'Source-mean percentile',453,-.3513),(28,'Available-data mean',453,-.3510),(28,'24-h context mean',453,-.3509)]
LATENCY=[('Jetson Orin Nano','Model-only',.690587,.675552,.731138,.960481),('RTX 3080','Model-only',.205226,.171072,.354354,.885665),('Jetson Orin Nano','End-to-end',.932956,.883082,.937253,.976195),('RTX 3080','End-to-end',.337380,.232600,.912830,2.373438)]
def csvout(name,head,rows):
    with (ROOT/'data'/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(head);w.writerows(rows)
if __name__ == "__main__":
    csvout('performance.csv',['days','windows','mae','rmse','delta_mae','delta_rmse','A_pct','B_pct'],zip(DAYS,WINDOWS,MAE,RMSE,DMAE,DRMSE,A,B))
if __name__ == "__main__":
    csvout('site_results.csv',['site','buildings','delta_mae_28d','f1_1d_pct','f1_28d_pct','A_28d_pct','B_28d_pct'],zip(SITES,NS,H1,F1_1,F1_28,ASITE,BSITE))
if __name__ == "__main__":
    csvout('failure_counts.csv',['outcome','denominator']+[str(d) for d in DAYS[1:]],[[k,453 if k=='F4' else 1359]+v for k,v in COUNTS.items()])
if __name__ == "__main__":
    csvout('exploratory_associations.csv',['days','characteristic','n','spearman_rho'],ASSOC)
if __name__ == "__main__":
    csvout('latency_summary.csv',['device','mode','mean_ms','median_ms','p95_ms','p99_ms'],LATENCY)
def save(fig,name,pending=False):
    folder=ROOT/('figure8_pending' if pending else 'figures')
    for ext in ['png','svg','pdf']:
        buffer=io.BytesIO();fig.savefig(buffer,format=ext,dpi=600,bbox_inches='tight',facecolor='white');(folder/(name+'.'+ext)).write_bytes(buffer.getvalue())
    plt.close(fig)
def tidy(ax):
    ax.grid(axis='y',color=GRID,linewidth=.55)
    ax.set_axisbelow(True)
def budget(ax,nonzero=False):
    ds=DAYS[1:] if nonzero else DAYS;ax.set_xticks(range(len(ds)),[str(d) for d in ds]);ax.set_xlabel('Target adaptation budget (days; discrete conditions)')

class Diagram:
    def __init__(self,name,height=8):
        self.name=name;self.W=1000;self.H=height*100
        self.fig,self.ax=plt.subplots(figsize=(10,height));self.ax.set(xlim=(0,self.W),ylim=(self.H,0));self.ax.axis('off');self.fig.subplots_adjust(0,0,1,1)
        self.mx=ET.Element('mxfile',host='app.diagrams.net');dia=ET.SubElement(self.mx,'diagram',id=name,name=name);model=ET.SubElement(dia,'mxGraphModel',page='1',pageWidth='1000',pageHeight=str(self.H));self.root=ET.SubElement(model,'root');ET.SubElement(self.root,'mxCell',id='0');ET.SubElement(self.root,'mxCell',id='1',parent='0');self.i=2
    def cell(self,value,x,y,w,h,style):
        c=ET.SubElement(self.root,'mxCell',id=str(self.i),value=value,style=style,vertex='1',parent='1');self.i+=1;ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
    def text(self,x,y,s,size=11,bold=False,align='left',color=INK,w=940,h=32):
        self.ax.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',ha=align,va='top',color=color,linespacing=1.4)
        xx=x if align=='left' else x-w/2
        self.cell(html.escape(s).replace('\n','<br>'),xx,y,w,h,f'text;html=1;align={align};verticalAlign=top;fontSize={size*1.3889};fontFamily=Arial;fontColor={color};fontStyle={1 if bold else 0};whiteSpace=wrap;')
    def box(self,x,y,w,h,s,fill=LIGHT,edge=BLUE,size=11):
        # Straight corners, white field, and a thin rule reproduce journal schematics
        # more reliably than dashboard-like rounded cards.
        face = '#FFFFFF' if fill in (LIGHT, '#F6F7F8') else fill
        self.ax.add_patch(Rectangle((x,y),w,h,linewidth=.9,edgecolor=edge,facecolor=face))
        self.ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=size,color=INK,linespacing=1.4)
        self.cell(html.escape(s).replace('\n','<br>'),x,y,w,h,f'rounded=0;shadow=0;glass=0;whiteSpace=wrap;html=1;fillColor={face};strokeColor={edge};strokeWidth=1;fontColor={INK};fontFamily=Arial;fontSize={size*1.3889};spacing=8;')
    def line(self,x1,y1,x2,y2,color=GRAY,arrow=False,dash=False):
        self.ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->' if arrow else '-',color=color,lw=.9,linestyle='--' if dash else '-'))
        c=ET.SubElement(self.root,'mxCell',id=str(self.i),style=f'endArrow={"classic" if arrow else "none"};endFill=1;html=1;strokeWidth=1;strokeColor={color};dashed={int(dash)};',edge='1',parent='1');self.i+=1;g=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'});ET.SubElement(g,'mxPoint',x=str(x1),y=str(y1),**{'as':'sourcePoint'});ET.SubElement(g,'mxPoint',x=str(x2),y=str(y2),**{'as':'targetPoint'})
    def finish(self,pending=False):
        ET.indent(self.mx);ET.ElementTree(self.mx).write(ROOT/('figure8_pending' if pending else 'drawio')/(self.name+'.drawio'),encoding='utf-8',xml_declaration=True);save(self.fig,self.name,pending)

def figure1():
    d=Diagram('Figure_1',9)
    d.text(30,20,'(a) Site-separated cohorts',14,True)
    d.box(290,65,420,60,'BDG2 hourly electricity records\nQuality screening and cohort assignment')
    for x,w,s in [(30,280,'Source\n20 buildings / 4 sites\nPretraining and source validation'),(360,280,'Development\n12 buildings / 6 sites\nProtocol development'),(690,280,'Final-evaluation proposal\n510 buildings / 9 sites\nPrespecified candidate pool')]:
        d.line(500,125,x+w/2,155,arrow=True);d.box(x,155,w,100,s)
    d.box(690,285,280,65,'Quality gate: exclude 57\nContinuous zero runs ≥30 days',fill='#FFF5EA',edge=ORANGE)
    d.line(830,255,830,285,arrow=True);d.line(830,350,830,375,arrow=True)
    d.box(690,375,280,60,'Final evaluation\n453 buildings / 7 sites',fill='#EAF4F0',edge=GREEN)
    d.text(35,295,'Non-overlapping source, development, and final site sets.\nMouse and Shrew had no eligible final buildings.\nSite identities and cohort roles are listed in Table 1.',11,w=600,h=100)
    d.text(30,470,'(b) Available observations and fixed evaluation',14,True)
    d.box(30,520,180,70,'Initial context\n24 hours')
    d.box(235,520,390,70,'Adaptation observations\n1 / 3 / 7 / 14 / 28 days')
    d.box(655,520,315,70,'Common evaluation period\n28 days / 672 hourly targets',fill='#EAF4F0',edge=GREEN)
    d.line(210,555,235,555,arrow=True);d.line(625,555,655,555,arrow=True)
    d.text(40,609,'Budgets (days)       0       1       3       7       14       28\nTraining windows    0       1      49     145     313      649',11,w=900,h=65)
    d.text(40,669,'Schematic only; budgets are alternatives. Adaptation and evaluation targets do not overlap.',10,color=GRAY)
    d.text(30,724,'(c) Matched model comparison',14,True)
    d.box(30,777,260,65,'Source-pretrained model\nThree fixed random seeds')
    d.box(355,762,280,90,'0 days: source-only\n1–28 days: head-only adaptation\nLSTM representation frozen')
    d.box(705,777,265,65,'Same evaluation targets\nMAE, RMSE, A/B, F1–F4')
    d.line(290,809,355,809,arrow=True);d.line(635,809,705,809,arrow=True);d.finish()

def figure2():
    d=Diagram('Figure_2',8.4)
    d.text(30,20,'(a) Input representation',14,True)
    d.box(30,72,330,100,'Historical electricity\nPrevious 24 hourly observations\nFrozen source normalization')
    d.box(410,72,560,100,'Deterministic target-horizon calendar\nHour sine/cosine + weekday sine/cosine\nWeekend indicator')
    d.line(195,172,500,212,arrow=True);d.line(690,172,500,212,arrow=True)
    d.box(345,212,310,65,'Input sequence: 24 × 6\n1 electricity + 5 calendar variables')
    d.text(30,316,'(b) Shared forecasting architecture',14,True)
    d.box(30,364,260,90,'Single-layer LSTM\nHidden size 32\nFrozen during target adaptation')
    d.box(370,364,260,90,'Linear forecasting head\n24 outputs\nUpdated during adaptation',fill='#FFF5EA',edge=ORANGE)
    d.box(710,364,260,90,'Direct 24-hour forecast\nInverse normalization\nOriginal meter units')
    d.line(500,277,15,294);d.line(15,294,15,409);d.line(15,409,30,409,arrow=True);d.line(290,409,370,409,arrow=True);d.line(630,409,710,409,arrow=True)
    d.text(50,478,'5,912 parameters at inference • MSE loss • Source pretraining uses the full network',11,w=900)
    d.text(30,540,'(c) Canonical checkpoint for edge validation',14,True)
    d.box(30,590,290,105,'Source-only checkpoint\nSeed 20260924\nArchitecture and weights fixed')
    d.box(395,590,250,105,'RTX 3080\nReference inference\nCUDA FP32 / batch 1')
    d.box(720,590,250,105,'Jetson Orin Nano\nTransferred checkpoint\nCUDA FP32 / batch 1')
    d.line(320,635,395,635,arrow=True);d.line(645,635,720,635,arrow=True)
    d.text(40,734,'Validation: numerical consistency and inference cost for the shared architecture.\nThis experiment does not establish the accuracy of every adapted checkpoint.',11,w=910,h=70);d.finish()

def figure3():
    fig=plt.figure(figsize=(10,8.1),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1.05]);axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
    for ax,y,base,title in zip(axes[:2],[MAE,RMSE],[[20.8740,18.1359],[39.5247,30.0582]],['(a) Mean absolute error','(b) Root mean squared error']):
        ax.plot(range(6),y,'o-',color=BLUE,lw=2,label='Transferred LSTM');ax.axhline(base[0],color=ORANGE,ls='--',label='Previous-day');ax.axhline(base[1],color=GREEN,ls=':',lw=2,label='Weekly-same-hour');budget(ax);tidy(ax);ax.set(title=title,ylabel='Site-equal error (original meter units)');ax.legend(frameon=False,fontsize=9);ax.annotate(f'{y[-1]:.4f}',(5,y[-1]),xytext=(-6,-22),textcoords='offset points',ha='right',color=BLUE)
    ax=axes[2];ax.plot(range(1,6),A[1:],'o-',color=BLUE,lw=2,label='A: MAE improves over source-only');ax.plot(range(1,6),B[1:],'s-',color=ORANGE,lw=2,label='B: MAE beats both temporal baselines');ax.set_xticks(range(1,6),[str(d) for d in DAYS[1:]]);ax.set(xlabel='Target adaptation budget (days; discrete conditions)',ylabel='Site-equal rate (%)',ylim=(0,105),title='(c) Improvement and baseline competitiveness');tidy(ax);ax.legend(loc='upper left',frameon=False);ax.annotate('94.84%',(5,A[-1]),xytext=(-12,-18),textcoords='offset points',ha='right',color=BLUE);ax.annotate('26.85%',(5,B[-1]),xytext=(-12,8),textcoords='offset points',ha='right',color=ORANGE);save(fig,'Figure_3')

def figure4():
    fig=plt.figure(figsize=(10,8.8),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1.15,1]);ax=fig.add_subplot(gs[0,:]);ys=np.arange(8)
    ax.axvline(0,color=GRAY,lw=1);ax.scatter(H1,ys[:7],color=BLUE,s=36);ax.errorbar(-16.482888133,7,xerr=[[14.424681746],[9.91991016445]],fmt='D',color=INK,capsize=5,ms=7)
    ax.set_yticks(ys,[f'{s} (n = {n})' for s,n in zip(SITES,NS)]+['Site-equal estimate']);ax.invert_yaxis();ax.set(xlim=(-63,8),xlabel='ΔMAE at 28 days (adapted − source-only; original meter units)',title='(a) H1 primary hypothesis: site effects and overall 95% CI')
    for y,v in enumerate(H1):ax.text(v-.8,y,f'{v:.2f}',va='center',ha='right',fontsize=9)
    ax.text(-59,5.9,'Site-equal ΔMAE: −16.4829\n95% CI: [−30.9076, −6.5630]',fontsize=9)
    for ax,l,r,c1,c2,title,lab1,lab2 in [(fig.add_subplot(gs[1,0]),F1_1,F1_28,ORANGE,BLUE,'(b) H2 secondary: F1 frequency','1 day','28 days'),(fig.add_subplot(gs[1,1]),ASITE,BSITE,BLUE,ORANGE,'(c) H3 secondary: A versus B','A: source improvement','B: beats both')]:
        y=np.arange(7);ax.hlines(y,np.minimum(l,r),np.maximum(l,r),color='#B6C3CB',lw=2);ax.scatter(l,y,color=c1,s=35,label=lab1);ax.scatter(r,y,color=c2,s=35,marker='s',label=lab2);ax.set_yticks(y,SITES);ax.invert_yaxis();ax.set(xlim=(-3,104),xlabel='Within-site rate (%)',title=title);ax.legend(loc='lower left',bbox_to_anchor=(0,-.31),frameon=False,fontsize=9)
    save(fig,'Figure_4')

def figure5():
    fig=plt.figure(figsize=(10,7.8),layout='constrained');gs=fig.add_gridspec(2,2);axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
    for ax,keys,title in [(axs[0],['F1','F2'],'(a) Adaptation-induced degradation'),(axs[1],['F3-PD','F3-W','F3-both'],'(b) Temporal baseline underperformance'),(axs[2],['F4'],'(c) Seed-dependent directional instability')]:
        for i,k in enumerate(keys):ax.plot(range(5),np.array(COUNTS[k])*100/(453 if k=='F4' else 1359),marker=['o','s','^'][i],color=[BLUE,ORANGE,GREEN][i],lw=2,label=k)
        budget(ax,True);tidy(ax);ax.set(title=title,ylabel='Buildings with F4 (%)' if keys==['F4'] else 'Raw building–seed event rate (%)');ax.set_ylim(0,45 if keys==['F4'] else 100 if keys[0]=='F3-PD' else 70);ax.legend(frameon=False)
    axs[2].text(.98,.92,'F4: improving and degrading seeds\nwithin the same building and budget',ha='right',va='top',transform=axs[2].transAxes,fontsize=10)
    save(fig,'Figure_5')

def figure6():
    fig=plt.figure(figsize=(10,7.4),layout='constrained')
    gs=fig.add_gridspec(2,3,height_ratios=[1,1],wspace=.42,hspace=.48)
    panel_labels=['(a)','(b)','(c)','(d)','(e)']
    for i,day in enumerate(DAYS[1:]):
        ax=fig.add_subplot(gs[i//3,i%3]);items=[r for r in ASSOC if r[0]==day];y=np.arange(3)
        ax.axvline(0,color=GRAY,lw=.8);vals=[r[3] for r in items]
        colors=[BLUE if v<0 else ORANGE for v in vals]
        ax.scatter(vals,y,c=colors,s=34,marker='o',edgecolors='white',linewidths=.45,zorder=3)
        ax.set_yticks(y,[f'{r[1]}\n(n = {r[2]})' for r in items],fontsize=8.2)
        ax.invert_yaxis();ax.set_xlim(-.50,.50);ax.set_xticks([-.5,0,.5]);ax.tick_params(axis='x',labelsize=8)
        ax.set_title(f'{panel_labels[i]} {day}-day budget',loc='left',fontsize=10.2)
        ax.set_ylim(2.55,-.55);ax.grid(axis='x',color=GRID,lw=.5);ax.set_axisbelow(True)
        for j,r in enumerate(items):
            ha='left' if r[3] <= 0.34 else 'right';dx=6 if ha=='left' else -6
            ax.annotate(f'{r[3]:+.3f}',(r[3],j),xytext=(dx,0),textcoords='offset points',ha=ha,va='center',fontsize=7.8,color=INK)
        if i in (3,4):ax.set_xlabel('Spearman ρ with ΔMAE',fontsize=9)
    ax=fig.add_subplot(gs[1,2]);ax.axis('off')
    ax.set_title('(f) Information boundary',loc='left',fontsize=10.2,pad=8)
    ax.add_patch(Rectangle((.03,.59),.94,.22,transform=ax.transAxes,facecolor='white',edgecolor=BLUE,lw=.9))
    ax.text(.50,.70,'Initial 24-h context\n+ budget-available\nobservations',transform=ax.transAxes,ha='center',va='center',fontsize=7.5,linespacing=1.05)
    ax.annotate('',xy=(.50,.46),xytext=(.50,.59),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color=GRAY,lw=.9))
    ax.add_patch(Rectangle((.03,.23),.94,.23,transform=ax.transAxes,facecolor='#F2F2F2',edgecolor=GRAY,lw=.9))
    ax.text(.50,.345,'Fixed evaluation\nperiod: truth excluded\nfrom characteristics',transform=ax.transAxes,ha='center',va='center',fontsize=7.5,linespacing=1.05)
    ax.text(.03,.07,'All cutoffs precede evaluation.',transform=ax.transAxes,fontsize=8.5,color=GRAY)
    fig.suptitle('Three largest absolute descriptive associations per adaptation budget',x=.01,ha='left',fontsize=12,weight='bold')
    save(fig,'Figure_6')

def figure7():
    d=Diagram('Figure_7',9.5)
    d.text(30,15,'(a) Transfer and numerical-consistency validation',14,True)
    for x,w,s in [(30,270,'Frozen source-only checkpoint\nSeed 20260924 / 5,912 parameters'),(360,280,'Integrity and runtime checks\nWeights, shapes, FP32, eval mode'),(700,270,'Jetson CUDA validation\n64 vectors / 1,536 outputs')]:d.box(x,70,w,80,s,size=10)
    d.line(300,110,360,110,arrow=True);d.line(640,110,700,110,arrow=True)
    d.box(90,180,820,70,'Predefined consistency criteria: PASS\nMaximum absolute difference: 1.4782 × 10^-5 normalized / 0.00524 original scale',fill='#EAF4F0',edge=GREEN,size=11)
    d.text(30,290,'(b) Latency measurement boundaries',14,True)
    d.text(30,344,'Model-only',11,True,w=160)
    d.box(230,329,195,55,'GPU-resident input');d.box(475,329,195,55,'Model forward');d.box(720,329,250,55,'GPU-resident output');d.line(425,356,475,356,arrow=True);d.line(670,356,720,356,arrow=True)
    d.text(30,428,'Deployment\nend-to-end',11,True,w=170,h=60)
    for x,s in [(230,'CPU → GPU'),(475,'Model forward'),(720,'GPU → CPU')]:d.box(x,417,195 if x<720 else 250,55,s)
    d.line(425,445,475,445,arrow=True);d.line(670,445,720,445,arrow=True)
    d.text(230,494,'Excluded: disk I/O, Python startup, checkpoint loading',10,color=GRAY,w=720)
    d.text(30,548,'(c) Batch-1 FP32 latency summaries',14,True)
    d.text(350,590,'Median ●     P95 │',11,w=500)
    x0,x1=345,895
    for i,(dev,mode,mean,med,p95,p99) in enumerate(LATENCY):
        y=650+i*52;c=BLUE if dev.startswith('Jetson') else ORANGE
        d.text(35,y-10,dev+' / '+mode,10,w=310)
        d.line(x0,y,x1,y,color='#E4E9EC');d.line(x0+med/1.1*(x1-x0),y,x0+p95/1.1*(x1-x0),y,color=c)
        d.ax.plot(x0+med/1.1*(x1-x0),y,'o',color=c,ms=7);d.cell('',x0+med/1.1*(x1-x0)-4,y-4,8,8,f'ellipse;fillColor={c};strokeColor={c};')
        d.line(x0+p95/1.1*(x1-x0),y-8,x0+p95/1.1*(x1-x0),y+8,color=c)
        d.text(905,y-10,f'{med:.3f}\n{p95:.3f}',9,w=80,h=40)
    for v in [0,.2,.4,.6,.8,1.0]:d.text(x0+v/1.1*(x1-x0),837,f'{v:.1f}',9,align='center',w=40)
    d.text(565,865,'Latency (ms)',10,align='center',w=250)
    d.text(35,900,'5,000 observations per mode and device; no outlier removal. Median and P95 are not a full distribution.',9,w=945)
    d.finish()

def figure8_layout():
    d=Diagram('Figure_8_LAYOUT_ONLY',8.5);d.text(30,20,'FIGURE 8 — LAYOUT ONLY — CASE DATA REQUIRED',14,True,color=ORANGE)
    panels=[('(a) Repeated adaptation-induced degradation','Hog_office_Roger • 1 day • seed 20260926','Plot all 672 hourly targets: observed, source-only, adapted.\nSelection: most seed-mean F1 budgets; selection value = 5.'),('(b) Seed-dependent directional instability','Eagle_education_Brooke • 14 days • all three seeds','Plot per-seed ΔMAE with a zero reference line.\nReported maximum–minimum seed range = 226.533.'),('(c) MAE improvement with RMSE degradation','Hog_office_Gustavo • 1 day • seed 20260926','Plot all 672 hourly targets: observed, source-only, adapted.\nΔMAE < 0; reported ΔRMSE = +6.484847.')]
    for i,(title,sub,body) in enumerate(panels):
        y=90+i*240;d.text(30,y,title,13,True);d.text(30,y+40,sub,11);d.box(30,y+82,940,110,body,fill='#F6F7F8',edge=GRAY,size=11)
    d.finish(True)

if __name__=='__main__':
    for fun in [figure1,figure2,figure3,figure4,figure5,figure6,figure7,figure8_layout]:fun()
    print(ROOT)
