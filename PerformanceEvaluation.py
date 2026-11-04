"""CRR and verification ROC/FMR/FNMR, with reproducible CSV and figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc


def PerformanceEvaluation(model,X,y,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    summary=[];scores={}
    for metric in ['L1','L2','cosine']:
        dist=model.distances(X,metric)
        pred=model.labels[dist.argmin(axis=1)]
        crr=100*np.mean(pred==y)
        summary.append({'measure':metric,'correct':int(np.sum(pred==y)),'total':len(y),'CRR_percent':crr})
        # Verification scores: every probe is compared with all 108 enrolled class centers.
        genuine=(y[:,None]==model.labels[None,:]).ravel().astype(int)
        similarity=(-dist).ravel()
        fpr,tpr,threshold=roc_curve(genuine,similarity)
        scores[metric]=(fpr,tpr,threshold,auc(fpr,tpr),genuine,similarity)
        pd.DataFrame({'FMR':fpr,'FNMR':1-tpr,'TPR':tpr,'threshold_neg_distance':threshold}).to_csv(out/f'ROC_{metric}.csv',index=False)
    result=pd.DataFrame(summary)
    result.to_csv(out/'Table3_CRR.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(result.measure,result.CRR_percent);ax.axhline(80,color='red',linestyle='--',label='Required 80%');ax.set(ylim=(0,100),ylabel='CRR (%)',title='Identification CRR — 432 test samples');ax.legend();fig.tight_layout();fig.savefig(out/'Figure10_CRR.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for key,(fpr,tpr,_,area,_,_) in scores.items():ax.plot(fpr,tpr,label=f'{key}: AUC={area:.4f}')
    ax.plot([0,1],[0,1],'k--',alpha=.35)
    ax.set(xlabel='False Match Rate (FMR)',ylabel='True Match Rate (1 - FNMR)',title='Verification ROC — class-center comparisons',xlim=(0,1),ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(out/'Figure11_ROC.png',dpi=180);plt.close(fig)
    # Additional verification plot: FMR versus FNMR
    fig, ax = plt.subplots(figsize=(7, 5))

    for metric, (fpr, tpr, _, area, _, _) in scores.items():

        fnmr = 1 - tpr

        ax.plot(
            fpr,
            fnmr,
            label=f'{metric}: AUC={area:.4f}'
        )

    ax.set(
        xlabel='False Match Rate (FMR)',
        ylabel='False Nonmatch Rate (FNMR)',
        title='Verification Performance — FMR vs FNMR',
        xlim=(0, 0.1),
        ylim=(0, 1)
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out / 'Figure11_FMR_FNMR.png',
        dpi=180
    )

    plt.close(fig)
    rows=[]
    for metric,(fpr,tpr,threshold,area,_,_) in scores.items():
        idx=np.argmin(np.abs(fpr-(1-tpr)))
        row={'measure':metric,'AUC':area,'EER_approx_percent':100*(fpr[idx]+1-tpr[idx])/2}
        for target in [0.001,0.01,0.05]:
            positions=np.where(fpr<=target)[0]
            j=positions[-1] if len(positions) else 0
            row[f'FNMR_at_FMR_le_{target:g}_percent']=100*(1-tpr[j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'Table4_Verification.csv',index=False)
    return result,pd.DataFrame(rows)
