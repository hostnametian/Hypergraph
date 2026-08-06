"""Bootstrap stability and model-selection uncertainty for Reviewer 1 comment 5.

Uses the existing 45-city summary matrix; no network simulations are rerun.
Outputs publication-ready CSV files under results_revision/statistical_uncertainty.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results_revision/statistical_uncertainty'
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT/'results_phase7_clustering/city_features.csv')

# Primary hypotheses were defined before inspecting bootstrap draws.
H = [
    ('H1', 'PPCR vs. LWCC AUC (random node removal)', 'transfer_ratio', 'auc_lwcc_R3', 'Pearson r'),
    ('H2', 'PPCR vs. route-support cascade depth', 'transfer_ratio', 'cascade_depth_C1', 'Pearson r'),
]
cas = pd.read_csv(ROOT/'results_run_resilience_cascade/walk_200m/all_cities_resilience_summary_cascade.csv')
cas['tau'] = cas['tau'].astype(float)
rand = cas[cas.attack_type.eq('C1_random_node_tau0.2')].set_index('city')['mean_cascade_depth']
targ = cas[cas.attack_type.eq('C4_targeted_node_tau0.2')].set_index('city')['mean_cascade_depth']
# Cascade summaries use Chinese city labels; align them to the feature matrix.
cn_to_en = dict(zip(df['city_cn'], df['city']))
paired = df['city_cn'].map(targ-rand).to_numpy(float)

rng = np.random.default_rng(20260807)
B = 5000
rows=[]
for hid, desc, xcol, ycol, stat in H:
    x=df[xcol].to_numpy(float); y=df[ycol].to_numpy(float); n=len(x)
    vals=np.empty(B)
    for b in range(B):
        ix=rng.integers(0,n,n)
        vals[b]=pearsonr(x[ix],y[ix])[0] if np.std(x[ix])>0 and np.std(y[ix])>0 else np.nan
    vals=vals[~np.isnan(vals)]
    rows.append({'hypothesis':hid,'comparison':desc,'estimate':pearsonr(x,y)[0],
                 'ci_2.5':np.quantile(vals,.025),'ci_97.5':np.quantile(vals,.975),
                 'bootstrap_reps':len(vals)})
# H3: paired targeted-minus-random cascade depth; positive means targeted deeper.
vals=np.empty(B); n=len(paired)
for b in range(B): vals[b]=np.mean(paired[rng.integers(0,n,n)])
rows.append({'hypothesis':'H3','comparison':'Targeted minus random cascade depth (tau=0.2)',
             'estimate':paired.mean(),'ci_2.5':np.quantile(vals,.025),'ci_97.5':np.quantile(vals,.975),
             'bootstrap_reps':B})
pd.DataFrame(rows).to_csv(OUT/'primary_hypotheses_bootstrap.csv',index=False)

# Model-selection uncertainty: standardized LASSO selection and coefficient stability.
features=['log10_n_nodes','log10_n_hyperedges','transfer_ratio','metro_node_ratio','avg_hyperdegree','avg_hyperedge_size']
outcomes={'H1_static':'auc_lwcc_R3','H2_cascade':'cascade_depth_C1','H3_breadth':'auc_collapse_C4'}
X=df[features].to_numpy(float); X=StandardScaler().fit_transform(X)
sel={k:np.zeros(len(features),int) for k in outcomes}; coef={k:[] for k in outcomes}; r2={k:[] for k in outcomes}
for b in range(2000):
    ix=rng.integers(0,len(df),len(df)); xb=X[ix]
    for k,ycol in outcomes.items():
        y=df[ycol].to_numpy(float); ys=(y[ix]-y[ix].mean())/y[ix].std(ddof=0)
        model=LassoCV(cv=5, random_state=1000+b, n_alphas=100, max_iter=20000).fit(xb,ys)
        c=model.coef_; sel[k]+=np.abs(c)>1e-8; coef[k].append(c)
        pred=model.predict(xb); r2[k].append(1-np.sum((ys-pred)**2)/np.sum((ys-ys.mean())**2))
out=[]
for k in outcomes:
    a=np.asarray(coef[k]);
    for j,f in enumerate(features):
        out.append({'outcome':k,'predictor':f,'selection_frequency':sel[k][j]/2000,
                    'coef_median':np.median(a[:,j]),'coef_ci_2.5':np.quantile(a[:,j],.025),
                    'coef_ci_97.5':np.quantile(a[:,j],.975),'bootstrap_reps':2000})
    out.append({'outcome':k,'predictor':'[model fit] bootstrap R2','selection_frequency':np.nan,
                'coef_median':np.median(r2[k]),'coef_ci_2.5':np.quantile(r2[k],.025),
                'coef_ci_97.5':np.quantile(r2[k],.975),'bootstrap_reps':2000})
pd.DataFrame(out).to_csv(OUT/'lasso_selection_uncertainty.csv',index=False)

# Compare reduced (3-predictor) and full (6-predictor) standardized OLS in bootstrap samples.
reduced=['log10_n_nodes','transfer_ratio','metro_node_ratio']
rows=[]
for k,ycol in outcomes.items():
    y=df[ycol].to_numpy(float); deltas=[]
    for b in range(B):
        ix=rng.integers(0,len(df),len(df)); yy=y[ix];
        def fit(cols):
            xx=StandardScaler().fit_transform(df.loc[ix,cols].to_numpy(float)); yy0=(yy-yy.mean())/yy.std(ddof=0)
            xx1=np.c_[np.ones(len(ix)),xx]; beta=np.linalg.lstsq(xx1,yy0,rcond=None)[0]; pred=xx1@beta
            return 1-np.sum((yy0-pred)**2)/np.sum((yy0-yy0.mean())**2)
        deltas.append(fit(features)-fit(reduced))
    rows.append({'outcome':k,'full_minus_reduced_R2':np.mean(deltas),'ci_2.5':np.quantile(deltas,.025),'ci_97.5':np.quantile(deltas,.975),'bootstrap_reps':B})
pd.DataFrame(rows).to_csv(OUT/'model_comparison_bootstrap.csv',index=False)
print('Wrote', OUT)
