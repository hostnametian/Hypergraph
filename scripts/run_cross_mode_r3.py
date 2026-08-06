#!/usr/bin/env python3
"""Cross-mode endpoints for random transfer-edge removal (R3).

Computes LWCC, cross-mode LCC, and cross-mode reachability under identical
transfer-edge attack fractions for all 45 cities. Cross-mode reachability is
estimated from matched bus and metro source samples to keep the calculation
tractable for the largest cities.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from run_resilience_experiments import load_city_inventory, load_city_artifacts

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'results_build_transit_hypergraphs'; ANALYSIS=ROOT/'results_analyze_transit_hypergraphs'
OUT=ROOT/'results_revision'/'cross_mode_r3'; OUT.mkdir(parents=True,exist_ok=True)
CITIES=ROOT/'metadata/cities_with_bus_and_metro.csv'; VERSION='walk_200m'
FRACTIONS=np.linspace(0,1,11); REPS=20; SOURCE_PER_MODE=10; SEED=271828
REPRESENTATIVE_CITIES={'贵阳','成都','滁州','青岛','石家庄','福州'}

def cross_mode_metrics(nodes, src_all, tgt_all, keep, source_per_mode=100, rng=None):
    ids=nodes.node_id.astype(str).to_numpy(); n=len(ids); id2={x:i for i,x in enumerate(ids)}
    src=src_all[keep]; tgt=tgt_all[keep]
    if len(src)==0: return 0.,0.,0.
    A=csr_matrix((np.ones(len(src),dtype=np.int8),(src,tgt)),shape=(n,n))
    active=np.zeros(n,bool); active[src]=True; active[tgt]=True
    _,lab=connected_components(A,directed=True,connection='weak')
    modes=nodes['mode'].astype(str).to_numpy()
    best=0
    for comp in np.unique(lab[active]):
        ix=np.where((lab==comp)&active)[0]
        if np.any(modes[ix]=='bus') and np.any(modes[ix]=='metro'): best=max(best,len(ix))
    bus=np.where(modes=='bus')[0]; metro=np.where(modes=='metro')[0]
    if len(bus)==0 or len(metro)==0:return best/n,0.,0.
    rng = np.random.default_rng() if rng is None else rng
    bsrc=rng.choice(bus,min(source_per_mode,len(bus)),replace=False); msrc=rng.choice(metro,min(source_per_mode,len(metro)),replace=False)
    D=dijkstra(A,directed=True,unweighted=True,indices=np.r_[bsrc,msrc])
    b_to_m=np.isfinite(D[:len(bsrc)][:,metro]).sum(); m_to_b=np.isfinite(D[len(bsrc):][:,bus]).sum()
    denom=len(bsrc)*len(metro)+len(msrc)*len(bus)
    return best/n,(b_to_m+m_to_b)/denom,active.sum()/n

def main():
    inv=load_city_inventory(CITIES); inv=inv[inv['城市中文'].astype(str).isin(REPRESENTATIVE_CITIES)]; rows=[]
    for _,city in inv.iterrows():
        a=load_city_artifacts(city,BUILD,ANALYSIS,VERSION); nodes=a['nodes']; edges=a['projection']
        ids=nodes.node_id.astype(str).to_numpy(); id2={x:i for i,x in enumerate(ids)}
        src_all=np.array([id2.get(x,-1) for x in edges.source_node_id.astype(str)],int)
        tgt_all=np.array([id2.get(x,-1) for x in edges.target_node_id.astype(str)],int)
        valid=(src_all>=0)&(tgt_all>=0); src_all=src_all[valid]; tgt_all=tgt_all[valid]; edges=edges.loc[valid].reset_index(drop=True)
        transfer_ids=edges.loc[edges.edge_kind.astype(str).eq('transfer'),'edge_id'].astype(str).unique()
        if len(transfer_ids)==0: continue
        rng=np.random.default_rng(SEED+hash(str(a['city']))%100000)
        curves=[]
        for rep in range(REPS):
            order=rng.permutation(transfer_ids)
            for f in FRACTIONS:
                nrem=int(round(f*len(order))); removed=set(order[:nrem])
                keep=np.ones(len(edges),bool); kind=edges.edge_kind.astype(str).to_numpy(); eid=edges.edge_id.astype(str).to_numpy()
                keep[(kind=='transfer') & np.isin(eid,list(removed))]=False
                lcc,reach,active=cross_mode_metrics(nodes,src_all,tgt_all,keep,rng=rng)
                # LWCC over all active nodes, using the same projection.
                src=src_all[keep]; tgt=tgt_all[keep]
                if len(src):
                    A=csr_matrix((np.ones(len(src)),(src,tgt)),shape=(len(nodes),len(nodes))); act=np.zeros(len(nodes),bool);act[src]=1;act[tgt]=1
                    _,lab=connected_components(A,directed=True,connection='weak'); lw=max(np.bincount(lab[act]))/len(nodes)
                else: lw=0.
                rows.append(dict(city=a['city'],rep=rep,fraction=f,lwcc=lw,cross_mode_lcc=lcc,cross_mode_reachability=reach,active_fraction=active))
        print(a['city'],'OK')
    d=pd.DataFrame(rows); d.to_csv(OUT/'cross_mode_r3_curves.csv',index=False,encoding='utf-8-sig')
    agg=d.groupby(['city','fraction'])[['lwcc','cross_mode_lcc','cross_mode_reachability']].mean().reset_index(); agg.to_csv(OUT/'cross_mode_r3_city_curves.csv',index=False,encoding='utf-8-sig')
    auc=agg.groupby('city').agg(lwcc_auc=('lwcc',lambda x:np.trapezoid(x,FRACTIONS)),cross_mode_lcc_auc=('cross_mode_lcc',lambda x:np.trapezoid(x,FRACTIONS)),cross_mode_reachability_auc=('cross_mode_reachability',lambda x:np.trapezoid(x,FRACTIONS))).reset_index(); auc.to_csv(OUT/'cross_mode_r3_auc.csv',index=False,encoding='utf-8-sig')
    print(auc.describe().to_string())
if __name__=='__main__': main()
