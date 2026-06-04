#!/usr/bin/env python3
"""Build cardiovascular-risk variables used in the CVR lesion analyses."""

import gc
import math
import pickle
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import nibabel as nib
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf


def read_table(path, header=True, sep=None):
    return pd.read_csv(path, sep=sep if sep is not None else r"\s+", header=0 if header else None, engine="python")


def write_table(df, path, header=True):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(df).to_csv(path, sep=" ", index=False, header=header)


def load_nifti(path):
    p = Path(path)
    candidates = [p] if p.exists() else [Path(str(p) + ext) for ext in (".nii.gz", ".nii")]
    for c in candidates:
        if c.exists():
            img = nib.load(str(c)); return img.get_fdata(), img
    raise FileNotFoundError(path)


def save_nifti(data, path, like=None):
    out = Path(path)
    if not (str(out).endswith(".nii") or str(out).endswith(".nii.gz")):
        out = Path(str(out) + ".nii.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), like.affine if like is not None else np.eye(4), like.header.copy() if like is not None else None), str(out))


def r_linear_get(arr, indices):
    return np.asarray(arr).ravel(order="F")[np.asarray(indices).astype(int).ravel() - 1]


def load_rdata(path):
    path = str(path)
    try:
        import pyreadr
        res = pyreadr.read_r(path)
        return dict(res.items())
    except Exception:
        pass
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_rdata(path, **objects):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(objects, fh, protocol=pickle.HIGHEST_PROTOCOL)


def binarize(data):
    return (np.asarray(data) >= 0.5).astype(np.uint8)


def center(s):
    return s - np.nanmean(s)


def describe(x, name):
    s = pd.Series(np.asarray(x).ravel()).dropna(); print(f"{name}: n={len(s)} min={s.min() if len(s) else np.nan} mean={s.mean() if len(s) else np.nan} max={s.max() if len(s) else np.nan}")



TEMPDIR = "/well/nichols/users/kindalov/FMRIB/Longitudinal/prelim/temp"

def _col(df, name):
    return pd.to_numeric(df[name], errors="coerce")

def smoking_score(df, visit):
    n=len(df); status=_col(df,f"X20116.{visit}.0").replace({-3:np.nan})
    start=_col(df,f"X3436.{visit}.0").where(lambda s: ~s.isin([-1,-3]))
    start=start.combine_first(_col(df,f"X2867.{visit}.0").where(lambda s: ~s.isin([-1,-3])))
    stop=_col(df,f"X6194.{visit}.0").where(lambda s: ~s.isin([-1,-3]))
    stop=stop.combine_first(_col(df,f"X2897.{visit}.0").where(lambda s: ~s.isin([-1,-3])))
    years=(stop-start).where(stop.notna(), df[f"age_vis{visit}"]-start)
    num=_col(df,f"X6183.{visit}.0").where(lambda s: s>=0).combine_first(_col(df,f"X2887.{visit}.0").where(lambda s: s>=0))
    stopped=((_col(df,f"X2907.{visit}.0")==1)|(_col(df,f"X3486.{visit}.0")==2)).astype(float)
    yearscomp=years.where(stopped==0, years-0.5); yearscomp=yearscomp.mask(years==0,0)
    pack=(num/20)*yearscomp; pack=pack.mask(yearscomp==0,0).mask((start<=16)&(stop<=16)).mask((start==stop)&(stopped==1),0)
    pack=pack.mask(status==0,0)
    score=pd.Series(np.nan,index=df.index); score[pack<10]=0; score[(pack>=10)&(pack<50)]=1; score[pack>=50]=2; score[status==0]=0
    return score

def medication(df, visit):
    m=_col(df,f"X6177.{visit}.0"); f=_col(df,f"X6153.{visit}.0")
    chol=pd.Series(0.0,index=df.index); bp=pd.Series(0.0,index=df.index)
    chol[(m==1)|(f==1)]=1; bp[(m==2)|(f==2)]=1
    bad=(m.isin([-1,-3])|f.isin([-1,-3])|(m.isna()&f.isna()))
    chol[bad]=np.nan; bp[bad]=np.nan
    return chol,bp

def diabetes(df, visit):
    return _col(df,f"X2443.{visit}.0").replace({-1:np.nan,-3:np.nan})

def bp_risk(df, visit, med_bp):
    da=(_col(df,f"X4079.{visit}.0")+_col(df,f"X4079.{visit}.1"))/2; dm=(_col(df,f"X94.{visit}.0")+_col(df,f"X94.{visit}.1"))/2
    sa=(_col(df,f"X4080.{visit}.0")+_col(df,f"X4080.{visit}.1"))/2; sman=(_col(df,f"X93.{visit}.0")+_col(df,f"X93.{visit}.1"))/2
    high=((da>=90)|(dm>=90)|(sa>=140)|(sman>=140)); missing=da.isna()&dm.isna()&sa.isna()&sman.isna()
    bp=pd.Series(0.0,index=df.index); bp[high]=1; bp[missing]=np.nan
    risk=pd.Series(np.nan,index=df.index); risk[(med_bp==1)|(bp==1)]=1; risk[(med_bp==0)&(bp==0)]=0
    return risk, sa.combine_first(sman)

def whr(df, visit):
    ratio=_col(df,f"X48.{visit}.0")/_col(df,f"X49.{visit}.0")
    out=pd.Series(0.0,index=df.index); out[(df["X31.0.0"]==0)&(ratio>=0.85)]=1; out[(df["X31.0.0"]==1)&(ratio>=0.9)]=1; out[ratio.isna()]=np.nan
    return out

def main():
    vis2=read_table("/well/nichols/users/kindalov/FMRIB/Longitudinal/funpack/Vis2_CVR.tsv",sep="\t"); vis2=vis2.rename(columns={vis2.columns[0]:"eid_34077"})
    vis3=read_table("/well/nichols/users/kindalov/FMRIB/Longitudinal/funpack/Vis3_CVR.tsv",sep="\t"); vis3=vis3.rename(columns={vis3.columns[0]:"eid_34077"})
    df=read_table(Path(TEMPDIR)/"df_visits_cleaned_Apr2021.dat").drop(columns=[c for c in ["X53.2.0","X53.3.0","X25000.2.0","X25000.3.0"] if c in read_table(Path(TEMPDIR)/"df_visits_cleaned_Apr2021.dat").columns])
    dfv=df.merge(vis2,on="eid_34077",how="left").merge(vis3,on="eid_34077",how="left")
    chol2,bpm2=medication(dfv,2); chol3,bpm3=medication(dfv,3)
    dia2, dia3 = diabetes(dfv,2), diabetes(dfv,3); dia2[(dia2.isna())&(dia3==0)]=0; dia3[(dia3.isna())&(dia2==1)]=1
    bp2, sys1 = bp_risk(dfv,2,bpm2); bp3, sys2 = bp_risk(dfv,3,bpm3)
    apoe=read_table("/well/nichols/users/kindalov/ApoE_Extract/ApoE.dat"); apoe["apoe_score"]=np.where(apoe["e3.e4"]==1,1,0); apoe["apoe_score"]=np.where(apoe["e4.e4"]==1,2,apoe["apoe_score"]); apoe=apoe.rename(columns={apoe.columns[2]:"eid_34077"})
    dfv["smoking_score_vis2"]=smoking_score(dfv,2); dfv["smoking_score_vis3"]=smoking_score(dfv,3)
    dfv["medication_chol_vis2"]=chol2; dfv["medication_chol_vis3"]=chol3; dfv["BP_risk_vis2"]=bp2; dfv["BP_risk_vis3"]=bp3
    dfv["diabetes_vis2"]=dia2; dfv["diabetes_vis3"]=dia3; dfv["whr_indicator_vis2"]=whr(dfv,2); dfv["whr_indicator_vis3"]=whr(dfv,3); dfv["systolic_vis1"]=sys1; dfv["systolic_vis2"]=sys2
    dfv=dfv.merge(apoe[["eid_34077","apoe_score"]],on="eid_34077",how="left")
    v2=["smoking_score_vis2","medication_chol_vis2","BP_risk_vis2","diabetes_vis2","whr_indicator_vis2","apoe_score"]
    v3=["smoking_score_vis3","medication_chol_vis3","BP_risk_vis3","diabetes_vis3","whr_indicator_vis3","apoe_score"]
    dfv["CVR_vis2"]=dfv[v2].sum(axis=1,min_count=len(v2)); dfv["CVR_vis3"]=dfv[v3].sum(axis=1,min_count=len(v3))
    complete_df=dfv.dropna(subset=["CVR_vis2","CVR_vis3"])
    save_rdata(Path(TEMPDIR)/"CVR_9June2021.Rdata", df_visits=dfv, complete_df=complete_df)
    print(complete_df[["CVR_vis2","CVR_vis3"]].describe())
if __name__ == "__main__": main()
