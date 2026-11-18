"""Exact frozen training/evaluation functions, relocated for portability.
See SOURCE_MAP.json. No historical experiment imports or local absolute paths.
"""
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC,SVC
from owner_dog import engine
from owner_dog.classifiers import fit_head,score_head
BASE=Path(__file__).resolve().parent
PLAN=json.loads((BASE/'data/plan.json').read_text(encoding='utf8'))
PHOTO=json.loads((BASE/'data/photo_selection.json').read_text(encoding='utf8'))
ORDERS={r:np.array(PHOTO['orders'][r]) for r in ['person','dog']}
SPECS={'joint_eye8':dict(kind='joint',dim=8,shared=False,radius=False)}
PARAMS={'logistic':{'lambda':.01},'linear_svm':{'lambda':.01},'rbf_svm':{'C':1000.,'gamma':'scale'},'rank_svm':{'lambda':.01}}

class Representation:
    def __init__(self,name):self.name=name;self.spec=SPECS[name];self.pcas={};self.fill={}
    def prepare(self,inputs,tr):
        for role in ['person','dog']:
            x=inputs[role+'__eye'][tr];self.fill[role]=engine.normalize_rows(np.nanmean(x,axis=(0,1))[None,:])[0]
        return self
    def photos(self,inputs,role,branch):
        global_=inputs[role+'__global'];eye=inputs[role+'__eye'];eye=np.where(np.isfinite(eye),eye,self.fill[role])
        eye=engine.normalize_rows(eye.reshape(600,20)).reshape(200,3,20)
        if branch=='global':return global_
        if branch=='eye':return eye
        return engine.normalize_rows(np.concatenate([global_,eye],axis=-1).reshape(600,-1)).reshape(200,3,-1)
    def fit(self,inputs,tr):
        self.prepare(inputs,tr);kind=self.spec['kind'];branches=['global','eye'] if kind=='split' else [kind]
        for branch in branches:
            values={r:engine.make_set_features(self.photos(inputs,r,branch),ORDERS[r],3,0) for r in ['person','dog']}
            dim=4 if kind=='split' else self.spec['dim']
            if self.spec['shared']:
                pc=PCA(n_components=dim,svd_solver='full',whiten=False).fit(np.r_[values['person'][tr],values['dog'][tr]])
                for r in values:self.pcas[r,branch]=pc
            else:
                for r,x in values.items():self.pcas[r,branch]=PCA(n_components=dim,svd_solver='full',whiten=False).fit(x[tr])
        return self
    def transform(self,inputs,role,k,path=0):
        kind=self.spec['kind'];out=[]
        for branch in (['global','eye'] if kind=='split' else [kind]):
            x=engine.make_set_features(self.photos(inputs,role,branch),ORDERS[role],k,path)
            z=self.pcas[role,branch].transform(x)
            out.append(z if self.spec['radius'] else engine.normalize_rows(z))
        if len(out)==1:return out[0]
        z=np.concatenate(out,axis=1)
        return z if self.spec['radius'] else engine.normalize_rows(z)

class Front(Representation):
    def __init__(self,k):super().__init__('joint_eye8');self.k=k
    def prepare(self,inputs,tr):
        for role in ['person','dog']:
            x=inputs[role+'__eye']
            if self.k<3:
                selected=engine.selected_photos(ORDERS[role],self.k,0)
                x=x[np.arange(len(x))[:,None],selected]
            self.fill[role]=engine.normalize_rows(np.nanmean(x[tr],axis=(0,1))[None,:])[0]
        return self
    def fit(self,inputs,tr):
        self.prepare(inputs,tr)
        for role in ['person','dog']:
            x=engine.make_set_features(self.photos(inputs,role,'joint'),ORDERS[role],self.k,0)
            self.pcas[role,'joint']=PCA(n_components=8,svd_solver='full',whiten=False).fit(x[tr])
        return self

def design(rep,inputs,tr,k,negatives):
    h=rep.transform(inputs,'person',k,0)[tr];d=rep.transform(inputs,'dog',k,0)[tr]
    return engine.pair_design(h,d,tr,20261102,negatives,'separate')

def original_head_fit(dd,name,param,seed):
    if name!='rank_svm':return fit_head(dd['x'],dd['y'],dd['weights'],name,param,seed)
    # One query's true pair must outrank its sampled false pairs. No cross-query differences.
    scaler=StandardScaler().fit(dd['x'],sample_weight=dd['weights']);z=scaler.transform(dd['x'])
    n=int(dd['y'].sum());positions=dd['positions'];query=positions//n;truth=z[dd['y']==1]
    bad=dd['y']==0;delta=truth[query[bad]]-z[bad]
    x=np.r_[delta,-delta];y=np.r_[np.ones(len(delta)),np.zeros(len(delta))];w=np.full(len(y),1/len(y))
    classifier=LinearSVC(C=1/param['lambda'],fit_intercept=False,dual=False,tol=1e-8,max_iter=10000)
    classifier.fit(x,y,sample_weight=w)
    return dict(scaler=scaler,classifier=classifier,name=name,params=param)

def head_fit(dd,name,param,seed):
    if name!='rbf_svm':return original_head_fit(dd,name,param,seed)
    scaler=StandardScaler().fit(dd['x'],sample_weight=dd['weights'])
    classifier=SVC(C=param['C'],gamma=param['gamma'],cache_size=512)
    classifier.fit(scaler.transform(dd['x']),dd['y'],sample_weight=dd['weights'])
    return dict(scaler=scaler,classifier=classifier,name=name,params=param)

def evaluate(rep,head,inputs,split,train_k,name,counts=(1,3)):
    # Score the SAME frozen ten candidates, not the thirty unused dogs per query.
    # This changes neither candidates nor predictions; it avoids redundant SVC work.
    te=np.asarray(split['test']);lookup={int(v):j for j,v in enumerate(te)};out=[]
    columns=np.array([[lookup[int(v)] for v in row] for row in split['galleries']['mixed10']])
    for k in counts:
        for path in ([0] if k==3 else [0,1,2]):
            h=rep.transform(inputs,'person',k,path)[te];d=rep.transform(inputs,'dog',k,path)[te]
            x=engine.validation_features(h,d,te,split['galleries']['mixed10'],'separate')
            values=score_head(head,x).reshape(len(te),10);matrix=np.full((len(te),len(te)),np.nan)
            matrix[np.arange(len(te))[:,None],columns]=values
            case=dict(id=name,feature_group='joint_eye8',photos=k,path=path)
            out += [dict(r,train_photos=train_k) for r in engine.query_rows(matrix,split,name,case,'actual_train_photo_sweep')]
    return out

def averages(rr):return {k:float(np.mean([r[k] for r in rr])) for k in ['top1','top3','mrr']}

def per_owner(rr,k):
    d=defaultdict(list)
    for r in rr:
        if r['photos']==k:d[int(r['pair_id'])].append([r['top1'],r['top3'],r['mrr']])
    return {i:np.mean(x,0) for i,x in d.items()}
