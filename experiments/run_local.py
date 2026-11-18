"""Run the frozen experiment with separately supplied local research features."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,csv,hashlib,json,time
from pathlib import Path
from collections import defaultdict
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from pipeline import BASE,PLAN,PARAMS,Front,design,head_fit,evaluate,averages

OUT=BASE/'outputs/public_reproduction'
def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def train_subset(split,n,repeat):
    pool=np.asarray(split['train'])
    if n==160:return pool.copy()
    order=np.random.default_rng(np.random.SeedSequence([split['seed'],split['fold'],repeat,901])).permutation(pool)
    return np.sort(order[:n])
def run(args):
    path=args.features
    if not path.exists():raise SystemExit('Local research features are required. See experiments/README.md; images and feature caches are not public.')
    fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
    raw=dict(np.load(path,allow_pickle=False))
    for role in ['person','dog']:
        assert raw[role+'__global'].shape==(200,3,2724)
        assert raw[role+'__eye'].shape==(200,3,20)
    dest=OUT/f'seed{args.seed}';dest.mkdir(parents=True,exist_ok=True)
    for split in [s for s in PLAN['outer'] if s['seed']==args.seed and s['fold'] in args.folds]:
        for n in args.sizes:
            for repeat in ([0] if n==160 else range(3)):
                tr=train_subset(split,n,repeat);assert not set(tr)&set(split['test'])
                for k in args.photos:
                    rep=Front(k).fit(raw,tr);dd=design(rep,raw,tr,k,None)
                    assert len(dd['y'])==n*n and int(dd['y'].sum())==n
                    for name in args.models:
                        target=dest/f"f{split['fold']}_n{n}_r{repeat}_k{k}_{name}.joblib"
                        if target.exists():
                            prior=joblib.load(target)
                            if prior['input_sha256']!=fingerprint or prior['params']!=PARAMS[name]:raise ValueError('Checkpoint input/config mismatch; use a separate outputs directory.')
                            continue
                        tick=time.monotonic();head=head_fit(dd,name,PARAMS[name],args.seed)
                        rows=[dict(row,actual_train_size=n,subset_repeat=repeat) for row in evaluate(rep,head,raw,split,k,name,counts=(1,3))]
                        joblib.dump(dict(params=PARAMS[name],input_sha256=fingerprint,records=rows,representation=rep,head=head),target)
                        print(args.seed,split['fold'],n,k,name,round(time.monotonic()-tick,2),'s',flush=True)
def analyze():
    groups=defaultdict(list);cases=0
    for split in PLAN['outer']:
        for n in [40,80,120,160]:
            for repeat in ([0] if n==160 else range(3)):
                for k in [1,2,3]:
                    for name in PARAMS:
                        path=OUT/f"seed{split['seed']}"/f"f{split['fold']}_n{n}_r{repeat}_k{k}_{name}.joblib"
                        if not path.exists():raise SystemExit(f'Full comparison is incomplete: {path.name}. No partial results will be reported as the full experiment.')
                        saved=joblib.load(path);assert saved['params']==PARAMS[name];cases+=1
                        for row in saved['records']:groups[name,n,k,int(row['photos'])].append(row)
    result=[dict(model=name,actual_train_size=n,train_photos=k,eval_photos=v,**averages(rr)) for (name,n,k,v),rr in groups.items()]
    write_csv(OUT/'summary.csv',result);print('Aggregated',cases,'model cases into',len(result),'conditions.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,default=BASE/'data/features.npz');p.add_argument('--seed',type=int,choices=[20261101,20261102,20261103],default=20261101)
    p.add_argument('--folds',type=int,nargs='+',choices=range(5),default=list(range(5)))
    p.add_argument('--sizes',type=int,nargs='+',choices=[40,80,120,160],default=[40,80,120,160])
    p.add_argument('--photos',type=int,nargs='+',choices=[1,2,3],default=[1,2,3]);p.add_argument('--models',nargs='+',choices=list(PARAMS),default=list(PARAMS));p.add_argument('--analyze',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=1):analyze() if args.analyze else run(args)
