"""Build the site's data from the public aggregate CSVs; no model fitting."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
def rows(name):
    with (ROOT/'site/data'/name).open(encoding='utf-8-sig',newline='') as f:
        out=list(csv.DictReader(f))
    for row in out:
        for k,v in row.items():
            if k in ['actual_train_size','train_photos','eval_photos','n_eval_photos']:row[k]=int(v)
            elif k in ['top1','top3','mrr']:row[k]=float(v)
    return out
if __name__=='__main__':
    result=dict(comparison=rows('model_comparison.csv'),ablation=rows('feature_ablation.csv'),selection=rows('pca_selection.csv'),units='fraction',candidates=10,relations=200,train_per_fold=160,test_per_fold=40,folds=5,seeds=3)
    (ROOT/'site/data/results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Built public report data: 96 comparison conditions.')
