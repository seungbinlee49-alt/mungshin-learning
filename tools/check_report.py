"""Dependency-free checks for public results, links and data boundaries."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import csv,json,math,re
ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'site'
class Page(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.append(attrs['id'])
        for key in ['href','src']:
            if attrs.get(key):self.links.append(attrs[key])
def read_csv(name):
    with (SITE/'data'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    obj=json.loads((SITE/'data/results.json').read_text(encoding='utf8'))
    assert len(obj['comparison'])==96 and len(obj['ablation'])==5
    csv_rows=read_csv('model_comparison.csv')
    for a,b in zip(obj['comparison'],csv_rows):
        assert a['model']==b['model']
        for key in ['actual_train_size','train_photos','eval_photos','top1','top3','mrr']:assert math.isclose(a[key],float(b[key]),abs_tol=1e-14)
        assert 0<=a['top1']<=a['top3']<=1
    keys={(r['model'],r['actual_train_size'],r['train_photos'],r['eval_photos']) for r in obj['comparison']};assert len(keys)==96
    for count,t1,t3 in [(1,22.2222222222,53.8888888889),(3,26.6666666667,61.5)]:
        row=next(r for r in obj['comparison'] if (r['model'],r['actual_train_size'],r['train_photos'],r['eval_photos'])==('logistic',160,3,count))
        assert math.isclose(row['top1']*100,t1,abs_tol=1e-8) and math.isclose(row['top3']*100,t3,abs_tol=1e-8)
    assert obj['candidates']==10 and obj['relations']==200
    ab={r['name']:r for r in obj['ablation']};assert math.isclose((ab['full']['top3']-ab['without_hsv']['top3'])*100,11.5555555556,abs_tol=1e-8)
    page=Page();page.feed((SITE/'index.html').read_text(encoding='utf8'));assert len(page.ids)==len(set(page.ids))
    for link in page.links:
        parsed=urlparse(link)
        if parsed.scheme or parsed.netloc:continue
        if parsed.path:assert (SITE/unquote(parsed.path)).is_file(),link
        elif parsed.fragment:assert parsed.fragment in page.ids,link
    assert not list(SITE.rglob('*.png')) and not list(SITE.rglob('*.jpg')),'Raw photographic assets are not part of this report.'
    for suffix in ['*.npz','*.joblib','*.zip']:assert not list(ROOT.glob('**/'+suffix)),f'Nonpublic research artifact: {suffix}'
    print('PASS: 96 source-aligned conditions, final metrics, ablation, unique IDs, local links, and public asset boundary.')
if __name__=='__main__':main()
