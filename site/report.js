'use strict';
const $=id=>document.getElementById(id);
const models={logistic:'로지스틱',linear_svm:'선형 SVM',rbf_svm:'RBF-SVM',rank_svm:'RankSVM'};
const names={full:'전체 특징',without_hog:'HOG 제거',without_global_lbp:'전체 LBP 제거',without_hsv:'HSV 제거',without_eye_lbp:'눈 LBP 제거'};
const colors={top1:'#285440',top3:'#a3b68a'};
let data=null,evaluation=1;
const pct=x=>(x*100).toFixed(2);
function text(x,y,label,cls='',anchor='start'){return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${label}</text>`;}
function svg(content,height,label){return `<svg viewBox="0 0 680 ${height}" role="img" aria-label="${label}"><title>${label}</title>${content}</svg>`;}
function barChart(rows,dual=true){
 const left=120,right=611,width=right-left,max=70,rowH=dual?66:46,height=rows.length*rowH+63;
 let s='';for(let v=0;v<=70;v+=10){const x=left+v/max*width;s+=`<line class="grid-line" x1="${x}" x2="${x}" y1="8" y2="${height-30}"/>`+text(x,height-10,`${v}%`,'','middle');}
 rows.forEach((r,i)=>{const y=22+i*rowH;s+=text(left-16,y+11,r.label,'bar-label','end');for(const [j,key] of (dual?['top1','top3']:['top3']).entries()){const yy=y+j*20,value=r[key]*100;s+=`<rect x="${left}" y="${yy-4}" width="${value/max*width}" height="13" rx="2" fill="${dual?colors[key]:(i===0?colors.top1:colors.top3)}"><title>${r.label} ${key==='top1'?'Top-1':'Top-3'} ${value.toFixed(2)}%</title></rect>`+text(left+value/max*width+7,yy+7,value.toFixed(2),'value-label');}});
 return svg(s,height,dual?`모델 비교, 평가 ${evaluation===1?'단일 사진':'3장 평균'}`:'단일 사진 특징 제거 실험, Top-3');
}
function learningChart(rows){
 const l=55,r=560,t=24,b=248;let s='';const x=i=>l+i*(r-l)/3,y=v=>b-v/70*(b-t);
 for(let v=0;v<=70;v+=10){s+=`<line class="grid-line" x1="${l}" x2="${r+55}" y1="${y(v)}" y2="${y(v)}"/>`+text(l-12,y(v)+4,`${v}%`,'','end');}
 for(const [v,label] of [[10,'무작위 Top-1'],[30,'무작위 Top-3']]){s+=`<line class="baseline" x1="${l}" x2="${r+55}" y1="${y(v)}" y2="${y(v)}"/>`+text(l+8,y(v)-7,label,'baseline-label');}
 for(const key of ['top1','top3']){s+=`<polyline points="${rows.map((row,i)=>`${x(i)},${y(row[key]*100)}`).join(' ')}" fill="none" stroke="${colors[key]}" stroke-width="2.5"/>`;rows.forEach((row,i)=>{const value=row[key]*100;s+=`<circle cx="${x(i)}" cy="${y(value)}" r="4.5" fill="${colors[key]}"><title>${row.actual_train_size}쌍 ${key} ${value.toFixed(2)}%</title></circle>`+text(x(i),y(value)-12,value.toFixed(2),'value-label','middle');});s+=text(r+13,y(rows[3][key]*100)+5,key==='top1'?'Top-1':'Top-3','series-label');}
 rows.forEach((row,i)=>{s+=text(x(i),b+25,`${row.actual_train_size}쌍`,'','middle');});s+=text((l+r)/2,b+51,'실제 학습 관계 수','','middle');return svg(s,310,'학습 관계 수 40, 80, 120, 160쌍의 Top-1, Top-3');
}
function render(){
 const rows=data.comparison.filter(r=>r.eval_photos===evaluation);
 const modelRows=Object.keys(models).map(model=>({...rows.find(r=>r.model===model&&r.actual_train_size===160&&r.train_photos===3),label:models[model]}));
 const photos=[1,2,3].map(k=>rows.find(r=>r.model==='logistic'&&r.actual_train_size===160&&r.train_photos===k));
 const curve=[40,80,120,160].map(n=>rows.find(r=>r.model==='logistic'&&r.actual_train_size===n&&r.train_photos===3));
 const headline=modelRows[0];$('headlineTop1').innerHTML=pct(headline.top1)+'<span>%</span>';$('headlineTop3').innerHTML=pct(headline.top3)+'<span>%</span>';
 $('evalExplanation').textContent=evaluation===1?'한 번의 점수 계산에 사람 1장·개별 1장. 고정 세 사진 경로의 지표를 평균합니다.':'평가에서도 대상별 3장의 특징을 평균합니다. 단일 사진 입력보다 더 많은 정보를 사용하는 별도 조건입니다.';
 $('modelRows').innerHTML=modelRows.map(r=>`<tr class="${r.model==='logistic'?'selected-row':''}"><td>${r.label}</td><td>${pct(r.top1)}</td><td>${pct(r.top3)}</td></tr>`).join('');
 $('photoRows').innerHTML=photos.map(r=>`<tr><td>${r.train_photos}장</td><td>${pct(r.top1)}</td><td>${pct(r.top3)}</td></tr>`).join('');
 $('learningRows').innerHTML=curve.map(r=>`<tr><td>${r.actual_train_size}쌍</td><td>${pct(r.top1)}</td><td>${pct(r.top3)}</td></tr>`).join('');
 $('modelChart').innerHTML=barChart(modelRows);$('learningChart').innerHTML=learningChart(curve);
 document.querySelectorAll('[data-eval]').forEach(b=>{const on=Number(b.dataset.eval)===evaluation;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on));});
}
document.querySelectorAll('[data-eval]').forEach(b=>b.addEventListener('click',()=>{if(!data)return;evaluation=Number(b.dataset.eval);render();}));
async function init(){try{const response=await fetch('data/results.json');if(!response.ok)throw Error('data');data=await response.json();render();$('ablationChart').innerHTML=barChart(data.ablation.map(r=>({...r,label:names[r.name]})),false);}catch{document.querySelectorAll('[data-eval]').forEach(b=>b.disabled=true);$('evalExplanation').textContent='그래프를 불러오지 못했습니다. 아래 표는 단일 사진 평가 기준이며 전체 결과는 CSV로 확인할 수 있습니다.';}}
const links=[...document.querySelectorAll('.toc a')];
if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(!entry.isIntersecting)continue;links.forEach(a=>a.classList.toggle('current',a.hash==='#'+entry.target.id));}},{rootMargin:'-10% 0px -65% 0px'});document.querySelectorAll('article>section').forEach(s=>observer.observe(s));}
init();
