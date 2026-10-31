#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/home/saketh/Codebase')
from dataloader import get_transform
from models.model_convnext_metric import SiameseConvNeXtBMetric

ROOT=Path('/home/saketh/json/final_named_artifacts_20260601')
ANCHOR=ROOT/'trusted_q100_original_only_verify_20260727'
OUT=ROOT/'trusted_q100_watermark_variant_sweep_20260727'
POOL={
 'cifar': ROOT/'phase3_large_pool_subselect_all4_20260717/cifar/large_pool_cache/cifar_pool_asym_k17000_u9000_cifar-generalized-gap-accepted-003_425324b4d3.npz',
 'flickr': ROOT/'phase3_large_pool_subselect_all4_20260717/flickr/large_pool_cache/flickr_pool_asym_k10000_u5000_flickr-fresh-bank-metric-layer-best_a338926af2.npz',
 'ucid': ROOT/'phase3_large_pool_subselect_all4_20260717/ucid/fresh20_unique_4500_850/cache/ucid_fresh20_unique_k4500_u850.npz',
 'amazon': ROOT/'phase3_large_pool_subselect_all4_20260717/amazon/large_pool_cache/amazon_pool_asym_k5000_u5000.npz',
}
PATHS={
 'cifar': ROOT/'phase3_large_pool_subselect_all4_20260717/cifar/large_pool_cache/cifar_pool_asym_k17000_u9000_cifar-generalized-gap-accepted-003_425324b4d3_paths.json',
 'flickr': ROOT/'phase3_large_pool_subselect_all4_20260717/flickr/large_pool_cache/flickr_pool_asym_k10000_u5000_flickr-fresh-bank-metric-layer-best_a338926af2_paths.json',
 'ucid': ROOT/'phase3_large_pool_subselect_all4_20260717/ucid/fresh20_unique_4500_850/cache/ucid_fresh20_unique_k4500_u850_paths.json',
 'amazon': ROOT/'phase3_large_pool_subselect_all4_20260717/amazon/large_pool_cache/amazon_pool_asym_k5000_u5000_paths.json',
}
CKPTS={
 'cifar': Path('/home/saketh/Codebase/checkpoints/generalized_gap_guard_diagonal_search_20260717/cifar/cifar-generalized-gap-accepted-003.pt'),
 'flickr': Path('/home/saketh/Codebase/checkpoints/fresh_bank_metric_layer_search_20attacks_20260715/flickr/flickr-fresh-bank-metric-layer-best.pt'),
 'ucid': Path('/home/saketh/Codebase/checkpoints/fresh_bank_metric_layer_search_20attacks_20260715/ucid/ucid-fresh-bank-metric-layer-best.pt'),
 'amazon': Path('/home/saketh/Codebase/checkpoints/fresh_bank_metric_layer_search_20attacks_20260715/amazon/amazon-fresh-bank-metric-layer-best.pt'),
}
TEXTS=['50% OFF','SALE','AUTHENTIC','VERIFIED','COPY','ORIGINAL','CLEARANCE','FREE SHIPPING','BRAND NEW','LIMITED','MEGA DEAL','HOT PICK','GENUINE','BEST PRICE','NEW ARRIVAL','OFFICIAL','CHECKED','QUALITY','LIMITED STOCK','TRUSTED','CERTIFIED','DISCOUNT','SPECIAL','TOP RATED','VALUE PACK']
POSITIONS=['tl','tr','bl','br','center','top','bottom']
COLORS=[(20,20,20),(255,255,255),(220,40,40),(20,90,220),(20,140,70),(220,120,20)]
TEXT_SCALES=[0.45,0.55,0.65,0.75,0.9,1.05,1.2]
TEXT_ALPHAS=[0.18,0.26,0.34,0.42,0.52]
ASSET_SCALES=[0.12,0.18,0.24,0.30,0.35,0.45,0.55,0.65,0.75]
ASSET_ALPHAS=[0.18,0.28,0.38,0.50,0.65,0.78]
SHAPES=['badge','banner','circle','corner','stamp','ribbon','tag','box']

def selected_indices(qdoc:dict[str,Any], side:str):
    for k in [side+'_indices','selected_'+side+'_indices']:
        if k in qdoc: return [int(x) for x in qdoc[k]]
    raise KeyError(side)

def read_rgb(path):
    im=cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def pos_xy(pos,w,h,bw,bh):
    m=max(4,int(min(w,h)*0.04))
    return {
      'tl':(m,m),'tr':(max(0,w-bw-m),m),'bl':(m,max(0,h-bh-m)),'br':(max(0,w-bw-m),max(0,h-bh-m)),
      'center':(max(0,(w-bw)//2),max(0,(h-bh)//2)),'top':(max(0,(w-bw)//2),m),'bottom':(max(0,(w-bw)//2),max(0,h-bh-m))
    }[pos]

def apply_text(img, var):
    out=img.copy(); h,w=out.shape[:2]
    text=var['text']; scale=var['scale']*max(0.8,min(1.4,min(h,w)/224.0)); thick=max(1,int(round(scale*2)))
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,thick)
    bw,bh=tw+14,th+16; x,y=pos_xy(var['pos'],w,h,bw,bh); yt=min(h-4,y+th+10)
    overlay=out.copy(); bg=(255,255,255) if sum(var['color'])<380 else (20,20,20)
    cv2.rectangle(overlay,(x,y),(min(w-1,x+bw),min(h-1,y+bh)),bg,-1)
    out=cv2.addWeighted(overlay,var['alpha'],out,1-var['alpha'],0)
    cv2.putText(out,text,(x+7,yt),cv2.FONT_HERSHEY_SIMPLEX,scale,var['color'],thick,cv2.LINE_AA)
    return out

def apply_asset(img,var):
    out=img.copy(); h,w=out.shape[:2]
    target=max(10,int(min(h,w)*var['scale'])); aw=max(14,int(target*var['ratio'])); ah=target
    asset=np.zeros((ah,aw,3),dtype=np.uint8); asset[:]=(245,245,245)
    c=var['color']; shape=var['shape']
    if shape=='badge':
        cv2.rectangle(asset,(1,1),(aw-2,ah-2),c,2); cv2.putText(asset,'OK',(max(2,aw//5),max(12,ah//2+4)),cv2.FONT_HERSHEY_SIMPLEX,max(.35,ah/45),c,2,cv2.LINE_AA)
    elif shape=='banner':
        asset[:]=c; cv2.putText(asset,'SALE',(max(2,aw//8),max(12,ah//2+4)),cv2.FONT_HERSHEY_SIMPLEX,max(.35,ah/55),(255,255,255),2,cv2.LINE_AA)
    elif shape=='circle':
        cv2.circle(asset,(aw//2,ah//2),max(4,min(aw,ah)//2-2),c,2); cv2.putText(asset,'V',(max(2,aw//2-6),max(12,ah//2+6)),cv2.FONT_HERSHEY_SIMPLEX,max(.5,ah/38),c,2,cv2.LINE_AA)
    elif shape=='corner':
        pts=np.array([[0,0],[aw,0],[0,ah]],np.int32); cv2.fillConvexPoly(asset,pts,c)
    elif shape=='stamp':
        cv2.rectangle(asset,(2,2),(aw-3,ah-3),c,2); cv2.line(asset,(4,ah-5),(aw-5,4),c,2)
    elif shape=='ribbon':
        asset[:]=c; cv2.circle(asset,(0,ah//2),ah//2,(245,245,245),-1); cv2.circle(asset,(aw,ah//2),ah//2,(245,245,245),-1)
    elif shape=='tag':
        pts=np.array([[0,0],[aw-8,0],[aw,ah//2],[aw-8,ah],[0,ah]],np.int32); cv2.fillConvexPoly(asset,pts,c)
    else:
        cv2.rectangle(asset,(0,0),(aw-1,ah-1),c,-1); cv2.rectangle(asset,(3,3),(aw-4,ah-4),(255,255,255),2)
    x,y=pos_xy(var['pos'],w,h,aw,ah); roi=out[y:y+ah,x:x+aw].astype('float32')
    out[y:y+ah,x:x+aw]=np.clip(roi*(1-var['alpha'])+asset.astype('float32')*var['alpha'],0,255).astype('uint8')
    return out

def build_variants(kind, limit):
    vars=[]
    if kind=='text':
        for scale in sorted(TEXT_SCALES, reverse=True):
            for alpha in TEXT_ALPHAS:
                for pos in POSITIONS:
                    for text in TEXTS:
                        for color in COLORS:
                            vars.append({'kind':'text','text':text,'pos':pos,'scale':scale,'alpha':alpha,'color':color})
                            if len(vars)>=limit: return vars
    else:
        for scale in sorted(ASSET_SCALES, reverse=True):
            for alpha in ASSET_ALPHAS:
                for pos in POSITIONS:
                    for shape in SHAPES:
                        for color in COLORS:
                            vars.append({'kind':'asset','shape':shape,'pos':pos,'scale':scale,'alpha':alpha,'color':color,'ratio':1.6 if shape in {'banner','ribbon','tag'} else 1.0})
                            if len(vars)>=limit: return vars
    return vars

def load_model(ds,device):
    ck=torch.load(CKPTS[ds],map_location=device); m=SiameseConvNeXtBMetric().to(device).eval(); m.load_state_dict(ck['model_state_dict'],strict=False); return m

@torch.no_grad()
def embed_variant(model,paths,device,kind,var,batch=96):
    tr=get_transform(); outs=[]; xs=[]
    for p in paths:
        img=read_rgb(p); img=apply_text(img,var) if kind=='text' else apply_asset(img,var)
        xs.append(tr(img))
        if len(xs)>=batch:
            x=torch.stack(xs).to(device); outs.append(model(x,x)[0].detach().cpu().numpy()); xs=[]
    if xs:
        x=torch.stack(xs).to(device); outs.append(model(x,x)[0].detach().cpu().numpy())
    return np.concatenate(outs).astype('float32')

@torch.no_grad()
def embed_original(model,paths,device,batch=96):
    tr=get_transform(); outs=[]; xs=[]
    for p in paths:
        xs.append(tr(read_rgb(p)))
        if len(xs)>=batch:
            x=torch.stack(xs).to(device); outs.append(model(x,x)[0].detach().cpu().numpy()); xs=[]
    if xs:
        x=torch.stack(xs).to(device); outs.append(model(x,x)[0].detach().cpu().numpy())
    return np.concatenate(outs).astype('float32')

def nearest(bank,q,device):
    bt=torch.from_numpy(np.ascontiguousarray(bank)).float().to(device); outs=[]
    with torch.no_grad():
        for i in range(0,len(q),64):
            qq=torch.from_numpy(np.ascontiguousarray(q[i:i+64])).float().to(device); best=torch.full((len(qq),),float('inf'),device=device)
            for j in range(0,len(bank),4096):
                b=bt[j:j+4096]; d2=((qq[:,None,:]-b[None,:,:])**2).sum(2).min(1).values; best=torch.minimum(best,d2)
            outs.append(torch.sqrt(torch.clamp(best,min=0)).cpu().numpy())
    return np.concatenate(outs).astype('float32')

def eval_variant(model,bank,kp,up,tau,device,kind,var):
    ke=embed_variant(model,kp,device,kind,var); ue=embed_variant(model,up,device,kind,var)
    kd=nearest(bank,ke,device); ud=nearest(bank,ue,device)
    k=float((kd<=tau).mean()*100); u=float((ud>tau).mean()*100); return k,u,(k+u)/2,float(kd.min()),float(ud.min())

def run_dataset(ds,kind,limit,need,device):
    anchor=json.load(open(ANCHOR/f'{ds}_trusted_q100_original_only_verify.json'))
    qdoc=json.load(open(anchor['q_json'])); pdoc=json.load(open(PATHS[ds]))
    ki=selected_indices(qdoc,'known'); ui=selected_indices(qdoc,'unknown')
    kp=qdoc.get('selected_known_paths') or [pdoc['known_paths'][i] for i in ki]
    up=qdoc.get('selected_unknown_paths') or [pdoc['unknown_paths'][i] for i in ui]
    tau=float(anchor['tau']); bank=np.load(POOL[ds],allow_pickle=False)['bank'].astype('float32')
    model=load_model(ds,device)
    oe=embed_original(model,kp,device); ou=embed_original(model,up,device)
    ok=float((nearest(bank,oe,device)<=tau).mean()*100); ouk=float((nearest(bank,ou,device)>tau).mean()*100)
    valid=[]; checked=0
    outdir=OUT/ds; outdir.mkdir(parents=True,exist_ok=True)
    progress=outdir/f'{ds}_{kind}_watermark_variant_sweep_progress.json'
    for vid,var in enumerate(build_variants(kind,limit)):
        checked+=1
        k,u,t,kmin,umin=eval_variant(model,bank,kp,up,tau,device,kind,var)
        rec={'variant_id':vid,**var,'known':round(k,2),'unknown':round(u,2),'total':round(t,2),'known_min_dist':kmin,'unknown_min_dist':umin,'valid': bool(k>=70 and u>=70)}
        if rec['valid']: valid.append(rec)
        if checked%10==0 or rec['valid']:
            progress.write_text(json.dumps({'dataset':ds,'kind':kind,'checked':checked,'valid_count':len(valid),'original_sanity_known':ok,'original_sanity_unknown':ouk,'latest_valid':valid[-10:]},indent=2))
        print(f'[{ds} {kind}] {checked}/{limit} valid={len(valid)} last={k:.1f}/{u:.1f}/{t:.1f}', flush=True)
        if len(valid)>=need: break
    valid_sorted=sorted(valid,key=lambda r:(r['total'], min(r['known'],r['unknown']), r.get('scale',0)), reverse=True)
    payload={'dataset':ds,'kind':kind,'checked':checked,'valid_count':len(valid),'target_valid':need,'original_sanity_known':ok,'original_sanity_unknown':ouk,'anchor_known':anchor['known_accuracy'],'anchor_unknown':anchor['unknown_accuracy'],'tau':tau,'pool':str(POOL[ds]),'q_json':anchor['q_json'],'checkpoint':str(CKPTS[ds]),'valid_variants':valid_sorted,'all_valid_unsorted':valid}
    (outdir/f'{ds}_{kind}_watermark_variant_sweep_results.json').write_text(json.dumps(payload,indent=2))
    lines=[f'# {ds.upper()} {kind} watermark valid variants','',f'Checked: {checked}',f'Valid: {len(valid)}','','| Rank | Variant | Pos | Scale | Alpha | Known | Unknown | Total |','|---:|---|---|---:|---:|---:|---:|---:|']
    for i,r in enumerate(valid_sorted[:100],1):
        name=r.get('text') or r.get('shape')
        lines.append(f"| {i} | `{name}` | {r['pos']} | {r.get('scale',0):.2f} | {r.get('alpha',0):.3f} | {r['known']:.2f} | {r['unknown']:.2f} | {r['total']:.2f} |")
    (outdir/f'{ds}_{kind}_watermark_variant_sweep_results.md').write_text('\n'.join(lines)+'\n')
    return payload

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['cifar','ucid','flickr','amazon']); ap.add_argument('--kind',required=True,choices=['text','asset']); ap.add_argument('--limit',type=int,default=500); ap.add_argument('--need',type=int,default=100); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); run_dataset(args.dataset,args.kind,args.limit,args.need,args.device)
if __name__=='__main__': main()
