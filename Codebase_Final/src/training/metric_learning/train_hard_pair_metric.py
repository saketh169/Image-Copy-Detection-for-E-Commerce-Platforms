#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/saketh/Codebase')
import multi_split_global_tau_generalized as M
from models.model_convnext_metric import SiameseConvNeXtBMetric

ROOT=Path('/home/saketh/json/final_named_artifacts_20260601')
OUT=ROOT/'metric_layer_pair_hardmine_train20_20260716'
CKROOT=Path('/home/saketh/Codebase/checkpoints/fresh_bank_metric_layer_search_20attacks_20260715')
SAVE_ROOT=Path('/home/saketh/Codebase/checkpoints/metric_layer_pair_hardmine_train20_20260716')
EXCLUDE={'crop40','bg_color_change','heavy_bright','bright','rotate40','flip_v'}
ATTACK_NAMES=[a[0] if isinstance(a,(tuple,list)) else a for a in M.G.ATTACKS]
KEEP=[i for i,a in enumerate(ATTACK_NAMES) if a not in EXCLUDE]
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

def filt(arr,q):
    return np.concatenate([arr[i*q:(i+1)*q] for i in KEEP],axis=0).astype('float32')

def metric_np(bank, query, scale, qchunk=64, bchunk=4096):
    bt=torch.from_numpy(np.ascontiguousarray(bank)).float().to(DEVICE)
    sc=torch.from_numpy(scale.astype('float32')).float().to(DEVICE)
    outs=[]; idxs=[]
    with torch.no_grad():
        for i in range(0,len(query),qchunk):
            qt=torch.from_numpy(np.ascontiguousarray(query[i:i+qchunk])).float().to(DEVICE)
            best=torch.full((qt.shape[0],),float('inf'),device=DEVICE); besti=torch.zeros((qt.shape[0],),dtype=torch.long,device=DEVICE)
            for j in range(0,len(bank),bchunk):
                bb=bt[j:j+bchunk]
                d2=(((qt[:,None,:]-bb[None,:,:])*sc.reshape(1,1,-1))**2).sum(2)
                vals,ii=d2.min(1); mask=vals<best
                best=torch.where(mask,vals,best); besti=torch.where(mask,ii+j,besti)
            outs.append(torch.sqrt(torch.clamp(best,min=0)).cpu().numpy()); idxs.append(besti.cpu().numpy())
    return np.concatenate(outs).astype('float32'), np.concatenate(idxs).astype('int64')

def metrics(kd,ud,q,tau):
    k=kd<=tau; u=ud<=tau; ks=[]; us=[]; ts=[]
    for ai in range(len(kd)//q):
        st,en=ai*q,(ai+1)*q
        ak=float(k[st:en].mean()*100); au=float((~u[st:en]).mean()*100)
        ks.append(ak); us.append(au); ts.append((ak+au)/2)
    return {'tau':float(tau),'total':round((float(np.mean(ks))+float(np.mean(us)))/2,6),'known':round(float(np.mean(ks)),6),'unknown':round(float(np.mean(us)),6),'origK':round(ks[0],6),'origU':round(us[0],6),'minAtk':round(float(min(ts)),6),'minK':round(float(min(ks)),6),'minU':round(float(min(us)),6)}

def choose_tau(kd,ud,q, prefer_origu=True):
    vals=np.unique(np.concatenate([kd,ud]))
    vals=vals[np.isfinite(vals)]
    if vals.size == 0:
        return {'tau':0.0,'total':0.0,'known':0.0,'unknown':0.0,'origK':0.0,'origU':0.0,'minAtk':0.0,'minK':0.0,'minU':0.0}
    if vals.size>12000: vals=np.unique(np.quantile(vals,np.linspace(0,1,12000)))
    best=None
    for tau in vals:
        m=metrics(kd,ud,q,float(tau))
        # Hard guard: original known must stay 100 if possible; push OrigU high, keep both sides alive.
        score=0.0
        score += m['total']*8 + min(m['known'],m['unknown'])*16 - abs(m['known']-m['unknown'])*12
        score += m['origK']*15 + m['origU']*22
        score += m['minK']*1.5 + m['minU']*1.2
        score -= max(0,80-m['known'])*80 + max(0,80-m['unknown'])*80
        score -= max(0,40-m['minK'])*25 + max(0,40-m['minU'])*25
        if m['origK']<100: score -= (100-m['origK'])*500
        if best is None or score>best[0]: best=(score,m)
    return best[1]

def better(a,b):
    if b is None: return True
    # Balanced accept rule: do not accept one-sided known/unknown results.
    if a['origK'] < 100: return False
    if a['known'] < 80 or a['unknown'] < 80 or a['minK'] < 35 or a['minU'] < 40: return False
    if abs(a['known'] - a['unknown']) > 5.0: return False
    # Prefer balanced useful movement, then OrigU, then total.
    ka=(min(a['known'],a['unknown']), -abs(a['known']-a['unknown']), a['origU'], a['total'], a['minK'], a['minU'])
    kb=(min(b['known'],b['unknown']), -abs(b['known']-b['unknown']), b['origU'], b['total'], b['minK'], b['minU'])
    return ka > kb
def save_ckpt(ds, folded_ckpt, scale, row, epoch):
    ck=torch.load(folded_ckpt,map_location='cpu')
    sd=ck['model_state_dict']
    sd['metric_scale']=torch.tensor(scale.astype('float32'))
    ck['model_state_dict']=sd
    ck['global_tau']=row['tau']
    ck['metrics_q100_same_bank']=dict(row, dataset=ds, epoch=epoch, training='metric_layer_pair_hardmine_train20')
    ck['training_note']='Only final diagonal metric_scale was updated using pair/hard nearest-bank losses; backbone frozen.'
    out=SAVE_ROOT/ds/f'{ds}-metric-layer-pair-hardmine-epoch{epoch:02d}.pt'
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(ck,out)
    return out

def run(ds, epochs=20, lr=0.015):
    t0=time.time(); OUT.mkdir(parents=True,exist_ok=True)
    folded=CKROOT/ds/f'{ds}-fresh-bank-metric-layer-best.pt'
    ck=torch.load(folded,map_location='cpu')
    source=Path(ck['source_checkpoint'])
    init_scale=ck['model_state_dict']['metric_scale'].detach().cpu().numpy().astype('float32')
    oldq=dict(M.Q_MAX); olds={k:list(v) for k,v in M.SPLIT_SEEDS.items()}
    M.Q_MAX[ds]=100; M.SPLIT_SEEDS[ds]=[20260529]
    try:
        bank,splits=M.collect_splits(ds,source)
    finally:
        M.Q_MAX.update(oldq); M.SPLIT_SEEDS.update(olds)
    s=splits[0]; q=int(s['q'])
    kq=filt(s['known_query'],q); uq=filt(s['unknown_query'],q)
    bank=bank.astype('float32')
    scale=init_scale.copy()
    kd,ki=metric_np(bank,kq,scale); ud,ui=metric_np(bank,uq,scale)
    row=choose_tau(kd,ud,q); row['epoch']=0
    best=row; best_scale=scale.copy(); best_path=save_ckpt(ds,folded,best_scale,best,0)
    print('[{}] BASE total={} known={} unknown={} orig={}/{} min={}/{} tau={:.10g}'.format(ds,row['total'],row['known'],row['unknown'],row['origK'],row['origU'],row['minK'],row['minU'],row['tau']), flush=True)
    # Tensors for training against current nearest assignments.
    bank_t=torch.from_numpy(bank).float().to(DEVICE)
    kq_t=torch.from_numpy(kq).float().to(DEVICE)
    uq_t=torch.from_numpy(uq).float().to(DEVICE)
    orig_scale=torch.from_numpy(init_scale).float().to(DEVICE)
    log_scale=torch.nn.Parameter(torch.log(torch.from_numpy(scale).float().to(DEVICE).clamp_min(1e-6)))
    opt=torch.optim.AdamW([log_scale], lr=lr, weight_decay=1e-4)
    history=[dict(best, checkpoint=str(best_path))]
    for ep in range(1,epochs+1):
        # refresh hard nearest ids using current scale
        cur_scale=torch.exp(log_scale).detach().cpu().numpy().astype('float32')
        kd,ki=metric_np(bank,kq,cur_scale); ud,ui=metric_np(bank,uq,cur_scale)
        tau=choose_tau(kd,ud,q)['tau']
        ki_t=torch.from_numpy(ki).long().to(DEVICE); ui_t=torch.from_numpy(ui).long().to(DEVICE)
        for step in range(80):
            sc=torch.exp(log_scale).clamp(1e-4,1e4)
            # sample hard known above median and hard unknown below tau/near bank
            d2k=(((kq_t-bank_t[ki_t])*sc)**2).sum(1).clamp_min(0)
            d2u=(((uq_t-bank_t[ui_t])*sc)**2).sum(1).clamp_min(0)
            tau2 = float(tau) * float(tau)
            known_loss=F.relu(d2k - tau2*0.85).pow(2).mean()
            unk_loss=F.relu(tau2*1.18 - d2u).pow(2).mean()
            # Extra original unknown push, but do not let scale drift wildly.
            orig_u=d2u[:q]
            orig_u_loss=F.relu(tau2*1.25 - orig_u).pow(2).mean()*4.0
            drift=F.smooth_l1_loss(torch.log(sc), torch.log(orig_scale.clamp_min(1e-6))) * 0.015
            var_guard=F.relu(0.15 - torch.std(sc/orig_scale.clamp_min(1e-6))).pow(2)*0.01
            loss=known_loss*2.0 + unk_loss*2.0 + orig_u_loss + drift + var_guard
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([log_scale], 0.15); opt.step()
            with torch.no_grad():
                lo = torch.log(orig_scale.clamp_min(1e-6)) - 0.18
                hi = torch.log(orig_scale.clamp_min(1e-6)) + 0.18
                log_scale.data.clamp_(lo, hi)
                log_scale.data = torch.where(torch.isfinite(log_scale.data), log_scale.data, torch.log(orig_scale.clamp_min(1e-6)))
                log_scale.data.clamp_(lo, hi)
        cur_scale=torch.exp(log_scale).detach().cpu().numpy().astype('float32')
        kd,ki=metric_np(bank,kq,cur_scale); ud,ui=metric_np(bank,uq,cur_scale)
        row=choose_tau(kd,ud,q); row['epoch']=ep
        print('[{}] epoch {}/{} total={} known={} unknown={} orig={}/{} min={}/{} tau={:.10g}'.format(ds,ep,epochs,row['total'],row['known'],row['unknown'],row['origK'],row['origU'],row['minK'],row['minU'],row['tau']), flush=True)
        if better(row,best):
            best=row; best_scale=cur_scale.copy(); best_path=save_ckpt(ds,folded,best_scale,best,ep)
            print(f'[{ds}] ACCEPT epoch={ep} checkpoint={best_path}', flush=True)
        history.append(dict(row))
    result={'dataset':ds,'source_folded_checkpoint':str(folded),'best_checkpoint':str(best_path),'baseline':history[0],'best':best,'history':history,'elapsed_sec':round(time.time()-t0,2)}
    out=OUT/ds/f'{ds}_metric_layer_pair_hardmine_train20.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('[DONE]', json.dumps({'dataset':ds,'best':best,'checkpoint':str(best_path)},indent=2), flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['cifar','flickr','ucid','amazon']); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--lr',type=float,default=0.015)
    args=ap.parse_args(); run(args.dataset,args.epochs,args.lr)
