#!/usr/bin/env python3
"""Fine-tune original adapter for exhibition; replay all other original training scenes."""
import sys,json,time,hashlib,math,shutil
from pathlib import Path
from datetime import datetime,timezone
import argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader,default_collate
from depth_anything_3.thermal.adapter import load_giant,load_adapter,adapter_state_dict
from depth_anything_3.thermal.hybrid_data import build_manifest
from depth_anything_3.thermal.selective_data import SelectiveDataset
from depth_anything_3.thermal.selective_loss import repair_loss,replay_loss
from depth_anything_3.thermal.loss import geometry_loss
from depth_anything_3.thermal.hybrid_loss import frozen_adapter_forward,confidence_preservation_loss
from depth_anything_3.utils.logger import logger

def atomic_json(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False));q.replace(p)
def atomic_save(p,v):
 q=p.with_suffix(p.suffix+'.tmp');torch.save(v,q);q.replace(p)
def move(b):return {k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items()}
def forward(m,b):
 with torch.autocast('cuda',dtype=torch.bfloat16):return m(b['images'],thermal_mask=b['thermal_mask'],ref_view_strategy='first',use_ray_pose=False,infer_gs=False)
def mean_dict(rows):return {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k],(float,int))}
def train_mode(m):
 m.train()
 # Deterministic replay to avoid needless movement away from the original adapter.
 for module in m.modules():
  if isinstance(module,torch.nn.Dropout):module.p=0.

@torch.no_grad()
def evaluate(model,normal,repair,guard):
 model.eval();out={}
 for name,dataset,is_repair in [('normal',normal,False),('exhibition',repair,True),('building_guard',guard,False)]:
  rows=[]
  for raw in DataLoader(dataset,batch_size=1,num_workers=0):
   b=move(raw);p=forward(model,b);_,metrics=(repair_loss if is_repair else geometry_loss)(p,b)
   r={'scene':b['scene'][0],'mode':b['mode'][0],**{k:float(v) for k,v in metrics.items()}}
   if is_repair:
    ratios=[]
    for i in range(p['depth'].shape[1]):
     sky=b['sky_mask'][0,i];valid=b['hard_valid'][0,i]
     scale=metrics['scale']
     ratios.append(float((p['depth'][0,i].float()[sky]*scale).median()/b['sky_lower'][0,i]))
    r['sky_to_lower_median']=float(np.median(ratios));r['sky_to_lower_min']=float(np.min(ratios))
   rows.append(r);del p,b
  out[name]={'all':mean_dict(rows),'modes':{m:mean_dict([r for r in rows if r['mode']==m]) for m in ('rgb','mixed','thermal')},'rows':rows}
 train_mode(model);return out

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
 c=json.loads(Path(a.config).read_text());out=Path(c['output']);ck=Path(c['checkpoint_dir']);out.mkdir(parents=True,exist_ok=True);ck.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.jsonl').exists():raise FileExistsError(out/'metrics.jsonl')
 torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.backends.cuda.matmul.allow_tf32=True;logger.level=1
 manifest=build_manifest(c['data_root'],c['official_split'],c['difficult_scenes']);atomic_json(out/'data_manifest.json',manifest)
 target=[s for s in manifest['train'] if s['name']=='exhibition_building'];normal=[s for s in manifest['train'] if s['name']!='exhibition_building']
 repair_train=SelectiveDataset(target,views=8,training=True,repair_split='train');repair_val=SelectiveDataset(target,views=8,training=False,repair_split='eval')
 replay=SelectiveDataset(normal,views=8,training=True);val=SelectiveDataset(manifest['eval'],views=8,training=False)
 guard=SelectiveDataset([s for s in normal if s['name']=='Building'],views=8,training=False)
 assert len(repair_train.frames[0])==119 and len(repair_val.frames[0])==16
 c.update(train_repair_pairs=119,eval_repair_pairs=16,normal_train_scenes=len(normal),original_adapter_sha256=hashlib.sha256(Path(c['initialize_adapter']).read_bytes()).hexdigest(),train_dropout_override=0)
 atomic_json(out/'run_config.json',c)
 model=load_giant(c['base_checkpoint']);load_adapter(model,c['initialize_adapter']);model.cuda();train_mode(model)
 teacher=torch.load(c['initialize_adapter'],map_location='cpu',weights_only=True);teacher_weights={n:v.cuda() for n,v in teacher['adapter'].items()};del teacher
 params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=c['learning_rate'],weight_decay=0.)
 def record(event,**data):
  r={'utc':datetime.now(timezone.utc).isoformat(),'event':event,**data}
  with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  print(json.dumps(r),flush=True)
 def save(step,metrics=None):
  state={'format_version':1,'adapter_config':model.thermal_adapter_config,'adapter':adapter_state_dict(model),'base_checkpoint':model.base_checkpoint,'step':step,'epoch':step/len(normal),'settings':c}
  p=ck/f'adapter_step_{step:04d}.pt';atomic_save(p,state)
  atomic_save(ck/'last.pt',state|{'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
  if metrics:atomic_json(out/f'eval_step_{step:04d}.json',{'step':step,**metrics})
  return p
 record('data_ready',normal_train_scenes=len(normal),normal_train_pairs=sum(s['pairs'] for s in normal),exhibition_train_pairs=119,exhibition_eval_pairs=16)
 if not a.smoke:
  baseline=evaluate(model,val,repair_val,guard);atomic_json(out/'baseline_eval.json',{'step':0,**baseline});record('baseline',normal_abs_rel=baseline['normal']['all']['abs_rel'],exhibition=baseline['exhibition']['modes'],building=baseline['building_guard']['modes'])
 else:baseline=None
 by_scene={i:[j for j,(si,_) in enumerate(replay.items) if si==i] for i in range(len(normal))}
 candidates=[];started=time.monotonic();success_streak=0
 for step in range(1,(1 if a.smoke else c['max_steps'])+1):
  optimizer.zero_grad(set_to_none=True);rate=c['learning_rate']*min(1,step/10)
  for pg in optimizer.param_groups:pg['lr']=rate
  repair_train.epoch=(step-1)//len(repair_train)
  batches=[('repair',repair_train[(step-1)%len(repair_train)])]
  for j in range(c['replay_per_step']):
   global_j=(step-1)*c['replay_per_step']+j;si=global_j%len(normal);cycle=global_j//len(normal);replay.epoch=cycle
   options=by_scene[si];idx=options[cycle%len(options)];batches.append(('replay',replay[idx]))
  sums={}
  for kind,raw in batches:
   b=move(default_collate([raw]))
   with frozen_adapter_forward(model,teacher_weights) as frozen,torch.no_grad():t=forward(frozen,b)
   p=forward(model,b)
   if kind=='repair':
    loss,m=repair_loss(p,b);loss=loss+.1*confidence_preservation_loss(p,t['depth_conf'].detach().float())
   else:loss,m=replay_loss(p,b,t)
   if not torch.isfinite(loss):raise FloatingPointError((step,kind))
   (loss/len(batches)).backward()
   sums.setdefault(kind,[]).append({k:float(v.detach()) for k,v in m.items()});del p,t,b,loss
  grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
  if step==1:
   grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters() if p.requires_grad};atomic_json(out/'first_step_gradients.json',grads)
   record('gradients',nonzero=sum(v is not None and v>0 for v in grads.values()),total=len(grads))
  optimizer.step()
  if step<=2 or step%10==0:record('train',step=step,repair=mean_dict(sums['repair']),replay=mean_dict(sums['replay']),grad_norm=float(grad),elapsed_seconds=time.monotonic()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
  if a.smoke:save(step);record('smoke_finished',step=step);return
  if step%c['eval_every']==0 or step==c['max_steps']:
   metrics=evaluate(model,val,repair_val,guard);path=save(step,metrics)
   normal_score=metrics['normal']['all']['abs_rel'];old_score=baseline['normal']['all']['abs_rel']
   order=max(r['sky_order'] for r in metrics['exhibition']['modes'].values())
   repair_score=metrics['exhibition']['all']['hard_abs_rel']
   bg=metrics['building_guard']['modes']['rgb']['abs_rel'];oldbg=baseline['building_guard']['modes']['rgb']['abs_rel']
   normal_ok=normal_score<=old_score*1.05+.001 and bg<=oldbg*1.10+.001
   order_ok=order<.08 and min(r['sky_to_lower_min'] for r in metrics['exhibition']['modes'].values())>.80
   solid_ok=repair_score<=baseline['exhibition']['all']['hard_abs_rel']*1.1
   passed=normal_ok and order_ok and solid_ok
   score=order+repair_score+10*max(0,normal_score-old_score)+5*max(0,bg-oldbg)
   row={'step':step,'path':str(path),'normal_abs_rel':normal_score,'exhibition_hard_abs_rel':repair_score,'max_sky_order_loss':order,'normal_ok':normal_ok,'order_ok':order_ok,'solid_ok':solid_ok,'passed':passed,'score':score}
   candidates.append(row);atomic_json(out/'candidates.json',candidates);record('validation',**row)
   success_streak=success_streak+1 if passed else 0
   if success_streak>=2 and step>=c['min_steps']:record('early_stop',step=step,reason='two passing validations');break
 eligible=[r for r in candidates if r['passed']]
 if not eligible:eligible=[r for r in candidates if r['normal_ok']] or candidates
 best=min(eligible,key=lambda r:r['score']);shutil.copy2(best['path'],ck/'adapter_selected.pt')
 atomic_json(out/'selected_eval.json',json.loads((out/f'eval_step_{best["step"]:04d}.json').read_text()))
 atomic_json(out/'selection.json',{'selected':best,'candidates':candidates,'all_acceptance_checks_passed':best['passed'],'sha256':hashlib.sha256((ck/'adapter_selected.pt').read_bytes()).hexdigest()})
 record('finished',selected=best,total_steps=step)
if __name__=='__main__':main()
