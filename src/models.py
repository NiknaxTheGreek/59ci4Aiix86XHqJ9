"""MonReader reproducible models methods."""
from .common import *
from .data import make_loader


class ScratchCNN(nn.Module):
    def __init__(self,depth=3,starting_filters=16,dropout=.2):
        super().__init__(); layers=[];nin=3;nout=starting_filters
        for _ in range(int(depth)):
            # The cap prevents exponentially expanding channels from requiring implausible VRAM.
            nout=min(int(nout),256)
            layers.extend([nn.Conv2d(nin,nout,3,padding=1),nn.BatchNorm2d(nout),nn.ReLU(inplace=True),nn.MaxPool2d(2)])
            nin=nout;nout*=2
        self.features=nn.Sequential(*layers)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.head=nn.Sequential(nn.Dropout(float(dropout)),nn.Linear(nin,1))
    def forward(self,x):return self.head(self.pool(self.features(x)).flatten(1)).flatten()

def get_resnet(pretrained=True):
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc=nn.Sequential(nn.Dropout(.30),nn.Linear(model.fc.in_features,1))
    return model

def build_optimizer(model,c):
    name=c['optimizer'].lower(); kw={'lr':float(c['learning_rate']),'weight_decay':float(c['weight_decay'])}
    if name=='adam':return torch.optim.Adam(model.parameters(),**kw)
    if name=='adamw':return torch.optim.AdamW(model.parameters(),**kw)
    if name=='rmsprop':return torch.optim.RMSprop(model.parameters(),**kw)
    if name=='sgd':return torch.optim.SGD(model.parameters(),momentum=.9,**kw)
    raise ValueError(name)

def predict_frame_table(model,table,settings,device,batch_size=32,gray=False,imagenet=False):
    loader=make_loader(table,settings,min(batch_size,settings['validation_batch_cap']),False,gray,imagenet)
    model.eval();prob=[]
    with torch.no_grad():
        for images,_ in loader:
            logits=model(images.to(device)).flatten();prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    result=table[['sample_id','relative_path','label','target','source_key','video_id','frame_number']].copy().reset_index(drop=True)
    result['prob_flip']=np.asarray(prob,dtype=float)
    result['prediction']=(result.prob_flip>=settings['decision_threshold']).astype(int)
    result['correct']=(result.prediction==result.target)
    return result

def performance(predictions):
    y=np.asarray(predictions.target,dtype=int);p=np.asarray(predictions.prob_flip,dtype=float)
    yp=(p>=.50).astype(int)
    d=dict(n_images=int(len(y)),accuracy=float(accuracy_score(y,yp)),
           balanced_accuracy=float(balanced_accuracy_score(y,yp)),
           macro_f1=float(f1_score(y,yp,average='macro',labels=[0,1],zero_division=0)),
           flip_precision=float(precision_score(y,yp,pos_label=1,zero_division=0)),
           flip_recall=float(recall_score(y,yp,pos_label=1,zero_division=0)))
    if len(np.unique(y))==2:
        d['roc_auc_flip']=float(roc_auc_score(y,p));d['pr_auc_flip']=float(average_precision_score(y,p))
    else:d['roc_auc_flip']=float('nan');d['pr_auc_flip']=float('nan')
    tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel()
    d.update(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
    return d

def fit_model(model,train,validation,settings,config,device,gray=False,imagenet=False,log_fn=None):
    model.to(device);optimizer=build_optimizer(model,config)
    loss_fn=nn.BCEWithLogitsLoss()
    trainloader=make_loader(train,settings,config['batch_size'],True,gray,imagenet)
    valload=make_loader(validation,settings,min(config['batch_size'],settings['validation_batch_cap']),False,gray,imagenet)
    best=-1.0;best_epoch=0;best_state=None;stale=0;hist=[]
    for epoch in range(1,settings['max_epochs']+1):
        model.train();losses=[]
        for bi,(images,targets) in enumerate(trainloader):
            logits=model(images.to(device)).flatten();loss=loss_fn(logits,targets.to(device))
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            losses.append(float(loss.item()))
            if settings.get('max_batches') is not None and bi+1>=settings['max_batches']:break
        model.eval();vprobs=[];vy=[];vloss=[]
        with torch.no_grad():
            for vi,(images,targets) in enumerate(valload):
                logits=model(images.to(device)).flatten();vloss.append(float(loss_fn(logits,targets.to(device)).item()))
                vprobs.extend(torch.sigmoid(logits).cpu().numpy().tolist());vy.extend(targets.long().numpy().tolist())
                if settings.get('max_batches') is not None and vi+1>=settings['max_batches']:break
        score=float(f1_score(vy,(np.array(vprobs)>=.50).astype(int),average='macro',labels=[0,1],zero_division=0))
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=float(np.mean(vloss)),validation_macro_f1=score)
        hist.append(row)
        if log_fn is not None:log_fn(row)
        if score>best: # strict > selects first epoch in a tie
            best=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=settings['patience']:break
    if best_state is None:raise RuntimeError('No checkpoint selected')
    model.load_state_dict(best_state)
    return model,pd.DataFrame(hist),dict(best_epoch=int(best_epoch),best_validation_macro_f1=float(best))

def fit_resnet_two_stage(model, train, validation, settings, device, resume=False):
    """Two-stage ImageNet fine-tuning; later incremental stages do not refreeze."""
    histories=[];selected=[]
    if not resume:
        for p in model.parameters(): p.requires_grad=False
        for p in model.fc.parameters(): p.requires_grad=True
        head_settings={**settings,'max_epochs':5,'patience':5}
        head_cfg=dict(optimizer='adamw',learning_rate=.001,weight_decay=.0001,batch_size=16)
        model,h,info=fit_model(model,train,validation,head_settings,head_cfg,device,imagenet=True)
        h['stage']='head_only';histories.append(h)
        selected.append((info['best_validation_macro_f1'],copy.deepcopy(model.state_dict()),'head_only'))
    for p in model.parameters():p.requires_grad=True
    cfg=dict(optimizer='adamw',learning_rate=.0001,weight_decay=.0001,batch_size=16)
    finetune_settings={**settings,'max_epochs':5 if resume else 20,'patience':3 if resume else 5}
    model,h,info=fit_model(model,train,validation,finetune_settings,cfg,device,imagenet=True)
    h['stage']='continued_update' if resume else 'full_finetune';histories.append(h)
    selected.append((info['best_validation_macro_f1'],copy.deepcopy(model.state_dict()),h['stage'].iloc[0]))
    best=max(selected,key=lambda item:item[0]);model.load_state_dict(best[1])
    return model,pd.concat(histories,ignore_index=True),dict(best_validation_macro_f1=float(best[0]),selected_stage=best[2])

def train_resnet_regime(split,regime,settings,device):
    out=Path(settings['out'])/'resnet18'/regime;out.mkdir(parents=True,exist_ok=True)
    setup(settings['seed']);model=get_resnet(settings['resnet_pretrained'])
    tr=split[split.role=='train'];val=split[split.role=='validation'];te=split[split.role=='test']
    model,hist,info=fit_resnet_two_stage(model,tr,val,settings,device)
    hist.to_csv(out/'training_history.csv',index=False)
    torch.save({'architecture':'resnet18','weights':model.cpu().state_dict(),'recipe':'two_stage_imagenet','selection':info},out/'selected_checkpoint.pt')
    model.to(device)
    vp=predict_frame_table(model,val,settings,device,16,imagenet=True)
    tp=predict_frame_table(model,te,settings,device,16,imagenet=True)
    vp.to_csv(out/'validation_predictions.csv',index=False);tp.to_csv(out/'heldout_test_predictions.csv',index=False)
    result={**{f'val_{k}':v for k,v in performance(vp).items()},**{f'test_{k}':v for k,v in performance(tp).items()},'regime':regime,'model_family':'resnet18',**info}
    write_json(out/'metrics.json',result)
    return result

def train_grayscale(split,regime,settings,device):
    out=Path(settings['out'])/'grayscale'/regime;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((Path(settings['out'])/'scratch'/regime/'selected_config.json').read_text())['config']
    setup(settings['seed'])
    model=ScratchCNN(cfg['depth'],cfg['starting_filters'],cfg['dropout'])
    tr=split[split.role=='train'];val=split[split.role=='validation'];te=split[split.role=='test']
    model,hist,info=fit_model(model,tr,val,settings,cfg,device,gray=True)
    hist.to_csv(out/'training_history.csv',index=False)
    torch.save({'architecture':'scratch_grayscale','config':cfg,'weights':model.cpu().state_dict()},out/'checkpoint.pt')
    model.to(device)
    pred=predict_frame_table(model,te,settings,device,cfg['batch_size'],gray=True)
    pred.to_csv(out/'heldout_test_predictions.csv',index=False)
    m={**performance(pred),'regime':regime,'model_family':'scratch_grayscale',**info}
    write_json(out/'metrics.json',m);return m

def load_resnet_checkpoint(settings,regime,device):
    p=Path(settings['out'])/'resnet18'/regime/'selected_checkpoint.pt'
    ckpt=torch.load(p,map_location='cpu',weights_only=False)
    model=get_resnet(pretrained=False);model.load_state_dict(ckpt['weights']);model.to(device);model.eval()
    return model
