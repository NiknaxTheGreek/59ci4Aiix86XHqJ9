"""MonReader reproducible visualization methods."""
from .common import *
from .models import load_resnet_checkpoint


def plot_regime_examples(splits, filename, seed=42):
    """Legible representative frames sampled from the actual assigned partitions."""
    rng=random.Random(seed)
    roles={'O':['train','validation','test','train'],
           'S':['train','validation','test','test'],
           'T':['train','validation','test','test'],
           'ST':['train','validation','context_unseen_early','test']}
    fig,axes=plt.subplots(4,4,figsize=(16,15),constrained_layout=True)
    for i,regime in enumerate(REGIMES):
        frame=splits[regime]
        for j,role in enumerate(roles[regime]):
            subset=frame[frame.role==role]
            ax=axes[i,j]
            if len(subset):
                row=subset.iloc[rng.randrange(len(subset))]
                with Image.open(row.path) as im:ax.imshow(im.convert('RGB'))
                ax.set_title(f'{role.replace("_"," ")}\nVideo {row.video_id} / frame {row.frame_number}',fontsize=11,pad=14)
            else:
                ax.text(.5,.5,'No examples',ha='center',va='center',transform=ax.transAxes)
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(.8)
        axes[i,0].set_ylabel(regime,rotation=0,labelpad=32,fontweight='bold',fontsize=14,va='center')
    fig.suptitle('Evaluation regimes — representative video frames',fontsize=17,fontweight='bold')
    filename=Path(filename);filename.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(filename,dpi=145,bbox_inches='tight');plt.close(fig)
    return filename

def plot_model_comparison(results,out):
    p=pd.DataFrame(results);fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    xs=np.arange(4);width=.34
    for i,(family,label) in enumerate([('scratch','Scratch CNN'),('resnet18','Pretrained ResNet18')]):
        vals=[float(p.loc[(p.regime==r)&(p.model_family==family),'test_macro_f1'].iloc[0]) for r in REGIMES]
        bars=ax.bar(xs+(i-.5)*width,vals,width,label=label)
        for b,v in zip(bars,vals):ax.annotate(f'{v:.3f}',(b.get_x()+b.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=10)
    ax.set_xticks(xs,['O\nOriginal','S\nSource-safe','T\nTemporal','ST\nSource + temporal'])
    ax.set_ylim(0,1.11);ax.set_ylabel('Held-out Macro F1');ax.set_title('Model comparison across evaluation regimes')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,frameon=False)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=165,bbox_inches='tight');plt.close(fig)
    return out

def plot_grayscale_examples(manifest,out,seed=42):
    fig,axes=plt.subplots(2,2,figsize=(8,9))
    for i,label in enumerate(('flip','notflip')):
        row=manifest[manifest.label==label].sort_values('sample_id').iloc[0]
        with Image.open(row.path) as im:rgb=np.asarray(im.convert('RGB'));gray=np.asarray(im.convert('L'))
        axes[i,0].imshow(rgb);axes[i,1].imshow(gray,cmap='gray',vmin=0,vmax=255)
        axes[i,0].set_ylabel(label,fontweight='bold');
        for j in range(2):axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
    axes[0,0].set_title('RGB');axes[0,1].set_title('Grayscale')
    fig.suptitle('Same original frame before and after grayscale conversion')
    fig.tight_layout();out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=150);plt.close(fig)
    return out

def cam_for_image(model,image_path,device,image_size,predicted_class):
    """Binary one-logit predicted-class Grad-CAM: flip=+logit; notflip=-logit."""
    with Image.open(image_path) as im:rgb=np.asarray(im.convert('RGB').resize((image_size,image_size)))
    x=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])(Image.fromarray(rgb)).unsqueeze(0).to(device)
    activations=[]
    handle=model.layer4[-1].conv2.register_forward_hook(lambda _m,_inp,out:activations.append(out))
    with torch.enable_grad():
        out=model(x);activation=activations[0];score=out.flatten()[0]*(1 if predicted_class==1 else -1)
        gradients=torch.autograd.grad(score,activation,retain_graph=False)[0]
        weights=gradients.mean(dim=(2,3),keepdim=True)
        cam=torch.relu((weights*activation).sum(dim=1,keepdim=True))
        cam=torch.nn.functional.interpolate(cam,size=(image_size,image_size),mode='bilinear',align_corners=False)[0,0]
        heat=cam.detach().cpu().numpy()
    handle.remove()
    heat=heat-heat.min();heat=heat/(heat.max()+1e-8)
    from matplotlib import colormaps
    color=colormaps['jet'](heat)[...,:3]
    overlay=(.45*rgb.astype(float)/255+.55*color).clip(0,1)
    return overlay,heat

def select_cam_cases(pred):
    p=pred.copy();p['is_error']=p.prediction!=p.target
    p['confidence']=np.where(p.prediction==1,p.prob_flip,1-p.prob_flip)
    candidates=[('Correct notflip',p[(~p.is_error)&(p.target==0)].sort_values(['confidence','sample_id'],ascending=[False,True])),
                ('Correct flip',p[(~p.is_error)&(p.target==1)].sort_values(['confidence','sample_id'],ascending=[False,True])),
                ('High-confidence error',p[p.is_error].sort_values(['confidence','sample_id'],ascending=[False,True])),
                ('Borderline error',p[p.is_error].assign(dist=lambda x:abs(x.prob_flip-.5)).sort_values(['dist','sample_id']))]
    selected=[];used=set()
    for role,df in candidates:
        hit=next((x for x in df.itertuples(index=False) if x.sample_id not in used),None)
        if hit is None:selected.append(dict(role=role,found=False,sample_id=None))
        else:selected.append(dict(role=role,found=True,sample_id=hit.sample_id,relative_path=hit.relative_path,
                                  target=int(hit.target),prob_flip=float(hit.prob_flip),prediction=int(hit.prediction)))
        if hit is not None:used.add(hit.sample_id)
    return selected

def plot_cam_matrix(settings,device,out):
    fig,axes=plt.subplots(4,4,figsize=(12,13));registry=[]
    for i,regime in enumerate(REGIMES):
        model=load_resnet_checkpoint(settings,regime,device)
        pred=pd.read_csv(Path(settings['out'])/'resnet18'/regime/'heldout_test_predictions.csv')
        picks=select_cam_cases(pred)
        for j,case in enumerate(picks):
            ax=axes[i,j]
            if case['found']:
                path=Path(settings['raw'])/case['relative_path']
                overlay,heat=cam_for_image(model,path,device,settings['image_size'],case['prediction'])
                ax.imshow(overlay);case['cam_max']=float(heat.max())
            else:ax.text(.5,.5,'No eligible case\nin this test set',ha='center',va='center',transform=ax.transAxes)
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(case['role'],fontsize=9)
            if j==0:ax.set_ylabel(regime,rotation=0,labelpad=20,fontweight='bold')
            registry.append({'regime':regime,**case})
    fig.suptitle('ResNet18 Grad-CAM: correct and error cases',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.97]);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=150);plt.close(fig)
    pd.DataFrame(registry).to_csv(out.parent/'gradcam_16_panel_selection.csv',index=False)
    return out

def select_21_interpretability_cases(settings):
    """Choose cases from frozen TEST predictions before viewing images/heatmaps."""
    rows=[];seen=set()
    # Five cases per regime and an extra non-duplicate borderline case if available.
    for regime in REGIMES:
        p=pd.read_csv(Path(settings['out'])/'resnet18'/regime/'heldout_test_predictions.csv')
        p['error']=p.prediction.ne(p.target)
        p['confidence']=np.where(p.prediction==1,p.prob_flip,1-p.prob_flip)
        p['distance']=abs(p.prob_flip-.5)
        pools=[('correct_notflip',p[(~p.error)&(p.target==0)].sort_values(['confidence','sample_id'],ascending=[False,True])),
               ('correct_flip',p[(~p.error)&(p.target==1)].sort_values(['confidence','sample_id'],ascending=[False,True])),
               ('highest_conf_error',p[p.error].sort_values(['confidence','sample_id'],ascending=[False,True])),
               ('near_threshold_error',p[p.error].sort_values(['distance','sample_id'])),
               ('near_threshold_correct',p[~p.error].sort_values(['distance','sample_id']))]
        for role,pool in pools:
            chosen=next((r for r in pool.itertuples(index=False) if (regime,r.sample_id) not in seen),None)
            if chosen is not None:
                seen.add((regime,chosen.sample_id));rows.append(dict(regime=regime,selection_role=role,
                   sample_id=chosen.sample_id,relative_path=chosen.relative_path,target=int(chosen.target),
                   predicted=int(chosen.prediction),prob_flip=float(chosen.prob_flip)))
    p=pd.read_csv(Path(settings['out'])/'resnet18'/'O'/'heldout_test_predictions.csv')
    p['distance']=abs(p.prob_flip-.5)
    extra=next((r for r in p.sort_values(['distance','sample_id']).itertuples(index=False) if ('O',r.sample_id) not in seen),None)
    if extra is not None:
        rows.append(dict(regime='O',selection_role='additional_borderline',sample_id=extra.sample_id,
                         relative_path=extra.relative_path,target=int(extra.target),predicted=int(extra.prediction),prob_flip=float(extra.prob_flip)))
    return pd.DataFrame(rows)

def generate_activation_diagnostics(settings,device,filename):
    """Save one mean-activation map for each of four ResNet blocks per selected frame."""
    registry=select_21_interpretability_cases(settings)
    output=Path(filename);output.parent.mkdir(parents=True,exist_ok=True)
    registry.to_csv(output.parent/'interpretability_case_registry.csv',index=False)
    diag=[];preview=[]
    for regime in REGIMES:
        model=load_resnet_checkpoint(settings,regime,device)
        selected=registry[registry.regime==regime]

        for row in selected.itertuples(index=False):
            with Image.open(Path(settings['raw'])/row.relative_path) as im:im=im.convert('RGB').resize((settings['image_size'],settings['image_size']))
            x=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])(im).unsqueeze(0).to(device)
            captures={};handles=[]
            for idx in (1,2,3,4):
                layer=getattr(model,f'layer{idx}')
                handles.append(layer.register_forward_hook(lambda m,inp,out,i=idx:captures.__setitem__(i,out.detach().cpu())))
            with torch.no_grad():model(x)
            for h in handles:h.remove()
            means=[]
            for idx in (1,2,3,4):
                fmap=captures[idx][0].abs().mean(0).numpy();fmap=fmap-fmap.min();fmap/=fmap.max()+1e-8
                means.append(fmap)
                map_path=output.parent/'activation_maps'/f'{regime}_{row.sample_id}_layer{idx}.png'
                map_path.parent.mkdir(parents=True,exist_ok=True)
                plt.imsave(map_path,fmap,cmap='viridis')
                diag.append(dict(regime=regime,sample_id=row.sample_id,layer=idx,
                                 mean_abs_activation=float(captures[idx].abs().mean()),map_path=str(map_path)))
            if len(preview)<4:preview.append((regime,means))
    pd.DataFrame(diag).to_csv(output.parent/'activation_summary.csv',index=False)
    fig,axes=plt.subplots(max(1,len(preview)),4,figsize=(11,2.6*max(1,len(preview))))
    axes=np.asarray(axes).reshape(max(1,len(preview)),4)
    for ri,(regime,maps) in enumerate(preview):
        for col,activation in enumerate(maps):
            axes[ri,col].imshow(activation,cmap='viridis');axes[ri,col].axis('off')
            if ri==0:axes[ri,col].set_title(f'ResNet layer {col+1}')
            if col==0:axes[ri,col].text(-.12,.5,regime,transform=axes[ri,col].transAxes,va='center',fontweight='bold')
    fig.suptitle('ResNet18 Intermediate Activation Progression',y=1.01)
    fig.tight_layout();fig.savefig(output,dpi=140,bbox_inches='tight');plt.close(fig)
    return registry,pd.DataFrame(diag)
