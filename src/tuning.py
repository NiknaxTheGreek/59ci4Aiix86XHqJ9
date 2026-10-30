"""MonReader reproducible tuning methods."""
from .common import *
from .models import ScratchCNN, fit_model, performance, predict_frame_table


def resolve_config(c):
    x=dict(c);x['weight_decay']=0. if x.get('weight_decay_mode')=='zero' else float(x['positive_weight_decay'])
    return x

def tune_one_regime_wandb(split,regime,settings,device):
    """Optional true W&B Bayesian sweep. Requires the researcher's W&B account.

    This is intentionally a NEW sweep; it neither retrieves nor modifies the
    author's earlier W&B experiments. Test data are evaluated only after the
    selected validation winner has been frozen.
    """
    try:import wandb
    except ImportError as exc:raise ImportError('Install wandb and authenticate before enabling W&B Bayesian sweeps') from exc

    out=Path(settings['out'])/'scratch'/regime;out.mkdir(parents=True,exist_ok=True)
    tr=split[split.role=='train'];val=split[split.role=='validation'];te=split[split.role=='test']
    sweep_cfg=dict(method='bayes',metric=dict(name='validation_macro_f1',goal='maximize'),
        parameters=dict(optimizer=dict(values=['adam','adamw','sgd','rmsprop']),
            learning_rate=dict(distribution='log_uniform_values',min=1e-5,max=1e-2),
            batch_size=dict(distribution='int_uniform',min=8,max=256),
            weight_decay_mode=dict(values=['zero','positive']),
            positive_weight_decay=dict(distribution='log_uniform_values',min=1e-6,max=1e-2),
            dropout=dict(distribution='uniform',min=0.,max=.5),
            depth=dict(distribution='int_uniform',min=1,max=7),
            starting_filters=dict(distribution='int_uniform',min=8,max=64)))
    project=settings['wandb_project']
    sweep_id=wandb.sweep(sweep_cfg,project=project,entity=settings.get('wandb_entity'))
    rows=[]; state={'attempts':0,'qualifying':0,'best':-1.,'model':None,'config':None,'info':None,'history':None}
    def do_train():
        with wandb.init(project=project,entity=settings.get('wandb_entity'),group=regime,job_type='bayesian_trial') as run:
            state['attempts']+=1
            c=resolve_config(dict(run.config))
            setup(settings['seed']+state['attempts'])
            try:
                model=ScratchCNN(c['depth'],c['starting_filters'],c['dropout'])
                model,hist,info=fit_model(model,tr,val,settings,c,device,log_fn=lambda rec:wandb.log(rec))
                v=info['best_validation_macro_f1']
                qualified=bool(np.isfinite(v) and v>0)
                if qualified:state['qualifying']+=1
                rows.append(dict(attempt=state['attempts'],run_id=run.id,qualifies=qualified,validation_macro_f1=v,**c,**info))
                wandb.log(dict(best_validation_macro_f1=v,qualified=int(qualified)))
                hist.to_csv(out/f'trial_{state["attempts"]:03d}_epoch_history.csv',index=False)
                if v>state['best']:
                    state.update(best=v,model=copy.deepcopy(model).cpu(),config=c,info=info,history=hist)
            except RuntimeError as exc:
                rows.append(dict(attempt=state['attempts'],run_id=run.id,status='failed',reason=str(exc)[:300]))
                if torch.cuda.is_available():torch.cuda.empty_cache()
                # OOM/failed attempts are logged but do not count towards 30 qualifying trials.
                return
    while state['qualifying']<settings['qualifying_trials'] and state['attempts']<settings['qualifying_trials']*5:
        wandb.agent(sweep_id,function=do_train,count=1,project=project,entity=settings.get('wandb_entity'))
    if state['qualifying']<settings['qualifying_trials'] or state['model'] is None:
        raise RuntimeError(f'W&B sweep has insufficient qualifying trials for {regime}')
    pd.DataFrame(rows).to_csv(out/'bayesian_trials.csv',index=False)
    state['history'].to_csv(out/'selected_training_history.csv',index=False)
    torch.save(dict(architecture='scratch',config=state['config'],weights=state['model'].state_dict(),
                    selection_metric='validation_macro_f1',best_epoch=state['info']['best_epoch']),out/'selected_checkpoint.pt')
    write_json(out/'selected_config.json',dict(config=state['config'],**state['info'],
                                                qualifying=state['qualifying'],attempts=state['attempts']))
    model=state['model'].to(device)
    valpred=predict_frame_table(model,val,settings,device,state['config']['batch_size'])
    testpred=predict_frame_table(model,te,settings,device,state['config']['batch_size'])
    valpred.to_csv(out/'validation_predictions.csv',index=False)
    testpred.to_csv(out/'heldout_test_predictions.csv',index=False)
    met={**{f'val_{k}':v for k,v in performance(valpred).items()},
         **{f'test_{k}':v for k,v in performance(testpred).items()},'regime':regime,'model_family':'scratch'}
    write_json(out/'metrics.json',met)
    return met
