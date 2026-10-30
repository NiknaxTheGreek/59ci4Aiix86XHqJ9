"""MonReader reproducible evaluation methods."""
from .common import *
from .models import performance


def source_bootstrap(pred,seed=42,reps=5000):
    rng=np.random.default_rng(seed);groups=[d for _,d in pred.groupby('source_key',sort=True)]
    if not groups:raise ValueError('No sources')
    scores=np.empty(reps)
    for i in range(reps):
        sampled=rng.integers(0,len(groups),len(groups));data=pd.concat([groups[k] for k in sampled],ignore_index=True)
        scores[i]=performance(data)['macro_f1']
    return dict(point_macro_f1=performance(pred)['macro_f1'],ci_lower=float(np.percentile(scores,2.5)),
                ci_upper=float(np.percentile(scores,97.5)),bootstrap_mean=float(scores.mean()),
                n_sources=len(groups),n_resamples=reps),scores

def paired_source_bootstrap(rgb,gray,seed=42,reps=5000):
    a=rgb.merge(gray[['sample_id','prob_flip']],on='sample_id',suffixes=('_rgb','_gray'),validate='one_to_one')
    if len(a)!=len(rgb) or len(a)!=len(gray):raise AssertionError('RGB/grayscale predictions are not exactly paired')
    groups=[d for _,d in a.groupby('source_key',sort=True)];rng=np.random.default_rng(seed);deltas=[]
    for _ in range(reps):
        b=pd.concat([groups[k] for k in rng.integers(0,len(groups),len(groups))])
        p_rgb=pd.DataFrame(dict(target=b.target,prob_flip=b.prob_flip_rgb))
        p_gray=pd.DataFrame(dict(target=b.target,prob_flip=b.prob_flip_gray))
        deltas.append(performance(p_gray)['macro_f1']-performance(p_rgb)['macro_f1'])
    return dict(point_delta=performance(gray)['macro_f1']-performance(rgb)['macro_f1'],
                ci_lower=float(np.percentile(deltas,2.5)),ci_upper=float(np.percentile(deltas,97.5)))

def classify_errors(pred,high_conf=.90):
    met=performance(pred);incorrect=pred[pred.target!=pred.prediction]
    confidence=np.where(incorrect.prediction==1,incorrect.prob_flip,1-incorrect.prob_flip)
    return dict(n_errors=int(len(incorrect)),error_rate=float(len(incorrect)/len(pred)),fp=met['fp'],fn=met['fn'],
                high_confidence_errors=int((confidence>=high_conf).sum()),
                high_confidence_error_share=float((confidence>=high_conf).mean()) if len(incorrect) else 0.)
