"""MonReader reproducible splits methods."""
from .common import *


def _finish(parts, regime):
    df=pd.concat(parts,ignore_index=True);df['regime']=regime
    return df.sort_values('sample_id').reset_index(drop=True)

def make_splits(manifest,seed=42):
    # O: 72/8/20 randomized *individual frames*; no source disjointness requirement.
    dev,test=train_test_split(manifest,test_size=.20,random_state=seed,stratify=manifest.label)
    train,val=train_test_split(dev,test_size=.10,random_state=seed,stratify=dev.label)
    three=[a.copy() for a in (train,val,test)]
    for a,r in zip(three,('train','validation','test')):a['role']=r
    O=_finish(three,'O')
    # S: source groups are the units assigned to train/validation/test.
    sources=manifest[['source_key','label']].drop_duplicates().reset_index(drop=True)
    dev_s,test_s=train_test_split(sources,test_size=.20,random_state=seed,stratify=sources.label)
    train_s,val_s=train_test_split(dev_s,test_size=.10,random_state=seed,stratify=dev_s.label)
    mapping={str(k):role for tab,role in [(train_s,'train'),(val_s,'validation'),(test_s,'test')] for k in tab.source_key}
    S=manifest.copy();S['role']=S.source_key.map(mapping);S['regime']='S'
    # T: chronological 80/10/10 within *every* reconstructed source.
    pieces=[]
    for source,g in manifest.groupby('source_key',sort=True):
        t=g.sort_values(['frame_number','sample_id']).copy(); n=len(t)
        if n<3:raise ValueError(f'At least three frames required for temporal source {source}')
        nv=max(1,round(.10*n));nt=max(1,round(.10*n));ntr=n-nv-nt
        if ntr<1:raise ValueError('Insufficient training frames')
        t['role']=['train']*ntr+['validation']*nv+['test']*nt
        pieces.append(t)
    T=_finish(pieces,'T')
    # ST: known sources -> early train/later val; held-out sources -> early context/later test.
    known,unseen=train_test_split(sources,test_size=.20,random_state=seed,stratify=sources.label)
    unseen_keys=set(unseen.source_key);pieces=[]
    for source,g in manifest.groupby('source_key',sort=True):
        t=g.sort_values(['frame_number','sample_id']).copy();n=len(t)
        if n<2:raise ValueError('ST requires >=2 frames per source')
        late=max(1,round(.20*n));early=n-late
        if early<1:raise ValueError('ST needs early source frames')
        if source in unseen_keys:
            t['role']=['context_unseen_early']*early+['test']*late
        else:t['role']=['train']*early+['validation']*late
        pieces.append(t)
    ST=_finish(pieces,'ST')
    result={'O':O,'S':S,'T':T,'ST':ST}
    for regime,df in result.items():validate_split(manifest,df,regime)
    return result

def validate_split(manifest, split, regime):
    original=set(manifest.sample_id)
    if len(split)!=len(manifest) or set(split.sample_id)!=original or split.sample_id.duplicated().any():
        raise AssertionError(f'{regime}: sample accounting failed')
    for r in ('train','validation','test'):
        t=split[split.role==r]
        if len(t)==0 or set(t.label)!={'flip','notflip'}:raise AssertionError(f'{regime}: {r} must include both classes')
    if regime in ('S','ST'):
        tests=set(split.loc[split.role=='test','source_key'])
        dev=set(split.loc[split.role.isin(['train','validation']),'source_key'])
        if tests&dev:raise AssertionError(f'{regime}: unseen-test source leakage')
    if regime=='S':
        if (split.groupby('source_key').role.nunique()>1).any():raise AssertionError('S: source assigned multiple roles')
    if regime=='T':
        for _,g in split.groupby('source_key'):
            tr=g.loc[g.role=='train','frame_number'];val=g.loc[g.role=='validation','frame_number'];te=g.loc[g.role=='test','frame_number']
            if not(tr.max()<val.min() and val.max()<te.min()):raise AssertionError('T chronology violation')
    if regime=='ST':
        for _,g in split.groupby('source_key'):
            if 'train' in set(g.role):
                a=g.loc[g.role=='train','frame_number'];b=g.loc[g.role=='validation','frame_number']
            else:
                a=g.loc[g.role=='context_unseen_early','frame_number'];b=g.loc[g.role=='test','frame_number']
            if not(a.max()<b.min()):raise AssertionError('ST chronology violation')

def save_splits(splits,out):
    p=Path(out)/'splits';p.mkdir(parents=True,exist_ok=True);rows=[]
    for regime,df in splits.items():
        f=p/f'{regime}_split.csv';df.to_csv(f,index=False)
        vc=df.role.value_counts();rows.append(dict(regime=regime,**{k:int(vc.get(k,0)) for k in ['train','validation','test','context_unseen_early']},sha256=file_sha256(f)))
    summary=pd.DataFrame(rows);summary.to_csv(Path(out)/'split_summary.csv',index=False)
    return summary
