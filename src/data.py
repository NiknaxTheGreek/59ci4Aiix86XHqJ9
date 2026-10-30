"""MonReader reproducible data methods."""
from .common import *


def discover_manifest(raw):
    """Strict filename parsing: CLASS/ENVIRONMENT_FRAME.jpg. Never invent identifiers."""
    raw=Path(raw)
    if not raw.exists():raise FileNotFoundError(f'Dataset directory missing: {raw}')
    paths=sorted(p for p in raw.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp'))
    if not paths:raise ValueError(f'No readable images under {raw}. Supply the real MonReader dataset.')
    rows=[];bad=[]
    for p in paths:
        rel=p.relative_to(raw).as_posix()
        labels=[part.lower() for part in p.parts if part.lower() in CLASS_TO_Y]
        m=FRAME_RE.fullmatch(p.stem)
        if len(set(labels))!=1 or m is None:
            bad.append(rel);continue
        label=labels[-1]; video=m.group('video_id'); frame=int(m.group('frame_number'))
        # Keep the zero-padded video identifier as it appears in the filename.
        sid=hashlib.sha256(rel.encode()).hexdigest()[:16]
        rows.append(dict(sample_id=sid,relative_path=rel,path=str(p.resolve()),label=label,
                         target=CLASS_TO_Y[label],supplied_split=next((part.lower() for part in p.relative_to(raw).parts if part.lower() in ('training','testing')),None),video_id=video,environment_id=video,
                         frame_number=frame,source_key=f'{label}::{video}'))
    if bad:raise ValueError('Could not infer label/video/frame for these files (first 10): '+str(bad[:10]))
    df=pd.DataFrame(rows).sort_values('relative_path').reset_index(drop=True)
    if not set(df.supplied_split).issubset({'training','testing'}):
        raise ValueError('Expected dataset folders training/ and testing/')
    if df.sample_id.duplicated().any():raise AssertionError('Sample ID collision')
    return df

def audit_manifest(df, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # A repeated number across classes is evidence to investigate, not evidence of same physical video.
    overlap=sorted(set(df.loc[df.label=='flip','video_id'])&set(df.loc[df.label=='notflip','video_id']))
    collisions=pd.DataFrame({'shared_numeric_environment_id':overlap})
    collisions.to_csv(out/'cross_label_environment_collisions.csv',index=False)
    dup_frame=df[df.duplicated(['source_key','frame_number'],keep=False)]
    dup_frame.to_csv(out/'repeated_source_frame_numbers.csv',index=False)
    # Decoded-pixel hashes identify exact content duplicates even if filenames differ.
    digests=[];bad=[]
    for row in df.itertuples(index=False):
        try:
            with Image.open(row.path) as im:
                pixels=np.asarray(im.convert('RGB'))
                digest=hashlib.sha256(pixels.shape.__repr__().encode()+pixels.tobytes()).hexdigest()
            digests.append((row.sample_id,digest))
        except Exception as exc:bad.append((row.relative_path,str(exc)))
    if bad:raise ValueError('Unreadable images: '+str(bad[:5]))
    duplicate_pixels=pd.DataFrame(digests,columns=['sample_id','pixel_sha256'])
    duplicate_pixels=duplicate_pixels[duplicate_pixels.duplicated('pixel_sha256',keep=False)]
    duplicate_pixels.to_csv(out/'duplicate_pixels.csv',index=False)
    if len(dup_frame):warnings.warn('Repeated frame numbers detected; inspect audit before accepting source chronology.')
    # Cross-label collisions are NOT automatically leakage; human verification may be needed.
    summary=dict(n_images=len(df),class_counts=df.label.value_counts().to_dict(),
                 n_reconstructed_sources=int(df.source_key.nunique()),n_numeric_ids=int(df.video_id.nunique()),
                 cross_label_numeric_collisions=len(overlap),repeated_source_frames=len(dup_frame),
                 duplicate_content_rows=len(duplicate_pixels))
    write_json(out/'dataset_audit.json',summary)
    return summary

class FrameDataset(Dataset):
    def __init__(self,table,image_size=224,gray=False,imagenet=False):
        self.table=table.reset_index(drop=True);self.gray=gray
        t=[transforms.Resize((image_size,image_size)),transforms.ToTensor()]
        if imagenet:t.append(transforms.Normalize([.485,.456,.406],[.229,.224,.225]))
        self.transform=transforms.Compose(t)
    def __len__(self):return len(self.table)
    def __getitem__(self,i):
        row=self.table.iloc[i]
        with Image.open(row.path) as im:
            if self.gray:
                # 1-channel intensities replicated to three channels to preserve architecture and shape.
                im=im.convert('L').convert('RGB')
            else:im=im.convert('RGB')
            x=self.transform(im)
        return x,torch.tensor(float(row.target),dtype=torch.float32)

def make_loader(table,settings,batch_size=16,shuffle=False,gray=False,imagenet=False):
    return DataLoader(FrameDataset(table,settings['image_size'],gray,imagenet),batch_size=min(int(batch_size),len(table)),shuffle=shuffle,num_workers=0,pin_memory=False)

def read_archived_result_table(path):
    """Read a preserved historical CSV, including the archive's sheet-prefixed exports.

    This is a provenance viewer, NOT a recomputation of historical test metrics.
    """
    from io import StringIO
    text=Path(path).read_text(encoding='utf-8')
    if text.startswith('<PARSED TEXT FOR SHEET'):
        lines=text.splitlines()
        lines=lines[1:]
        if not lines or '>' not in lines[0]:
            raise ValueError('Malformed archived CSV sheet prefix')
        lines[0]=lines[0].split('>',1)[1]
        text='\n'.join(lines)
    return pd.read_csv(StringIO(text))
