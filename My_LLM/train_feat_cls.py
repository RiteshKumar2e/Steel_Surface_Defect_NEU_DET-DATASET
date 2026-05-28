"""
One-file: build feature corpus + train tiny SteelGPT for classification.
Target: ~10 min on CPU.
"""
import os, sys, json, time, math, random, hashlib, cv2
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'NEU-DET', 'IMAGES'))
CORPUS_PATH = os.path.join(SCRIPT_DIR, 'feature_corpus.txt')
VOCAB_PATH  = os.path.join(SCRIPT_DIR, 'feature_vocab.json')
CKPT_PATH   = os.path.join(SCRIPT_DIR, 'checkpoints', 'feature_cls_best.pt')
CLASSES     = ['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']

# ── Hyperparams (tiny model = fast training) ──────────────────────────────
BLOCK_SIZE = 64; BATCH_SIZE = 32; MAX_EPOCHS = 30; LR = 5e-4
N_EMBD=64; N_HEAD=2; N_LAYER=2; FF_DIM=128; DROPOUT=0.0
DEVICE = torch.device('cpu')

# ── Quantisation ──────────────────────────────────────────────────────────
_Q = {
    'edge':     ([1,5,15,25],             ['Z','VL','LO','MD','HI']),
    'dark':     ([2,12,22],               ['Z','LO','MD','HI']),
    'bright':   ([2,10,30],               ['Z','LO','MD','HI']),
    'std':      ([13,20,32],              ['VL','LO','MD','HI']),
    'mean':     ([90,115,145],            ['VL','LO','MD','HI']),
    'h_edge':   ([0.44,0.54],             ['LO','MD','HI']),
    'lap':      ([50,200,800,1500],       ['VL','LO','MD','HI','VH']),
    'energy':   ([0.35,0.52,0.65,0.72],  ['VL','LO','MD','HI','VH']),
    'gcontrast':([0.08,0.15,0.35],       ['VL','LO','MD','HI']),
}
def _q(v,th,lb,noise=0.0):
    th2=[t*(1+noise*(random.random()*2-1)) for t in th]
    for i,t in enumerate(th2):
        if v<t: return lb[i]
    return lb[-1]
def feat2tok(f,noise=0.0):
    e=_q(f['edge'],*_Q['edge'],noise); d=_q(f['dark'],*_Q['dark'],noise)
    b=_q(f['bright'],*_Q['bright'],noise); s=_q(f['std'],*_Q['std'],noise)
    m=_q(f['mean'],*_Q['mean'],noise); h=_q(f['h_edge'],*_Q['h_edge'],noise)
    l=_q(f['lap'],*_Q['lap'],noise); n=_q(f['energy'],*_Q['energy'],noise)
    g=_q(f['gcontrast'],*_Q['gcontrast'],noise)
    return f'e:{e} d:{d} b:{b} s:{s} m:{m} h:{h} l:{l} n:{n} g:{g}'

# ── Feature extraction ─────────────────────────────────────────────────────
SKIMAGE_OK=True
try: from skimage.feature import graycomatrix,graycoprops
except: SKIMAGE_OK=False

def extract(img_path):
    img=cv2.imread(img_path)
    if img is None: return None
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,50,150)
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    h_e=float(np.sum(np.abs(gy))); tot=h_e+float(np.sum(np.abs(gx)))+1e-5
    lap=float(np.var(cv2.Laplacian(gray,cv2.CV_64F)))
    gen,gcont=0.5,0.2
    if SKIMAGE_OK:
        try:
            g2=(gray//32).astype(np.uint8)
            glcm=graycomatrix(g2,[1],[0,np.pi/4,np.pi/2,3*np.pi/4],levels=8,symmetric=True,normed=True)
            gen=float(np.mean(graycoprops(glcm,'energy'))); gcont=float(np.mean(graycoprops(glcm,'contrast')))
        except: pass
    return {'edge':float(np.sum(edges>0)/edges.size*100),
            'dark':float(np.sum(gray<80)/gray.size*100),
            'bright':float(np.sum(gray>180)/gray.size*100),
            'std':float(np.std(gray)),'mean':float(np.mean(gray)),
            'h_edge':h_e/tot,'lap':lap,'energy':gen,'gcontrast':gcont}

# ── Build corpus ──────────────────────────────────────────────────────────
def build_corpus(n_aug=5):
    if os.path.exists(CORPUS_PATH):
        print(f'Corpus exists ({os.path.getsize(CORPUS_PATH)//1024}KB), skipping rebuild.')
        return open(CORPUS_PATH,encoding='utf-8').read()
    random.seed(42); entries=[]
    for cls in CLASSES:
        d=os.path.join(IMAGES_DIR,cls)
        if not os.path.isdir(d): print(f'  SKIP {d}'); continue
        fns=sorted(f for f in os.listdir(d) if f.lower().endswith(('.jpg','.png','.bmp')))
        for fn in fns:
            feats=extract(os.path.join(d,fn))
            if feats is None: continue
            entries.append(f'[C] {feat2tok(feats,0.0)} -> {cls}')
            for _ in range(n_aug-1):
                entries.append(f'[C] {feat2tok(feats,0.15)} -> {cls}')
        print(f'  {cls}: {len(fns)}x{n_aug}={len(fns)*n_aug} entries')
    random.shuffle(entries)
    corpus='\n'.join(entries)+'\n'
    open(CORPUS_PATH,'w',encoding='utf-8').write(corpus)
    print(f'Corpus: {len(entries)} entries, {len(corpus)//1024}KB -> {CORPUS_PATH}')
    return corpus

# ── Tokenizer ─────────────────────────────────────────────────────────────
class CharTok:
    def __init__(self): self.stoi={}; self.itos={}
    @property
    def vocab_size(self): return len(self.stoi)
    def build(self,text):
        chars=sorted(set(text)); self.stoi={c:i for i,c in enumerate(chars)}; self.itos={i:c for c,i in self.stoi.items()}
        json.dump({'stoi':self.stoi,'itos':{str(k):v for k,v in self.itos.items()}},open(VOCAB_PATH,'w'))
        print(f'Vocab built: {self.vocab_size} chars -> {VOCAB_PATH}')
    def load(self):
        d=json.load(open(VOCAB_PATH)); self.stoi=d['stoi']; self.itos={int(k):v for k,v in d['itos'].items()}
        print(f'Vocab loaded: {self.vocab_size} chars')
    def encode(self,s): return [self.stoi.get(c,0) for c in s]
    def decode(self,ids): return ''.join(self.itos.get(i,'?') for i in ids)

# ── Dataset ───────────────────────────────────────────────────────────────
class FeatDS(Dataset):
    def __init__(self,ids,bs,stride=None):
        self.d=torch.tensor(ids,dtype=torch.long); self.bs=bs
        st=stride or bs//2; self.n=max(1,(len(self.d)-bs-1)//st); self.st=st
    def __len__(self): return self.n
    def __getitem__(self,i):
        s=i*self.st; return self.d[s:s+self.bs],self.d[s+1:s+self.bs+1]

# ── Tiny GPT ──────────────────────────────────────────────────────────────
class CausalSA(nn.Module):
    def __init__(self,ed,nh,bs,do):
        super().__init__(); self.nh=nh; self.hd=ed//nh
        self.qkv=nn.Linear(ed,3*ed,bias=False); self.proj=nn.Linear(ed,ed)
        self.do=nn.Dropout(do); self.register_buffer('mask',torch.tril(torch.ones(bs,bs)).view(1,1,bs,bs))
    def forward(self,x):
        B,T,C=x.shape; q,k,v=self.qkv(x).split(C,dim=-1)
        q=q.view(B,T,self.nh,self.hd).transpose(1,2); k=k.view(B,T,self.nh,self.hd).transpose(1,2); v=v.view(B,T,self.nh,self.hd).transpose(1,2)
        w=(q@k.transpose(-2,-1))/math.sqrt(self.hd); w=w.masked_fill(self.mask[:,:,:T,:T]==0,float('-inf')); w=F.softmax(w,-1); w=self.do(w)
        return self.proj((w@v).transpose(1,2).contiguous().view(B,T,C))
class Block(nn.Module):
    def __init__(self,ed,nh,bs,ffd,do):
        super().__init__(); self.ln1=nn.LayerNorm(ed); self.sa=CausalSA(ed,nh,bs,do)
        self.ln2=nn.LayerNorm(ed); self.ff=nn.Sequential(nn.Linear(ed,ffd),nn.GELU(),nn.Linear(ffd,ed),nn.Dropout(do))
    def forward(self,x): return x+self.sa(self.ln1(x))+self.ff(self.ln2(x+self.sa(self.ln1(x))))  # simplified
class TinyGPT(nn.Module):
    def __init__(self,vs,bs,ed,nh,nl,ffd,do):
        super().__init__()
        self.tok=nn.Embedding(vs,ed); self.pos=nn.Embedding(bs,ed); self.drop=nn.Dropout(do)
        self.blocks=nn.Sequential(*[Block(ed,nh,bs,ffd,do) for _ in range(nl)])
        self.ln=nn.LayerNorm(ed); self.head=nn.Linear(ed,vs,bias=False)
    def forward(self,idx,targets=None):
        B,T=idx.shape; pos=torch.arange(T,device=idx.device)
        x=self.drop(self.tok(idx)+self.pos(pos)); x=self.blocks(x); logits=self.head(self.ln(x))
        loss=None
        if targets is not None: loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss
    def count_params(self): return sum(p.numel() for p in self.parameters())

# ── Train ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model,loader):
    model.eval(); tot,n=0.0,0
    for x,y in loader:
        _,loss=model(x,y); tot+=loss.item(); n+=1
        if n>=50: break
    model.train(); return tot/max(n,1)

def train():
    print(f'Device: {DEVICE}')
    corpus=build_corpus(n_aug=5)
    tok=CharTok()
    if os.path.exists(VOCAB_PATH): tok.load()
    else: tok.build(corpus)
    ids=tok.encode(corpus); split=int(0.9*len(ids))
    train_ds=FeatDS(ids[:split],BLOCK_SIZE); val_ds=FeatDS(ids[split:],BLOCK_SIZE)
    print(f'Train: {split:,} tokens | {len(train_ds)} samples | Val: {len(ids)-split:,} tokens')
    train_dl=DataLoader(train_ds,BATCH_SIZE,shuffle=True,drop_last=True)
    val_dl=DataLoader(val_ds,BATCH_SIZE,shuffle=False,drop_last=True)
    model=TinyGPT(tok.vocab_size,BLOCK_SIZE,N_EMBD,N_HEAD,N_LAYER,FF_DIM,DROPOUT).to(DEVICE)
    print(f'TinyGPT params: {model.count_params():,}')
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=MAX_EPOCHS)
    os.makedirs(os.path.dirname(CKPT_PATH),exist_ok=True)
    best=float('inf')
    for ep in range(1,MAX_EPOCHS+1):
        model.train(); loss_sum=0; steps=0; t0=time.time()
        for x,y in train_dl:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
            _,loss=model(x,y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); loss_sum+=loss.item(); steps+=1
        sched.step(); tl=loss_sum/steps; el=time.time()-t0
        if ep%5==0 or ep==1 or ep==MAX_EPOCHS:
            vl=evaluate(model,val_dl)
            print(f'Epoch {ep:>2}/{MAX_EPOCHS} | train={tl:.4f} | val={vl:.4f} | lr={sched.get_last_lr()[0]:.5f} | {el:.1f}s')
            if vl<best:
                best=vl
                torch.save({'epoch':ep,'model_state':model.state_dict(),'val_loss':vl,
                            'block_size':BLOCK_SIZE,'embed_dim':N_EMBD,'num_heads':N_HEAD,
                            'num_layers':N_LAYER,'ff_dim':FF_DIM,'vocab_path':VOCAB_PATH,
                            'is_tiny':True},CKPT_PATH)
                print(f'  -> Best saved ({vl:.4f})')
        else:
            print(f'Epoch {ep:>2}/{MAX_EPOCHS} | train={tl:.4f} | {el:.1f}s')
    print(f'\nDone. Best val loss: {best:.4f} -> {CKPT_PATH}')

if __name__=='__main__':
    train()
