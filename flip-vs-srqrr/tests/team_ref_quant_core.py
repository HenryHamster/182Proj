"""Stage 39 reusable quantizer, derived from the Stage 36 reference.
Each QuantLinear or QuantConv stores the same four learned levels per group for both methods.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def decode(k,a,b):
    aa=a[:,None]; bb=b[:,None]
    return torch.where(k==0,-aa,torch.where(k==1,-bb,torch.where(k==2,bb,aa)))

def symbols_from_shadow(w,a,b):
    th=((a+b)*.5)[:,None]
    return ((w>=-th).to(torch.uint8)+(w>=0).to(torch.uint8)+(w>=th).to(torch.uint8))

def bounds_from_initial(a0, policy='relative', ratio=4.):
    if policy=='legacy':
        return (torch.full_like(a0,.006),torch.full_like(a0,.002),torch.full_like(a0,.004),torch.full_like(a0,2.))
    if policy!='relative':raise ValueError(policy)
    return (a0/ratio,torch.zeros_like(a0),a0*1e-6,a0*ratio)

def fit_project(ua,ub,ha,hb,bounds):
    """Weighted exact projection to amin<=a<=amax, bmin<=b<=a-gap.
    Four clipped boundary candidates plus feasible interior, vectorized per row.
    """
    amin,bmin,gap,amax=bounds
    amin=torch.maximum(amin,bmin+gap)
    ha=ha.clamp_min(1e-20); hb=hb.clamp_min(1e-20)
    valid=(ua>=amin)&(ua<=amax)&(ub>=bmin)&(ub<=ua-gap)
    a1=amin; b1=torch.maximum(torch.minimum(ub,amin-gap),bmin)
    a2=amax; b2=torch.maximum(torch.minimum(ub,amax-gap),bmin)
    b3=bmin; a3=torch.maximum(torch.minimum(ua,amax),amin)
    b4=(ha*(ua-gap)+hb*ub)/(ha+hb)
    b4=b4.clamp(min=torch.maximum(bmin,amin-gap),max=amax-gap)
    a4=b4+gap
    aa=torch.stack((ua,a1,a2,a3,a4),dim=0)
    bb=torch.stack((ub,b1,b2,b3,b4),dim=0)
    costs=ha.unsqueeze(0)*(aa-ua).square()+hb.unsqueeze(0)*(bb-ub).square()
    costs[0]=torch.where(valid,costs[0],torch.inf)
    choice=costs.argmin(0,keepdim=True)
    aout=aa.gather(0,choice).squeeze(0)
    bout=bb.gather(0,choice).squeeze(0)
    return aout,bout

class QuantLinear(nn.Module):
    """Four-level INT2 weight matrix with shared symmetric codebook per G weights.

    W_ij belongs to {-a_g,-b_g,b_g,a_g}, with a_g>b_g>=0.
    Shadow-STE stores a full FP32 shadow and *learnable* (a,b) per group;
    direct stores symbols+INT8 residual and *no* full precision master weights.
    """
    def __init__(self,d_in,d_out,layer_id,method,policy='relative',init_mode='fan_in',ratio=4.,seed=123,group_size=32):
        super().__init__()
        self.method=method
        self.layer_id=int(layer_id)
        self.in_features=d_in;self.out_features=d_out;self.group=group_size
        assert d_in % group_size == 0, (d_in,group_size)
        n_groups=(d_in*d_out)//group_size
        # Canonical, method-independent initialization. Use a local NumPy RNG,
        # NOT global Torch RNG or any optimizer-specific code path. This generates
        # precisely the same symbols for both GPEF and AdamW-STE on CPU.
        rng=np.random.default_rng(int(seed)+int(layer_id)*19013)
        symbols=rng.integers(0,4,size=(n_groups,group_size),dtype=np.uint8)
        k=torch.from_numpy(symbols.copy())
        if init_mode=='fan_in': sigma=1/math.sqrt(d_in)
        elif init_mode=='xavier':sigma=math.sqrt(2/(d_in+d_out))
        elif init_mode=='legacy':sigma=None
        else:raise ValueError(init_mode)
        if sigma is not None:
            p=((k==0)|(k==3)).to(torch.float32).mean(1)
            a=torch.full((n_groups,),sigma,dtype=torch.float32)/torch.sqrt(p+(1-p)/9.)
        else: a=torch.full((n_groups,),min(.7,1.2/math.sqrt(d_in)),dtype=torch.float32)
        b=a/3.
        self.bound_policy=policy; self.bound_ratio=float(ratio)
        self.register_buffer('span',a.detach().clone().clamp_min(1e-12))
        if method in ('muon','adamw_ste'):
            # Initial shadow equals the initial deployed weight, exactly.
            self.shadow=nn.Parameter(decode(k,a,b).reshape(d_out,d_in))
            self.a=nn.Parameter(a.clone());self.b=nn.Parameter(b.clone())
        else:
            self.register_buffer('k',k)
            self.register_buffer('a',a.clone());self.register_buffer('b',b.clone())
            self.register_buffer('error',torch.zeros_like(k,dtype=torch.int8))
        self._w=None;self._v=None

    def quantized(self):
        if self.method in ('muon','adamw_ste'):
            k=symbols_from_shadow(self.shadow.reshape(-1,self.group),self.a,self.b)
            q=decode(k,self.a,self.b).reshape(self.out_features,self.in_features)
            # STE for shadow, exact selected-codeword derivative for a,b.
            return self.shadow + (q-self.shadow).detach() + (q-q.detach())
        q=decode(self.k,self.a,self.b).reshape(self.out_features,self.in_features)
        if self.training and torch.is_grad_enabled():
            self._w=q.detach().requires_grad_(True)
            return self._w
        return q

    def forward(self,x,activation_weighting=True):
        w=self.quantized()
        if self.method in ('joint8','refined','joint8_matrix') and self.training and torch.is_grad_enabled() and activation_weighting:
            with torch.no_grad():self._v=x.detach().float().square().mean(dim=tuple(range(x.ndim-1)))
        else:self._v=None
        return F.linear(x,w)

    @torch.no_grad()
    def clamp_codebook(self):
        aa,bb=fit_project(self.a,self.b,torch.ones_like(self.a),torch.ones_like(self.b),self.bounds())
        self.a.copy_(aa);self.b.copy_(bb)

    def bounds(self):
        # Regenerate the (untuned) feasibility bounds without four extra persistent FP32 buffers.
        return bounds_from_initial(self.span,self.bound_policy,self.bound_ratio)
    def current_k(self):
        if self.method in ('muon','adamw_ste'):
            return symbols_from_shadow(self.shadow.reshape(-1,self.group),self.a,self.b)
        return self.k

class GroupDiscrete:
    def __init__(self, model, lr=.03, kind='ef', error_bits=8, passes=2,
                 total_steps=300, floor=.2, seed=47, diag_every=10, cpu_backend="auto"):
        assert kind in ('ef','ef_resample','stoch','round')
        assert error_bits in (0,4,8)
        self.layers=model.qlayers()
        self.lr=float(lr);self.kind=kind;self.error_bits=int(error_bits)
        self.passes=int(passes);self.t=0;self.total_steps=total_steps;self.floor=floor
        self.rng=torch.Generator(device='cpu').manual_seed(seed)
        self.diag_every=int(diag_every)
        self.last_diagnostics={};self.last_flip=0.
        self.cpu_backend = cpu_backend
        if cpu_backend not in ('auto','numba','torch'):
            raise ValueError('cpu_backend must be auto, numba, or torch')
        if cpu_backend != 'torch':
            try:
                from cpu_kernel import HAS_NUMBA, fused_update
                if not HAS_NUMBA and cpu_backend=='numba':
                    raise ImportError('Numba CPU kernel requested but numba is unavailable')
                self._fused = fused_update if HAS_NUMBA else None
            except ImportError:
                if cpu_backend == 'numba':raise
                self._fused=None
        else:self._fused=None

    @staticmethod
    def k_nearest(target,a,b):
        mid=(a+b)[:,None]/2
        return ((target>=-mid).to(torch.uint8)+(target>=0).to(torch.uint8)+(target>=mid).to(torch.uint8))

    @torch.no_grad()
    def k_stochastic(self,target,a,b):
        v=target.clamp(min=-a[:,None],max=a[:,None]);aa=a[:,None];bb=b[:,None]
        # q_low/q_up are the two adjacent members among (-a,-b,b,a)
        lo=torch.where(v< -bb,-aa,torch.where(v<bb,-bb,bb))
        hi=torch.where(v< -bb,-bb,torch.where(v<bb,bb,aa))
        prob=((v-lo)/(hi-lo).clamp_min(1e-10)).clamp(0.,1.)
        # This uses the global generator seeded by each experiment, with no extra state per weight.
        move=torch.rand(prob.shape,device=prob.device)<prob
        idx=torch.where(v< -bb,torch.zeros_like(v,dtype=torch.uint8),
                torch.where(v<bb,torch.ones_like(v,dtype=torch.uint8),torch.full_like(v,2,dtype=torch.uint8)))
        return idx+move.to(torch.uint8)

    @staticmethod
    @torch.no_grad()
    def fit_for_symbols(target,k,a,b,bounds,regularizer=0.0):
        # No extra scale optimizer or state; direct conditional least squares.
        sa=(k==3).float()-(k==0).float()
        sb=(k==2).float()-(k==1).float()
        na=sa.square().sum(1)+regularizer
        nb=sb.square().sum(1)+regularizer
        aa=torch.where(na>0,((sa*target).sum(1)+regularizer*a)/na.clamp_min(1e-8),a)
        bb=torch.where(nb>0,((sb*target).sum(1)+regularizer*b)/nb.clamp_min(1e-8),b)
        # If a codeword family is unused, preserve its previous magnitude rather than
        # arbitrarily setting that magnitude to zero (unidentified least squares).
        return fit_project(aa,bb,na.clamp_min(1e-8),nb.clamp_min(1e-8),bounds)

    @torch.no_grad()
    def step(self,gradient_scale=1.):
        fraction=self.t/max(1,self.total_steps-1)
        eta=self.lr*(self.floor+(1-self.floor)*.5*(1+math.cos(math.pi*fraction)))
        self.t+=1
        measure=self.diag_every>0 and (self.t==1 or self.t%self.diag_every==0)
        total_flip=0;total_weights=0;total_grad_dot=0.;total_change=0.;total_grad2=0.;total_error=0.;total_sat=0.
        for l in self.layers:
            if l._w is None or l._w.grad is None:continue
            if self._fused is not None and self.kind=='ef' and self.error_bits==8 and l._w.device.type=='cpu' and l.bound_policy=='relative':
                stats=self._fused(l,eta,self.passes,measure)
                if stats is not None:
                    total_flip+=stats[0];total_weights+=l.k.numel()
                    total_grad_dot+=stats[1];total_change+=stats[2]
                    total_grad2+=stats[3];total_error+=stats[4];total_sat+=stats[5]
                l._w=None;l._v=None
                continue
            g=l._w.grad.detach().reshape(-1,l.group)/gradient_scale
            # normalized per group: one scalar learning rate controls both indices and shared magnitudes
            rms=torch.sqrt(g.float().square().mean(1,keepdim=True)+1e-12)
            u=-(eta*l.a[:,None])*(g/rms).clamp(-4,4)
            old=decode(l.k,l.a,l.b)
            if self.kind in ('ef','ef_resample'):
                if self.error_bits==0: prev=torch.zeros_like(u)
                else:prev=l.error.float() * (l.span[:,None]/((1<<(self.error_bits-1))-1))
                target=old+u+prev
            else:target=old+u
            a=l.a;b=l.b; origk=l.k
            if self.kind=='stoch':
                knew=self.k_stochastic(target,a,b)
                a,b=self.fit_for_symbols(target,knew,a,b,l.bounds())
            else:
                # Coordinate descent for the real deployed Q, not the residual.
                # First fit old codewords, then select symbols, then fit again.
                for _ in range(self.passes):
                    a,b=self.fit_for_symbols(target,origk if _==0 else knew,a,b,l.bounds())
                    knew=self.k_nearest(target,a,b)
                # Final closest-codeword pass after last scale fit.
                knew=self.k_nearest(target,a,b)
            qnew=decode(knew,a,b)
            if self.kind in ('ef','ef_resample') and self.error_bits>0:
                debt=target-qnew
                levels=(1<<(self.error_bits-1))-1
                x=(debt/l.span[:,None]*levels).clamp(-levels,levels)
                if self.kind=='ef_resample':
                    # stochastic unbiased residual quantization, including sub-grid proposals
                    low=torch.floor(x);x=low+(torch.rand(x.shape,device=x.device)<x-low).float()
                else:x=torch.round(x)
                l.error.copy_(x.to(torch.int8))
                if measure:
                    total_sat+=(x.abs()>=levels).sum().item()
                    total_error+=debt.square().sum().item()
            else:
                l.error.zero_()
            if measure:
                delta=qnew-old
                total_flip+=(knew!=origk).sum().item();total_weights+=origk.numel()
                total_grad_dot+=(g*delta).sum().item()
                total_change+=delta.square().sum().item()
                total_grad2+=g.square().sum().item()
            l.k.copy_(knew);l.a.copy_(a);l.b.copy_(b)
            l._w=None;l._v=None
        if not measure:
            self.last_diagnostics=None
            return None
        flip=total_flip/max(1,total_weights)
        alignment=total_grad_dot/max(1e-10,math.sqrt(total_grad2*total_change))
        dw_rms=math.sqrt(total_change/max(1,total_weights))
        r_rms=math.sqrt(total_error/max(1,total_weights))
        sat=total_sat/max(1,total_weights)
        self.last_flip=flip
        self.last_diagnostics={'diag_flip_fraction':flip,
                               'diag_grad_dot_delta':total_grad_dot,
                               'diag_alignment_cosine':alignment,
                               'diag_weight_delta_rms':dw_rms,
                               'diag_residual_rms':r_rms,
                               'diag_residual_sat_fraction':sat,
                               'diag_bound_groups':0.0}
        return self.last_diagnostics

    def state_dict(self):return dict(t=self.t)
    def load_state_dict(self,state):self.t=state['t']


