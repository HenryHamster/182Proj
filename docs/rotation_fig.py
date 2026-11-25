import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, sys
out = sys.argv[1]; PHI = (5**.5-1)/2; rng = np.random.default_rng(0)
N, R = 2000, 2000
t = np.arange(1, N+1)
def run(p):
    uS = rng.random((R, N)); uR = np.mod(rng.random((R, 1)) + np.arange(N) * PHI, 1)
    return ((uS < p) - p), ((uR < p) - p)
eS0, eR0 = run(np.full((R, N), .3))          # constant p: rho = 1
eS1, eR1 = run(rng.random((R, N)))           # uniform p: rho = 1/2
SS1, SR1 = eS1.cumsum(1), eR1.cumsum(1)
print("uniform p: Var S_N/N", SS1[:, -1].var()/N, SR1[:, -1].var()/N, "ratio", SR1[:, -1].var()/SS1[:, -1].var())
a = .99
def ar(e):
    d = np.zeros(e.shape[0]); v = []
    for k in range(e.shape[1]):
        d = a*d + e[:, k]
        if k >= 500: v.append(d.var())
    return np.mean(v)
vS, vR = ar(eS1), ar(eR1)
print("uniform p: weight var, eta h=.01", vS, vR, "ratio", vR/vS, "theory SR", (1/6)/(1-a*a))
cS, cR = "#1baf7a", "#2a78d6"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
for k in range(5):
    ax[0].plot(t, eS0[k].cumsum(), color=cS, lw=.8, alpha=.8); ax[0].plot(t, eR0[k].cumsum(), color=cR, lw=.8, alpha=.8)
ax[0].plot([], [], color=cS, label="SR"); ax[0].plot([], [], color=cR, label="rotation")
ax[0].set_xlabel("step $t$"); ax[0].set_ylabel("accumulated error $S_t$ (grid units)")
ax[0].set_title("(a) same fraction $p=0.3$ every step", fontsize=9, loc="left"); ax[0].legend(frameon=False)
ax[1].plot(t, SS1.std(0), color=cS, lw=2, label="SR"); ax[1].plot(t, SR1.std(0), color=cR, lw=2, label="rotation")
ax[1].plot(t, np.sqrt(t/6), color="k", lw=.8, ls="--", label=r"theory: $\sqrt{t\,\mathbb{E}[p(1-p)]}$, $\times\sqrt{1-\rho}$")
ax[1].plot(t, np.sqrt(t/12), color="k", lw=.8, ls="--")
ax[1].set_xlabel("step $t$"); ax[1].set_ylabel("typical size of $S_t$ (std)")
ax[1].set_title(r"(b) $p_t$ random, uniform on $[0,1)$: $\rho=1/2$", fontsize=9, loc="left"); ax[1].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(out, bbox_inches="tight")
