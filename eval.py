import torch
import numpy as np
from scipy.stats import mannwhitneyu

def run_eval(probs_test, Ntest=50, valid=None):
    """Error rate of the current global θ on Ntest fresh trials (θ frozen, KC→MBON plastic)."""
    global B, s0, u0, rtarg0, tinfo, m0, d0, o0, w0, sbar0, dastdpbar0
    p = dict(probs_test)
    if valid is not None:
        p["valid"] = valid
    B = Ntest
    s0, u0, rtarg0, tinfo = genrandtrials(B, p)
    m0, d0, o0, w0, sbar0, dastdpbar0 = geninitialcond(B)
    with torch.no_grad():
        exec(open("runmodel.py").read(), globals())          # defines r, rtarg
    t_test = resettimes[1]                                    # test period = last interval
    err = (torch.abs(r - rtarg)[t_test:] > 0.2).any(dim=0).any(dim=0)   # (B,) bool
    return err.float().mean().item()

N_real, results = 20, {"control": [], "norecur": [], "init": []}
probs_fo = {"valid": 0.5, "firstorder": 1., "secondorder": 0., "extinction": 0.}

for seed in range(N_real):
    for cond in ["init", "control", "norecur"]:
        np.random.seed(seed); torch.manual_seed(seed)
        norecur, firstorder_only, doplot = (cond == "norecur"), True, False
        exec(open("definemodel.py").read(), globals())        # fresh θ for this realization
        exec(open("gentrials.py").read(), globals())
        if cond != "init":
            exec(open("train_body.py").read(), globals())     # train.py from `if norecur:` to the end of the epoch loop
            torch.save({v: globals()[v].detach().clone() for v in
                        ["Jmm","Jmo","Jdm","Jdd","Jdo","Jom","Jod","Joo","bm","bd","bo","wou","wrm"]},
                       f"theta_fo_{cond}_{seed}.pt")
        results[cond].append(run_eval(probs_fo, Ntest=50))

for c in ["norecur", "init"]:
    print(c, np.mean(results[c]), mannwhitneyu(results["control"], results[c]).pvalue)