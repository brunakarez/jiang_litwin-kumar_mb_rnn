import torch
import numpy as np

# This script expects the model parameters, trial inputs, and initial
# conditions to already exist in the calling namespace (train.py supplies
# them with exec()). Convert inputs and states to tensors for the simulation.
m = torch.from_numpy(m0)
d = torch.from_numpy(d0)
o = torch.from_numpy(o0)
if dostdp:
    # KC-to-MBON weights are initialized separately for each batch item.
    w = torch.from_numpy(w0)
# Temporary bounded candidate used by the online weight update.
wfast = torch.from_numpy(w0)
s = torch.from_numpy(s0)
u = torch.from_numpy(u0)
rtarg = torch.from_numpy(rtarg0)
# Eligibility traces begin with no prior KC or dopamine activity.
sbar = torch.from_numpy(sbar0)
dastdpbar = torch.from_numpy(dastdpbar0)

# Mark times with no KC or external input; DAN activity above baseline is
# penalized at these times. Expand the mask across the DAN population.
baselineinds = (np.sum(s0,1,keepdims=True) == 0) & (np.sum(u0,1,keepdims=True) == 0)
baselineinds_d = torch.from_numpy(np.repeat(baselineinds,D,axis=1).astype(np.float32))

# Detached histories are kept for plotting; r remains differentiable for loss.
ma = np.zeros([T,M,B])
da = np.zeros([T,D,B])
oa = np.zeros([T,O,B])
if dostdp:
    # Record KC-to-MBON weights through time (before each step's update).
    wa = np.zeros([T,M,S,B])

# Behavioral readout and accumulated dopamine-activity regularizer.
r = torch.zeros(T,R,B)
dacost = torch.zeros(1)

# Advance all trials in the batch through the model, one time step at a time.
for ti in range(T):
    if ti in resettimes:
        # Reset neural activity and eligibility traces at a new task interval.
        # KC-to-MBON weights are not reset at these interval boundaries.
        mnew = torch.from_numpy(m0)
        dnew = torch.from_numpy(d0)
        onew = torch.from_numpy(o0)
        sbar = torch.from_numpy(sbar0)
        dastdpbar = torch.from_numpy(dastdpbar0)
    else:
        # Update MBON rates from recurrent input, feedback, current KC input
        # through the plastic weights, and the MBON bias.
        if dostdp:
            mnew = (1-dt)*m + dt*torch.relu(torch.tanh(Jmm.mm(m) + Jmo.mm(o) + torch.einsum('ijb,jb->ib',(w,s[ti,:,:])) + bm))
        else:
            mnew = (1-dt)*m + dt*torch.relu(torch.tanh(Jmm.mm(m) + Jmo.mm(o) + w.mm(s[ti,:,:]) + bm))

        # Update DAN and feedback-neuron rates from their recurrent inputs;
        # external US/context input enters the feedback population via wou.
        dnew = (1-dt)*d + dt*torch.relu(torch.tanh(Jdm.mm(m) + Jdd.mm(d) + Jdo.mm(o) + bd))
        onew = (1-dt)*o + dt*torch.relu(torch.tanh(Jom.mm(m) + Jod.mm(d) + Joo.mm(o) + wou.mm(u[ti,:]) + bo))

    # Commit the new activity states and save detached copies for diagnostics.
    m = mnew
    d = dnew
    o = onew
    ma[ti,:,:] = m.detach()
    da[ti,:,:] = d.detach()
    oa[ti,:,:] = o.detach()
    if dostdp:
        wa[ti,:,:,:] = w.detach()

    # Decode current MBON activity into the behavioral readout.
    r[ti,:,:] = wrm.mm(m)


    if dostdp:
        # Effective compartment-specific dopamine activity gates plasticity.
        # Jmd is fixed to the identity matrix in definemodel.py.
        dastdp = torch.relu(Jmd.mm(d)) # dopamine activity

        # Anti-Hebbian timing rule: current DAN activity paired with the KC
        # eligibility trace, plus the DAN trace paired with current KC input.
        stdp_update = stdpfac * (-torch.einsum('ib,jb->ijb',(dastdp,sbar)) + torch.einsum('ib,jb->ijb',(dastdpbar,s[ti,:,:]))) #anti-Hebbian

        # Apply the bounded candidate update, then let the stored weight relax
        # toward it with time constant tauw.
        wfast = torch.relu(w + dt*(stdp_update - torch.relu(stdp_update - (wmax - w)))) #update that does not exceed wmax
        w = w + (dt/tauw)*(wfast - w)

        # Update low-pass eligibility traces for the next time step and add a
        # penalty for DAN activity above baseline when there is no input.
        dastdpbar = (1. - dt/tau)*dastdpbar + (dt/tau)*dastdp
        sbar = (1. - dt/tau)*sbar + (dt/tau)*s[ti,:,:] # update low-pass trace
        dacost += torch.sum(torch.pow(torch.relu(d*baselineinds_d[ti,:,:]-dinit),2))
