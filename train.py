import torch
import time
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# Training options. These scripts share variables through exec(), so the
# settings here control model definition, trial generation, and simulation.
doplot = True #show output during training
norecur = False #no recurrent MBON output circuitry
firstorder_only = False #only train for first-order trials
dostdp = True #do dopamine-dependent STDP updates

# Each epoch presents a fresh batch of trials to the model
B = 30 #batch size
alphada = 0.01 #regularization parameter for dopamine activity

# A valid trial follows the selected task structure. On other trials,
# gentrials.py randomly omits or substitutes task events.
probs = dict()
probs["valid"] = 0.5
if firstorder_only:
    Nepochs = 2000
    probs["firstorder"] = 1.
    probs["secondorder"] = 0.
    probs["extinction"] = 0.
else:
    Nepochs = 5000
    probs["firstorder"] = 0.
    probs["secondorder"] = 0.5
    probs["extinction"] = 0.5

lr = 0.001 * np.ones(Nepochs) #learning rate
Nepochspc = max(int(Nepochs/100),1) #plot training status after this interval

# Load model parameters and plotting helpers into this script's namespace.
exec(open("definemodel.py").read())
exec(open("plotutils.py").read())
if norecur:
    # Disable selected recurrent pathways and optimize only the remaining
    # trainable parameters (the feedback-to-DAN pathway, biases, inputs, readout).
    Jmm.data[:] = 0
    Jmo.data[:] = 0
    Jdm.data[:] = 0
    Jdd.data[:] = 0
    Jom.data[:] = 0
    Jod.data[:] = 0
    Joo.data[:] = 0
    train_vars = [Jdo,bm,bd,bo,wou,wrm]

exec(open("gentrials.py").read())

# RMSprop updates the network parameters after each batch. KC-to-MBON weights
# are updated online inside runmodel.py, not by this optimizer.
opt = torch.optim.RMSprop(train_vars,lr=lr[0])

# Store one scalar training loss per epoch.
track_loss = np.zeros(Nepochs)

lastt = time.time()
for ei in range(Nepochs):
    # Apply this epoch's learning rate (currently constant across epochs).
    for g in opt.param_groups:
        g['lr'] = lr[ei]

    # Generate B new trial sequences and their desired readout trajectories.
    s0,u0,rtarg0,tinfo = genrandtrials(B,probs)

    # Reset neural activity, plastic KC-to-MBON weights, and eligibility traces
    # at the beginning of each trial in the batch.
    m0,d0,o0,w0,sbar0,dastdpbar0 = geninitialcond(B)

    # Simulate the trial through time, producing activity, readout, and loss
    # components used below. runmodel.py reads the arrays and parameters above.
    exec(open("runmodel.py").read())

    # Match the model readout to the target valence and penalize excess
    # dopamine activity during baseline periods.
    loss_err = torch.sum(torch.pow(r-rtarg,2))/B
    loss_da = alphada*dacost/B
    loss = loss_err + loss_da
    track_loss[ei] = loss

    # Backpropagate through the simulated batch and update train_vars.
    loss.backward()
    opt.step()
    opt.zero_grad()

    # Periodically show the loss history plus one example trial (batch item 0).
    if (ei % Nepochspc) == 0:
        if doplot:
            plt.clf()
            plt.subplot(311)
            if ei > 0:
                plt.semilogy(track_loss[:ei])
            plt.xlim(0,Nepochs)
            plt.xlabel("epoch")
            plt.ylabel("loss")

            plt.subplot(312)
            plottarg(r.detach().numpy()[:,:,0],rtarg.detach().numpy()[:,:,0],tinfo[0])
            plt.ylabel("valence")

            plt.subplot(313)
            plotunits(da[:,:,0],tinfo[0])
            plt.ylabel("DAN activity")
            plt.xlabel("time")

            plt.tight_layout()

            plt.pause(.0001)
            plt.show()

        curt = time.time()
        print("\r" + str(int(ei/Nepochspc)) + "%, ", np.round(curt-lastt,2), "seconds", end="")
        lastt = curt


