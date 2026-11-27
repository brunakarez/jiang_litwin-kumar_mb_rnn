import numpy as np
import matplotlib.pyplot as plt

def plotfill(tinfo,hmin,hmax):
    Acol = "b"
    Bcol = "c"
    Ccol = "m"
    if tinfo["valence"] > 0:
        UScol = "g"
    else:
        UScol = "r"

    tA = tinfo["tA"]
    tUS = tinfo["tUS"]
    ttest = tinfo["ttest"]

    if tinfo["type"] == "firstorder":
        plt.fill_between([tA,tA+cslen],[hmin,hmin],[hmax,hmax],color=Acol,alpha=0.2,linewidth=0)
        if tinfo["doC"]:
            plt.fill_between([ttest,ttest+cslen],[hmin,hmin],[hmax,hmax],color=Ccol,alpha=0.2,linewidth=0)
        else:
            plt.fill_between([ttest,ttest+cslen],[hmin,hmin],[hmax,hmax],color=Acol,alpha=0.2,linewidth=0)
        if tinfo["doUS"]:
            plt.fill_between([tUS,tUS+uslen],[hmin,hmin],[hmax,hmax],color=UScol,alpha=0.2,linewidth=0)
    elif tinfo["type"] == "secondorder":
        tA2 = tinfo["tA2"]
        tB = tinfo["tB"]
        plt.fill_between([tA,tA+cslen],[hmin,hmin],[hmax,hmax],color=Acol,alpha=0.2,linewidth=0)
        if tinfo["doA2"]:
            plt.fill_between([tA2,tA2+cslen],[hmin,hmin],[hmax,hmax],color=Acol,alpha=0.2,linewidth=0)
        if tinfo["doC"]:
            plt.fill_between([ttest,ttest+cslen],[hmin,hmin],[hmax,hmax],color=Ccol,alpha=0.2,linewidth=0)
        else:
            plt.fill_between([ttest,ttest+cslen],[hmin,hmin],[hmax,hmax],color=Bcol,alpha=0.2,linewidth=0)
        plt.fill_between([tB,tB+cslen],[hmin,hmin],[hmax,hmax],color=Bcol,alpha=0.2,linewidth=0)
        if tinfo["doUS"]:
            plt.fill_between([tUS,tUS+uslen],[hmin,hmin],[hmax,hmax],color=UScol,alpha=0.2,linewidth=0)
    elif tinfo["type"] == "extinction":
        tA2 = tinfo["tA2"]
        tUS2 = tinfo["tUS2"]
        if tinfo["doA"]:
            plt.fill_between([tA,tA+cslen],[hmin,hmin],[hmax,hmax],color=Acol,alpha=0.2,linewidth=0)
        if tinfo["doA2"]:
            plt.fill_between([tA2,tA2+cslen],[hmin,hmin],[hmax,hmax],color=Acol,alpha=0.2,linewidth=0)
        else:
            plt.fill_between([tA2,tA2+cslen],[hmin,hmin],[hmax,hmax],color=Bcol,alpha=0.2,linewidth=0)
        if tinfo["doUS2"]:
            plt.fill_between([tUS2,tUS2+uslen],[hmin,hmin],[hmax,hmax],color=UScol,alpha=0.2,linewidth=0)
        plt.fill_between([ttest,ttest+cslen],[hmin,hmin],[hmax,hmax],color=Acol,alpha=0.2,linewidth=0)
        if tinfo["doUS"]:
            plt.fill_between([tUS,tUS+uslen],[hmin,hmin],[hmax,hmax],color=UScol,alpha=0.2,linewidth=0)

def plottarg(r,rtarg,tinfo): #plot target valence
    tranges = (np.arange(5,50),np.arange(65,110),np.arange(125,180)) #time intervals to plot

    for trange in tranges:
        plt.plot(trange,r[trange],"gray")
        plt.plot(trange,rtarg[trange],"k")

    if tinfo["valence"] == 1:
        hmin = -.1
        hmax = 1.1
    else:
        hmin = -1.1
        hmax = 0.1

    plotfill(tinfo,hmin,hmax)
    plt.xlim(0,T)
    plt.xticks(())
    plt.yticks(())


def plotunits(r,tinfo):
    """Plot each neuron's activity as an offset trace over trial segments.

    Parameters
    ----------
    r : ndarray, shape (T, D)
        Activity over time for one batch item, typically the DAN activity.
    tinfo : dict
        Metadata for the trial, used by ``plotfill`` to shade stimulus and
        reinforcement periods.

    The function draws into the current Matplotlib axes and returns nothing.
    The x-axis positions remain in simulation steps so they line up with the
    event shading; tick labels are converted to seconds using the global ``dt``.
    """
    # Show the three task intervals and leave the gaps between them blank.
    tranges = (np.arange(5,50),np.arange(65,110),np.arange(125,180))

    # Vertically separate activity traces; 1.5 leaves visible space between
    # adjacent traces whose rectified activity is typically between 0 and 1.
    hdiff = 1.5
    count = 0
    trial_length, nb_neurons = np.shape(r)
    nb_neurons_plot = nb_neurons//2
    inds = np.arange(nb_neurons_plot)  # indices of neurons to plot
    for ind in inds:
        for trange in tranges:
            # Plot the same neuron across each displayed interval, with a
            # per-neuron vertical offset to avoid overlapping traces.
            plt.plot(trange,r[trange,ind]+hdiff*count,"k")
        count += 1

    # Shade the trial events behind all traces.
    hmax = len(inds)*hdiff
    hmin = 0
    plotfill(tinfo,hmin,hmax)

    # Hide the neuron-index scale, but label simulation times in seconds.
    plt.yticks(())
    plt.xlim(0,trial_length)
    tick_interval_s = 20.
    tick_times_s = np.arange(0, trial_length*dt + 1e-9, tick_interval_s)
    tick_steps = tick_times_s / dt
    plt.xticks(tick_steps, [f"{time_s:g}" for time_s in tick_times_s])
    plt.xlabel("Time (s)")
