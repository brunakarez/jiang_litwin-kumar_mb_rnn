
#helper function to find the stimulus pattern corresponding to the first presented stimulus
def findstimA(s0):
    T,S,B = s0.shape
    sA = np.zeros([S,B])
    sums0 = np.sum(s0,1)
    for bi in range(B):
        tind = np.where(sums0[:,bi] > 0)[0][0]
        sA[:,bi] = s0[tind,:,bi]
    
    return sA


####################### initial conditions ########################
def geninitialcond(B):
    m0 = minit*np.ones([M,B],dtype=np.float32)
    d0 = dinit*np.ones([D,B],dtype=np.float32)
    o0 = oinit*np.ones([O,B],dtype=np.float32)
    w0 = np.ones([M,S,B],dtype=np.float32)/(S*f)
    sbar0 = np.zeros([S,B],dtype=np.float32)
    dabar0 = np.zeros([M,B],dtype=np.float32)

    return m0,d0,o0,w0,sbar0,dabar0


####################### trials ########################

def genrandtrials(B,probs):
    """Generate a batch of randomly selected conditioning trials.

    Each batch item gets a randomly chosen valence, trial type, and validity
    condition. Valid trials use the standard event sequence for their type;
    on invalid trials, the trial-specific stimulus/reinforcement flags are
    randomized to include omissions or substitutions.

    Parameters
    ----------
    B : int
        Number of trials to generate in the batch.
    probs : dict
        Sampling probabilities. Must contain ``"valid"``, ``"firstorder"``,
        ``"secondorder"``, and ``"extinction"``. The three trial-type
        probabilities must sum to 1.

    Returns
    -------
    s : ndarray, shape (T, S, B)
        Kenyon-cell stimulus activity for every time step and trial.
    u : ndarray, shape (T, U, B)
        External input activity, including reward or punishment signals.
    rtarg : ndarray, shape (T, R, B)
        Target readout activity (the valence the model should report).
    trialinfo : list of dict
        One metadata dictionary per trial, including its type, event times,
        stimulus identities, valence, and presentation/omission flags.
    """
    # Allocate time-series arrays; the last axis indexes independent trials.
    s = np.zeros([T,S,B],dtype=np.float32)
    u = np.zeros([T,U,B],dtype=np.float32)
    rtarg = np.zeros([T,R,B],dtype=np.float32)

    # Keep event metadata separately for plotting and inspecting each trial.
    trialinfo = [dict() for bi in range(B)]

    for bi in range(B):
        # Sample appetitive (+1) or aversive (-1) valence, task type, and
        # whether this trial follows the complete task structure.
        valence = np.random.choice([-1,1])
        ttype = np.random.choice(["firstorder","secondorder","extinction"],p=[probs["firstorder"],probs["secondorder"],probs["extinction"]])
        isvalid = np.random.rand() < probs["valid"]

        if ttype == "extinction":
            if isvalid:
                # Standard extinction sequence: pair A with US, present A
                # again without US, then test A.
                s[:,:,bi],u[:,:,bi],rtarg[:,:,bi],trialinfo[bi] = extinctiontrial(valence,True,False,True,True)
            else:
                # Randomize event flags to make an omission/substitution trial.
                s[:,:,bi],u[:,:,bi],rtarg[:,:,bi],trialinfo[bi] = extinctiontrial(valence,np.random.randint(2),np.random.randint(2),np.random.randint(2),np.random.randint(2))

        elif ttype == "secondorder":
            if isvalid:
                # Standard second-order sequence: pair A with US, pair B
                # with A, then test whether B predicts the learned valence.
                s[:,:,bi],u[:,:,bi],rtarg[:,:,bi],trialinfo[bi] = secondordertrial(valence,True,True,True,False)
            else:
                # Randomize the A/US/second-pairing/test-stimulus flags.
                s[:,:,bi],u[:,:,bi],rtarg[:,:,bi],trialinfo[bi] = secondordertrial(valence,np.random.randint(2),np.random.randint(2),np.random.randint(2),np.random.randint(2))

        elif ttype == "firstorder":
            if isvalid:
                # Standard first-order sequence: pair A with US and test A.
                s[:,:,bi],u[:,:,bi],rtarg[:,:,bi],trialinfo[bi] = firstordertrial(valence,True,False)
            else:
                # Randomly omit the US or substitute a different test cue C.
                s[:,:,bi],u[:,:,bi],rtarg[:,:,bi],trialinfo[bi] = firstordertrial(valence,np.random.randint(2),np.random.randint(2))

    return s,u,rtarg,trialinfo


####################### trial types ########################
def firstordertrial(valence: int, doUS: bool, doC: bool, teststart: int | None = 65):
    """Generate one first-order conditioning trial.

    A sparse CS A is presented, followed two seconds later by an optional
    reward or punishment. Later in the trial, the test presents either the
    trained CS A or a different CS C. The target valence is nonzero only when
    the US was delivered and the trained CS A is shown at test.

    Parameters
    ----------
    valence : int
        Valence of the US: positive for reward and negative for punishment.
    doUS : bool
        Whether to present the US after CS A.
    doC : bool
        If true, present the different stimulus C at test; otherwise present
        the trained stimulus A.
    teststart : int, optional
        Start of the test-onset sampling window, in seconds. The test starts
        randomly from ``teststart`` up to (but not including) ``teststart+10``.

    Returns
    -------
    s : ndarray, shape (T, S)
        Kenyon-cell stimulus activity over time.
    u : ndarray, shape (T, U)
        External US input over time. Channel 0 is positive valence and
        channel 1 is negative valence.
    rtarg : ndarray, shape (T, R)
        Target readout valence. It is set during the test presentation only
        for a US-paired CS A.
    trialinfo : dict
        Stimulus patterns, event times, valence, and presentation flags.
    """
    s = np.zeros([T,S])
    u = np.zeros([T,U])
    rtarg = np.zeros([T,R])

    # Draw independent sparse KC patterns for the trained CS A and test CS C.
    ainds = np.random.choice(S,int(f*S),replace=False)
    cinds = np.random.choice(S,int(f*S),replace=False)
    stimA = np.zeros(S); stimA[ainds] = 1
    stimC = np.zeros(S); stimC[cinds] = 1

    trialinfo = dict()

    # Sample CS A's onset in [5, 15) s, then convert seconds to time steps.
    tA = int(np.random.randint(5,15)/dt)
    s[tA:(tA+cslen),:] = stimA

    # The US follows the onset of CS A by dus time steps (2 s in this model).
    tUS = tA + dus

    # Test later with the trained CS A, or with the distinct CS C.
    ttest = int(np.random.randint(teststart,teststart+10)/dt)
    if doC:
        s[ttest:(ttest+cslen),:] = stimC
    else:
        s[ttest:(ttest+cslen),:] = stimA

    # Encode positive and negative US in separate external-input channels.
    if doUS:
        if valence > 0:
            u[tUS:(tUS+uslen),0] = 1.
        else:
            u[tUS:(tUS+uslen),1] = 1.

    # Only the reinforced cue should produce the trained valence at test.
    if doUS and not doC:
        rtarg[ttest:(ttest+cslen),0] = valence

    trialinfo["type"] = "firstorder"
    trialinfo["tA"] = tA
    trialinfo["tUS"] = tUS
    trialinfo["valence"] = valence
    trialinfo["ttest"] = ttest
    trialinfo["stimA"] = stimA
    trialinfo["stimC"] = stimC
    trialinfo["doUS"] = doUS
    trialinfo["doC"] = doC

    return s,u,rtarg,trialinfo

def secondordertrial(valence,doUS,doA,doA2,doC):
    ainds = np.random.choice(S,int(f*S),replace=False)
    binds = np.random.choice(S,int(f*S),replace=False)
    cinds = np.random.choice(S,int(f*S),replace=False)
    stimA = np.zeros(S); stimA[ainds] = 1
    stimB = np.zeros(S); stimB[binds] = 1
    stimC = np.zeros(S); stimC[cinds] = 1

    s = np.zeros([T,S])
    u = np.zeros([T,U])
    rtarg = np.zeros([T,R])

    tA = int(np.random.randint(5,15)/dt)
    tUS = tA + dus

    tB = int(np.random.randint(35,45)/dt)
    tA2 = tB + dus

    ttest = int(np.random.randint(65,75)/dt)

    trialinfo = dict()

    if doA:
        s[tA:(tA+cslen),:] = stimA

    if doUS:
        if valence > 0:
            u[tUS:(tUS+uslen),0] = 1.
        else:
            u[tUS:(tUS+uslen),1] = 1.

    s[tB:(tB+cslen),:] = stimB

    if doA2:
        s[tA2:(tA2+cslen),:] = stimA
    elif doC:
        s[tA2:(tA2+cslen),:] = stimC

    s[ttest:(ttest+cslen),:] = stimB

    if doUS and doA and doA2:
        rtarg[ttest:(ttest+cslen),0] = valence
        rtarg[tA2:(tA2+cslen),0] = valence

    trialinfo["type"] = "secondorder"
    trialinfo["tA"] = tA
    trialinfo["tUS"] = tUS
    trialinfo["tB"] = tB
    trialinfo["tA2"] = tA2
    trialinfo["valence"] = valence
    trialinfo["ttest"] = ttest
    trialinfo["stimA"] = stimA
    trialinfo["stimB"] = stimB
    trialinfo["stimC"] = stimC
    trialinfo["doUS"] = doUS
    trialinfo["doA"] = doA
    trialinfo["doA2"] = doA2
    trialinfo["doC"] = doC

    return s,u,rtarg,trialinfo

def extinctiontrial(valence,doUS,doUS2,doA,doA2):
    ainds = np.random.choice(S,int(f*S),replace=False)
    binds = np.random.choice(S,int(f*S),replace=False)
    stimA = np.zeros(S); stimA[ainds] = 1
    stimB = np.zeros(S); stimB[binds] = 1

    s = np.zeros([T,S])
    u = np.zeros([T,U])
    rtarg = np.zeros([T,R])

    tA = int(np.random.randint(5,15)/dt)
    tUS = tA + dus

    tA2 = int(np.random.randint(35,45)/dt)
    tUS2 = tA2 + dus

    ttest = int(np.random.randint(65,75)/dt)

    if doA:
        s[tA:(tA+cslen),:] = stimA

    if doUS:
        if valence > 0:
            u[tUS:(tUS+uslen),0] = 1.
        else:
            u[tUS:(tUS+uslen),1] = 1.

    if doA2:
        s[tA2:(tA2+cslen),:] = stimA
    else:
        s[tA2:(tA2+cslen),:] = stimB

    if doUS2:
        if valence > 0:
            u[tUS2:(tUS2+uslen),0] = 1.
        else:
            u[tUS2:(tUS2+uslen),1] = 1.

    s[ttest:(ttest+cslen),:] = stimA

    trialinfo = dict()

    if doA:
        if doUS:
            if doA2:
                rtarg[tA2:(tA2+cslen),0] = valence
                if doUS2: #two pairings
                    rtarg[ttest:(ttest+cslen),0] = valence
                else: #one pairing, extinction
                    rtarg[ttest:(ttest+cslen),0] = valence/2
            else: #one pairing, second odor presentation omitted (no extinction)
                rtarg[ttest:(ttest+cslen),0] = valence
        else: #no initial pairing
            if doA2 and doUS2:
                rtarg[ttest:(ttest+cslen),0] = valence
    else: #no initial odor
        if doA2 and doUS2:
            rtarg[ttest:(ttest+cslen),0] = valence
    trialinfo["type"] = "extinction"
    trialinfo["tA"] = tA
    trialinfo["tUS"] = tUS
    trialinfo["tUS2"] = tUS2
    trialinfo["tA2"] = tA2
    trialinfo["valence"] = valence
    trialinfo["ttest"] = ttest
    trialinfo["stimA"] = stimA
    trialinfo["stimB"] = stimB
    trialinfo["doUS"] = doUS
    trialinfo["doA"] = doA
    trialinfo["doA2"] = doA2
    trialinfo["doUS2"] = doUS2


    return s,u,rtarg,trialinfo

