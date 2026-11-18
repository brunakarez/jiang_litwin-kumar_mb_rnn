import torch

def update_kc_mbon_weights(W: torch.Tensor, r_kc: torch.Tensor, r_dan: torch.Tensor, rbar_kc: torch.Tensor, rbar_dan:torch.Tensor,
                           *, dt: float, tau: float, tauw: float, wmax: float, stdpfac: float | None = 1.0
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the KC→MBON plasticity rule by one timestep.
    
    Parameters
    ----------
    W : torch.Tensor
        The current weights of the KC→MBON synapses.
    r_kc : torch.Tensor
        The current firing rates of the KCs.
    r_dan : torch.Tensor
        The current firing rates of the DANs.
    rbar_kc : torch.Tensor
        The low-pass filtered firing rates of the KCs.
    rbar_dan : torch.Tensor
        The low-pass filtered firing rates of the DANs.
    dt : float
        The timestep size.
    tau : float
        The time constant for the low-pass filter.
    tauw : float
        The time constant for the weight update.
    wmax : float
        The maximum allowed weight value.
    stdpfac : float, optional
        A scaling factor for the STDP update, by default 1.0.
    
    Returns
    -------
    W_next : torch.Tensor
        The updated weights of the KC→MBON synapses.
    rbar_kc_next : torch.Tensor
        The updated low-pass filtered firing rates of the KCs.
    rbar_dan_next : torch.Tensor
        The updated low-pass filtered firing rates of the DANs.
    """
    dw = stdpfac * (
        -torch.einsum("ib,jb->ijb", (r_dan, rbar_kc))
        + torch.einsum("ib,jb->ijb", (rbar_dan, r_kc))
    )

    W_fast = torch.relu(
        W + dt * (dw - torch.relu(dw - (wmax - W)))
    )
    W_next = W + (dt / tauw) * (W_fast - W)

    rbar_kc_next = (1 - dt / tau) * rbar_kc + (dt / tau) * r_kc
    rbar_dan_next = (1 - dt / tau) * rbar_dan + (dt / tau) * r_dan

    return W_next, rbar_kc_next, rbar_dan_next