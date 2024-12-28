import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Tuple, Union, List

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def stochChooser(pVec, nPicks, binBounds='ceilingInclusive'):
    indices = np.arange(0, pVec.shape[0])
    cpVecHigh = np.cumsum(pVec)
    cpVecLow = np.cumsum(np.hstack([0.0, pVec[:-1]]).astype(float))
    seeds = np.random.rand(nPicks,)

    picks = np.nan*np.ones(nPicks,)
    for i, seed in enumerate(seeds):
        if binBounds == 'floorInclusive':
            ceiling_check = np.logical_or((cpVecHigh > seed), (1.0*np.ones_like(cpVecHigh) <= seed))
            floor_check = (cpVecLow <= seed)
            selMask = np.logical_and(ceiling_check, floor_check)
            pick = indices[selMask]
        #selMask = np.array([((s < cpVecHigh) or (s == 1.0))
        #                    and (s >= cpVecLow) for s in seed])
        #selMask = ((seed < cpVecHigh) or (seed == 1.0)) and (seed >= cpVecLow)

        if binBounds == 'ceilingInclusive':
            floor_check = np.logical_or((cpVecLow < seed), (0.0*np.ones_like(cpVecLow) >= seed))
            ceiling_check = (cpVecHigh >= seed)
            selMask = np.logical_and(ceiling_check, floor_check)
            pick = indices[selMask]

        #selMask = ((seed > cpVecLow) or (seed == 0.0)) and (seed >= cpVecHigh)
        #selMask = np.array([((s > cpVecLow) or (s == 0.0))
        #                    and (s >= cpVecHigh) for s in seed])

        picks[i] = pick
    return picks.astype(int)

def stochChooser_pt(
    pVec: Union[torch.Tensor, np.ndarray],
    nPicks: int,
    binBounds: str='ceilingInclusive',
    device: torch.device=torch.device("cpu")
) -> torch.Tensor:
    
    if isinstance(pVec, np.ndarray):
        print("Converting pVec to torch tensor...")
        pVec = torch.tensor(pVec, dtype=torch.float32, device=device)

    indices = torch.arange(0, pVec.shape[0], dtype=torch.int32, device=device)
    print(f"Value of indices: {indices}")
    cpVecHigh = torch.cumsum(pVec, 0)
    cpVecLow = torch.cumsum(torch.hstack([
        torch.tensor([0.0], device=device),
        pVec[:-1]
    ]), 0)
    seeds = torch.rand(nPicks, device=device)
    print(f"Value of seeds: {seeds}")

    picks = torch.nan * torch.ones(nPicks, device=device)
    for i, seed in enumerate(seeds):
        if binBounds == 'floorInclusive':
            ceiling_check = torch.logical_or(
                (cpVecHigh > seed),
                (1.0 * torch.ones_like(cpVecHigh) <= seed)
            )
            floor_check = (cpVecLow <= seed)
            selMask = torch.logical_and(
                ceiling_check,
                floor_check
            )
            pick = indices[selMask]

        if binBounds == 'ceilingInclusive':
            floor_check = torch.logical_or(
                (cpVecLow < seed),
                (0.0 * torch.ones_like(cpVecLow) >= seed)
            )
            ceiling_check = (cpVecHigh >= seed)
            selMask = torch.logical_and(
                ceiling_check,
                floor_check
            )
            print(f"Value of selMask: {selMask}")
            pick = indices[selMask]

        print(f"Value of pick: {pick}")
        picks[i] = pick

    return torch.tensor(picks, dtype=torch.int32)