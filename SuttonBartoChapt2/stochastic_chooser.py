import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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
