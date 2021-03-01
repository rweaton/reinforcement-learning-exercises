function NextMeanRewardVec = LowPassRewardFilter(MeanRewardVec, RewardVec, lambda)
    NextMeanRewardVec = lambda*MeanRewardVec - (1 - lambda)*RewardVec;
end