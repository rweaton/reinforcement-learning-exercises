function RewardArray = StochasticRewardsGenerator(IntratrialTimeOfReward, ProbabilityOfReward, nTrials, TrialDuration, TimeInc)


TimeVec = 0:TimeInc:TrialDuration;
nTimeSamples = length(TimeVec);

RewardArray = zeros(nTimeSamples, nTrials);

Indices = find((TimeVec >= IntratrialTimeOfReward - TimeInc) & ...
      (TimeVec <= IntratrialTimeOfReward + 2*TimeInc));

for i = 1:nTrials
        
    if rand(1) <= ProbabilityOfReward
        RewardArray(Indices, i) = 1/5;
    end
end



end