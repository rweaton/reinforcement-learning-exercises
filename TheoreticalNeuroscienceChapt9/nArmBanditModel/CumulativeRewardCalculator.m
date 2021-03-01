function CumulativeReward = CumulativeRewardCalculator(SimOutputStruct,RewardArray)

[nTrials, nActions] = size(RewardArray);
RewardOccurances = NaN*ones([nTrials, 1]);

for i = 1:nTrials
    
    RewardOccurances(i) = RewardArray(i,SimOutputStruct.ChoiceArray(i));

end

CumulativeReward = sum(RewardOccurances);