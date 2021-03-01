function OutputStruct = TemporalDiffLearningRuleSumOfTraces(Stimuli, Rewards, Lambda, Epsilon, nTrials)

[nStimuli, nTimeSteps] = size(Stimuli);

ExpectedRewardVec = zeros([nTimeSteps, nTrials]);
AssociationVec = zeros([nTimeSteps+1, nStimuli]);
BarredStimuliVec = zeros([nStimuli, nTimeSteps-1, nTrials]);
DeltaVec = zeros([nTimeSteps-1, nTrials]);

%epsilon = 0.05;
for j = 1:nTrials
    for i = 3:(nTimeSteps)
   
        BarredStimuliVec(:,i-1,j) = Lambda*BarredStimuliVec(:,i-2,j) + (1 - Lambda)*Stimuli(:,i-1);
        ExpectedRewardVec(i,j) = AssociationVec(i,:)*Stimuli(:,i);
        DeltaVec(i-1,j) = Rewards(i-1) + ExpectedRewardVec(i,j) - ExpectedRewardVec(i-1,j);
    
        AssociationVec(i+1,:) = AssociationVec(i,:) + Epsilon*DeltaVec(i-1,j)*BarredStimuliVec(:,i-1,j)';

    end
end

OutputStruct.ExpectedRewardVec = ExpectedRewardVec;
OutputStruct.AssociationVec = AssociationVec;
OutputStruct.DeltaVec = DeltaVec;
OutputStruct.BarredStimuliVec = BarredStimuliVec;

end