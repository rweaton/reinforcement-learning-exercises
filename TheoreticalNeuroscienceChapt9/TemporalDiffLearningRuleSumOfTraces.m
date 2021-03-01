function OutputStruct = TemporalDiffLearningRuleSumOfTraces(Stimuli, Rewards, Lambda, Epsilon)

[nStimuli, nIterations] = size(Stimuli);

ExpectedRewardVec = zeros([1, nIterations]);
AssociationVec = zeros([nStimuli, nIterations+1])';
BarredStimuliVec = zeros([nStimuli, nIterations-1]);
DeltaVec = zeros([1, nIterations-1]);

%epsilon = 0.05;

for i = 3:(nIterations)
   
    BarredStimuliVec(:,i-1) = Lambda*BarredStimuliVec(:,i-2) + (1 - Lambda)*Stimuli(:,i-1);
    ExpectedRewardVec(i) = AssociationVec(i,:)*Stimuli(:,i);
    DeltaVec(i-1) = Rewards(i-1) + ExpectedRewardVec(i) - ExpectedRewardVec(i-1);
    
    AssociationVec(i+1,:) = AssociationVec(i,:) + Epsilon*DeltaVec(i-1)*BarredStimuliVec(:,i-1)';

end

OutputStruct.ExpectedRewardVec = ExpectedRewardVec;
OutputStruct.AssociationVec = AssociationVec';
OutputStruct.DeltaVec = DeltaVec;
OutputStruct.BarredStimuliVec = BarredStimuliVec;

end