function OutputStruct = TemporalDiffLearningRuleSumOfTraces4(Stimuli, Rewards, Lambda, Epsilon)

[nStimuli, nTimeSteps, nTrials] = size(Stimuli);

ExpectedRewardVec = zeros([nTimeSteps, nTrials]);
%AssociationVec = zeros([nTimeSteps+1, nStimuli, nTrials]);
AssociationVec = zeros([1,nStimuli]);
BarredStimuliVec = zeros([nStimuli, nTimeSteps-1, nTrials]);
DeltaVec = zeros([nTimeSteps-1, nTrials]);

for j = 2:nTrials
    
    BarredStimuliVec(:,:,j) = BarredStimuliVec(:,:,j-1);
    %AssociationVec(:,:,j) = AssociationVec(:,:,j-1);
    
    for i = 3:nTimeSteps
   
        BarredStimuliVec(:,i-1,j) = Lambda*BarredStimuliVec(:,i-2,j) + (1 - Lambda)*Stimuli(:,i-1,j);
        %ExpectedRewardVec(i,j) = AssociationVec(i,:,j)*Stimuli(:,i);
        ExpectedRewardVec(i,j) = AssociationVec*Stimuli(:,i,j);
        DeltaVec(i-1,j) = Rewards(i-1,j) + ExpectedRewardVec(i,j) - ExpectedRewardVec(i-1,j);
    
        %AssociationVec(i+1,:,j) = AssociationVec(i,:,j) + Epsilon*DeltaVec(i-1,j)*BarredStimuliVec(:,i-1,j)';
        AssociationVec = AssociationVec + Epsilon*DeltaVec(i-1,j)*BarredStimuliVec(:,i-1,j)';

    end
    
end


OutputStruct.ExpectedRewardVec = ExpectedRewardVec;
OutputStruct.AssociationVec = AssociationVec;
OutputStruct.DeltaVec = DeltaVec;
OutputStruct.BarredStimuliVec = BarredStimuliVec;

end