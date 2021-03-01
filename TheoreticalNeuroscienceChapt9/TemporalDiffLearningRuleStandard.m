function OutputStruct = TemporalDiffLearningRuleStandard(Stimuli, Rewards, Epsilon, nTrials)

[nStimuli, nTimeSteps] = size(Stimuli);

ExpectedRewardVec = zeros([nTimeSteps, nTrials]);
AssociationVec = zeros([nTimeSteps+1, nStimuli, nTrials]);
DeltaVec = zeros([nTimeSteps-2, nTrials]);

for j = 2:nTrials
    
    AssociationVec(:,:,j) = AssociationVec(:,:,j-1);
    
    for t = 3:nTimeSteps
        
        sum = 0;
        
        for tau = 1:(t-1)
            sum = sum + AssociationVec(tau,:,j)*Stimuli(:,t-tau);
        end
        
        ExpectedRewardVec(t,j) = sum;
        
        DeltaVec(t-1,j) = Rewards(t-1) + ExpectedRewardVec(t,j) - ...
            ExpectedRewardVec(t-1,j);
        
        sum = [];
        
        for tau = 1:(t-2)
            AssociationVec(tau,:,j) = AssociationVec(tau,:,j) + ...
                Epsilon*DeltaVec(t-1,j)*Stimuli(:,t-1-tau)';
        end
        
        
    end
    
end


OutputStruct.ExpectedRewardVec = ExpectedRewardVec;
OutputStruct.AssociationVec = AssociationVec;
OutputStruct.DeltaVec = DeltaVec;

end