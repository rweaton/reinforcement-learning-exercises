function OutputStruct = RescorlaWagner(StimulusVec, RewardVec)

nIterations = length(StimulusVec);
ExpectedRewardVec = zeros(size(StimulusVec));
AssociationVec = zeros(size(StimulusVec));
DeltaVec = zeros(size(StimulusVec));
epsilon = 0.05;

for i = 1:(nIterations - 1)
   
    ExpectedRewardVec(i) = AssociationVec(i)*StimulusVec(i);
    DeltaVec(i) = RewardVec(i) - ExpectedRewardVec(i);
    
    AssociationVec(i+1) = AssociationVec(i) + epsilon*DeltaVec(i)*StimulusVec(i);
    
end

OutputStruct.ExpectedRewardVec = ExpectedRewardVec;
OutputStruct.AssociationVec = AssociationVec;
OutputStruct.DeltaVecc = DeltaVec;


figure; plot(AssociationVec,'k*')
xlabel('time steps');
ylabel('stimulus-reward association w')
title('Acquisition and Extiction by Rescorla-Wagner Model')