function OutputStruct = RescorlaWagnerMultiStimulus(StimuliVec, RewardVec)

[nStimuli, nIterations] = size(StimuliVec);

ExpectedRewardVec = zeros([1, nIterations]);
AssociationVec = zeros([nStimuli, nIterations])';
DeltaVec = zeros([1, nIterations]);

epsilon = 0.05;

for i = 1:(nIterations - 1)
   
    ExpectedRewardVec(i) = AssociationVec(i,:)*StimuliVec(:,i);
    DeltaVec(i) = RewardVec(i) - ExpectedRewardVec(i);
    
    AssociationVec(i+1,:) = AssociationVec(i,:) + epsilon*DeltaVec(i)*StimuliVec(:,i)';
    
end

OutputStruct.ExpectedRewardVec = ExpectedRewardVec;
OutputStruct.AssociationVec = AssociationVec;
OutputStruct.DeltaVec = DeltaVec;


% figure; plot(AssociationVec,'k*')
% xlabel('trial number');
% ylabel('stimulus-reward association w')
% title('Acquisition and Extiction by Rescorla-Wagner Model')