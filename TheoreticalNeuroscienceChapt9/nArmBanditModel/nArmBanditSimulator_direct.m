function SimOutputStructArray = nArmBanditSimulator_direct(RewardsArray, InitialActionValuesVec, params)


Beta = params(1);
Epsilon = params(2);
Lambda = params(3);

[nIterations, nActions] = size(RewardsArray);

FilteredRewardsArray = NaN*ones([nIterations, nActions]);
FilteredRewardsArray(1,:) = RewardsArray(1,:);

%SimOutputStructArray = struct('DeltaVec','NewActionValues');

ActionValuesVec = InitialActionValuesVec;

for i = 1:nIterations
   
    ProbVec = SoftMaxProbabilities(ActionValuesVec, Beta);
    ActionChoice = Chooser(ProbVec);
    
    if (i > 1)
        FilteredRewardsArray(i,:) = ...
            LowPassRewardFilter(FilteredRewardsArray(i-1,:), RewardsArray(i,:), Lambda);
    end
    %ActorStruct = IndirectActor(ActionValuesVec, RewardsArray(i,:), ActionChoice, Epsilon);
    ActorStruct = DirectActor(ActionValuesVec, ProbVec, RewardsArray(i,:), ...
        FilteredRewardsArray(i,:), ActionChoice, Epsilon);
    
    ActionValuesVec = ActorStruct.NewActionValuesVec;
    
    SimOutputStructArray(i) = ActorStruct;
    
end

SimOutputStructArray = SimOutputStructArray';

end