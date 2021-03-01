function SimOutputStructArray = nArmBanditSimulator(RewardsArray, InitialActionValuesVec, params)


Beta = params(1);
Epsilon = params(2);

[nIterations, nActions] = size(RewardsArray);

%SimOutputStructArray = struct('DeltaVec','NewActionValues');

ActionValuesVec = InitialActionValuesVec;

for i = 1:nIterations
   
    ProbVec = SoftMaxProbabilities(ActionValuesVec, Beta);
    ActionChoice = Chooser(ProbVec);
    
    ActorStruct = IndirectActor(ActionValuesVec, RewardsArray(i,:), ActionChoice, Epsilon);
    
    ActionValuesVec = ActorStruct.NewActionValuesVec;
    
    SimOutputStructArray(i) = ActorStruct;
    
end

SimOutputStructArray = SimOutputStructArray';

end