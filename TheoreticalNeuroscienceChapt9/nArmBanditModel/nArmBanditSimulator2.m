function OutputStruct = nArmBanditSimulator2(RewardsArray, InitialActionValuesVec, params)


Beta = params(1);
Eta = params(2);

[nIterations, nActions] = size(RewardsArray);

ActionValuesArray = NaN*ones(size(RewardsArray));
DeltaValuesArray = NaN*ones([nIterations,nActions]);
ChoiceList = NaN*ones([nIterations,1]);

ActionValuesArray(1,:) = InitialActionValuesVec;

for i = 1:(nIterations-1)
   
    ProbVec = SoftMaxProbabilities(ActionValuesArray(i,:), Beta);
    ActionChoice = Chooser(ProbVec);
    
    ActorStruct = IndirectActor(ActionValuesArray(i,:), RewardsArray(i,:), ActionChoice, Eta);
    
    ActionValuesArray(i+1,:) = ActorStruct.NewActionValuesVec;
    DeltaValuesArray(i+1,:) = ActorStruct.DeltaVec;
    ChoiceList(i+1,1) = ActionChoice;
    
end

OutputStruct.ActionValuesArray = ActionValuesArray;
OutputStruct.DeltaValuesArray = DeltaValuesArray;
OutputStruct.ChoiceList = ChoiceList;

end