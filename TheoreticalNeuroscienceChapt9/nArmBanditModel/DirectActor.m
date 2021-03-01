function OutputStruct = DirectActor(ActionValuesVec, ProbVec, RewardVec, MeanRewardVec, Choice, epsilon)

    DeltaVec = zeros(size(ActionValuesVec));
    NewActionValuesVec = ActionValuesVec;

    nActions = length(ActionValuesVec);

    for i = 1:nActions

        DeltaVec(i) = (KroneckerDelta(i,Choice) - ProbVec(i))*...
            (RewardVec(Choice) - MeanRewardVec(i));

        NewActionValuesVec(i)  = ActionValuesVec(i) + epsilon*DeltaVec(i);

    end
    
    OutputStruct.NewActionValuesVec = NewActionValuesVec;
    OutputStruct.DeltaVec = DeltaVec;
    OutputStruct.Choice = Choice;
    OutputStruct.FilteredRewardsVec = MeanRewardVec;