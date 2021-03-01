function OutputStruct = IndirectActor(ActionValuesVec, RewardVec, Choice, epsilon)

    DeltaVec = zeros(size(ActionValuesVec));
    NewActionValuesVec = ActionValuesVec;

    DeltaVec(:,Choice) = RewardVec(:,Choice) - ActionValuesVec(:,Choice);

    NewActionValuesVec = ActionValuesVec + epsilon*DeltaVec;

    OutputStruct.DeltaVec = DeltaVec;
    OutputStruct.NewActionValuesVec = NewActionValuesVec;
    OutputStruct.Choice = Choice;

end