function OutputStruct = DirectActor(ActionValuesVec, RewardVec, Choice, eta)

DeltaVec = zeros(size(RewardVec));
NewActionValuesVec = ActionValuesVec;


DeltaVec(Choice) = RewardVec(Choice) - ActionValuesVec(Choice);

NewActionValuesVec = NewActionValuesVec(Choice) + eta*DeltaVec(Choice);

OutputStruct.DeltaVec = DeltaVec;
OutputStruct.NewActionValuesVec = NewActionValuesVec;

end