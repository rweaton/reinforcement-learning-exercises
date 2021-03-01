function OutputStruct = OutputExtractor(SimOutputStructArray)

nIterations = length(SimOutputStructArray);

nActions = length(SimOutputStructArray(1).NewActionValuesVec);

ActionValuesArray = NaN*ones([nIterations, nActions]);
DeltaValuesArray = NaN*ones([nIterations, nActions]);
ChoiceArray = NaN*ones([nIterations,1]);

figure(1);
figure(2);

    
for j = 1:nIterations
    ActionValuesArray(j,:) = SimOutputStructArray(j).NewActionValuesVec;
    DeltaValuesArray(j,:) = SimOutputStructArray(j).DeltaVec;
    ChoiceArray(j,1) = SimOutputStructArray(j).Choice;
end

for i = 1:nActions
    figure(1); subplot(nActions, 1, i), plot(ActionValuesArray(:,i), 'k');
    figure(2); subplot(nActions, 1, i), plot(DeltaValuesArray(:,i), 'k');
end

OutputStruct.ActionValuesArray = ActionValuesArray;
OutputStruct.DeltaVec = DeltaValuesArray;
OutputStruct.ChoiceArray = ChoiceArray;

