function OutputStruct = OutputPlotter(SimOutputStructArray)

nIterations = length(SimOutputStructArray);

nActions = length(SimOutputStructArray(1).NewActionValuesVec);

ActionValuesArray = NaN*ones([nIterations, nActions]);
DeltaValuesArray = NaN*ones([nIterations, nActions]);
ChoiceArray = NaN*ones([1,nIterations]);

figure(1);
figure(2);

for i = 1:nActions
    
    for j = 1:nIterations
        
        ActionValuesArray(i,j) = SimOutputStructArray(j).NewActionValuesVec(i);
        DeltaValuesArray(i,j) = SimOutputStructArray(j).DeltaVec(i);        
    end
    
    figure(1); subplot(nActions, 1, i), plot(ActionValuesArray(i,:), 'k');
    figure(2); subplot(nActions, 1, i), plot(DeltaValuesArray(i,:), 'k');
end

for j = 1:nIterations
    ChoiceArray(1,j) = SimOutputStructArray(j).Choice;
end

OutputStruct.ActionValuesArray = ActionValuesArray;
OutputStruct.DeltaVec = DeltaValuesArray;
OutputStruct.ChoiceArray = ChoiceArray;

