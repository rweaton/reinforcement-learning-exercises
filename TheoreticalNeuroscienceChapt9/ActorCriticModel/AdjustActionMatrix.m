function NewActionMatrix = AdjustActionMatrix(ActionMatrix, StateVec, ProbVec, ActionChoice, Delta, epsilon)

    [nActions, nElements] = size(ActionMatrix);
    NewActionMatrix = zeros([nActions, nElements]);
    
    for i = 1:nActions
        
        NewActionMatrix(i,:) = ActionMatrix(i,:) + ...
            epsilon*(KroneckerDelta(ActionChoice,i) - ProbVec(i))*Delta*StateVec';
        
    end

end