function SimOutputStructArray = ActorCriticSimulator(StateVectors, ActionsList, nIterations, params)

    %% Requires major algorithmic Revision!!!! %%%%%%%%

    %ActionsList = ActionsList(:);
    epsilon = params(1);
    beta = params(2);

    %%% Each state vector is a column in the array StateVectors %%%
    [nElements, nStates] = size(StateVectors);
    nActions = length(ActionsList);
    
    ExpectedRewards = NaN*ones([nIterations, 1]);
    RewardVec = NaN*ones([nIterations, 1]);
    StateIndices = NaN*ones([nIterations, 1]);
    
    % Initialize variables for first iteration
    ExpectedRewards(1) = 0;
    CriticWeights = zeros([nStates, nElements]);
    ActionMatrix = zeros([nActions, nElements]);
    StateVec = StateVectors(:,1);
    StateIndex = 1;
    StateIndices(1) = 1;
    RewardVec(1) = 0; RewardVec(2) = 0;
    UpdateCritic = 1;
    
    for i = 2:nIterations
         
        %%%%%%%%%%%%%%%%%%%%%% CRITIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Evaluate expected reward for choice made at the end of the
        % previous iteration
        ExpectedRewards(i) = CriticWeights(StateIndex,:)*StateVec;
        
        if (UpdateCritic == 1)
        % Calculate the scale of change to be made to critic weights using
        % temporal difference rule
        Delta = RewardVec(i-1) + ExpectedRewards(i) - ExpectedRewards(i-1);
        
        % Update CriticWeights
        CriticWeights(StateIndices(i-1),:) = CriticWeights(StateIndices(i-1),:)...
            + epsilon*Delta*StateVectors(:,StateIndices(i-1))';
        
        elseif (UpdateCritic == 0)
            % When Critic is not updated Delta is set to zero
            Delta = 0;
        end
        
        %%%%%%%%%%%%%%%%%%%%% ACTOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ActionValuesVec = ActionMatrix*StateVec;

        ProbVec = SoftMaxProbabilities(ActionValuesVec, beta);

        ChoiceIndex = Chooser(ProbVec);
        Choice = ActionsList(ChoiceIndex);
        
        % NOTE: This function emulates the maze task.  Choice must be
        % either an 'l' or 'r' character
        NewStateStruct = StateEvaluator(StateVec, Choice);
        
        SimOutputStructArray(i) = NewStateStruct;
        
        % Calculate the updated ActionMatrix
        NewActionMatrix = AdjustActionMatrix(ActionMatrix, ...
            StateVectors(:,ChoiceIndex), ProbVec, ChoiceIndex, Delta, epsilon);
        
        % Retain index of current StateVec
        StateIndices(i) = StateIndex;
        
        % Replace old values with new ones for for evaluation by critic
        ActionMatrix = NewActionMatrix;
        StateVec = NewStateStruct.NextStateVec;
        StateIndex = NewStateStruct.NextStateIndex;
        RewardVec(i+1) = NewStateStruct.Reward;
        UpdateCritic = NewStateStruct.UpdateCritic;
    
                

    end

    0 == 0;
end