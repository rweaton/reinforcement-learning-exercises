function OutputStruct = ActorCriticSimulator3(StateVectors, ActionsList, nIterations, params)
    
    i = 0;
    DiscountRate = 1; 

    %ActionsList = ActionsList(:);
    epsilon_Critic = params(1);
    epsilon_Actor = params(2);
    beta = params(3);

    %%% Each state vector is a column in the array StateVectors %%%
    [nElements, nStates] = size(StateVectors);
    nActions = length(ActionsList);
   
    ExpectedRewards = NaN*ones([nIterations, 1]);
    RewardVec = NaN*ones([nIterations, 1]);
    StateIndices = NaN*ones([nIterations, 1]);
    DeltaVec = NaN*ones([nIterations, 1]);
    
    % Initialize variables for first iteration

    CriticWeights = zeros([1, nElements]);
    ActionMatrix = zeros([nActions, nElements]);
    StateIndices(1) = 1; 
    UpdateCritic = 1;
    
    CriticWeightsArray = NaN*ones([nElements, nIterations]);
    ActionMatrixArray = NaN*ones([nActions,nElements, nIterations]);
    
    ExpectedRewards(1) = CriticWeights*StateVectors(:,StateIndices(1));
    
    for i = 1:(nIterations-1)
        
        % Recompute ExpectedRewards after CriticWeights has been updated
        ExpectedRewards(i) = CriticWeights*StateVectors(:,StateIndices(i));
                
        %%%%%%%%%%%%%%%%%%%%% ACTOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ActionValuesVec = ActionMatrix*StateVectors(:,StateIndices(i));

        ProbVec = SoftMaxProbabilities(ActionValuesVec, beta);

        ChoiceIndex = Chooser(ProbVec);
        Choice = ActionsList(ChoiceIndex);
        
        % NOTE: This function emulates the maze task.  Choice must be
        % either an 'l' or 'r' character
        NewStateStruct = MazeTaskEvaluator(StateVectors, StateIndices(i), Choice);
        
        % Store output
        SimOutputStructArray(i) = NewStateStruct;
        
        StateIndices(i+1) = NewStateStruct.NextStateIndex;
        RewardVec(i) = NewStateStruct.Reward;
        %UpdateCritic = NewStateStruct.UpdateCritic;
        UpdateCritic = 1;
        EndOfTrial = NewStateStruct.EndOfTrial;

        ActionMatrixArray(:,:,i) = ActionMatrix;
        %%%%%%%%%%%%%%%%%%%%%% CRITIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Evaluate expected reward for choice made at the end of the
        % previous iteration
        if (EndOfTrial == 0)
            ExpectedRewards(i+1) = CriticWeights*StateVectors(:,StateIndices(i+1));
        elseif (EndOfTrial == 1)
            ExpectedRewards(i+1) = 0;
        end
        
        %%%%%%%%%% UPDATE CRITIC %%%%%%%%%%%%%%%
        if (UpdateCritic == 1)
        % Calculate the scale of change to be made to critic weights using
        % temporal difference rule
        Delta = RewardVec(i) + DiscountRate*ExpectedRewards(i+1) - ExpectedRewards(i);
        
        % Update CriticWeights
        CriticWeights = CriticWeights...
            + epsilon_Critic*Delta*StateVectors(:,StateIndices(i))';
        
        elseif (UpdateCritic == 0)
            % When Critic is not updated Delta is set to zero
            Delta = 0;
        end
        DeltaVec(i) = Delta; 
        
        %%%%%%%%% UPDATE ACTOR %%%%%%%%%%%%%%%%% 
        % Calculate the updated ActionMatrix
        NewActionMatrix = AdjustActionMatrix(ActionMatrix, ...
            StateVectors(:,StateIndices(i)), ProbVec, ChoiceIndex, Delta, epsilon_Actor);
        
        % Replace old values with new ones for for evaluation by critic
        ActionMatrix = NewActionMatrix;
        
        CriticWeightsArray(:,i) = CriticWeights';
            
    end
    
    AverageCriticWeights = sum(CriticWeightsArray(:,50:(nIterations-1)),2)/(nIterations - 50);
    
    figure; hold on;
    subplot(3,1,1), plot(CriticWeightsArray(1,:),'k'), hold on, ...
        %plot(AverageCriticWeights(1,1)*ones([1,nIterations]),'r--');
    
    subplot(3,1,2), plot(CriticWeightsArray(2,:),'k'), hold on, ... 
        %plot(AverageCriticWeights(2,2)*ones([1,nIterations]),'r--');
    
    subplot(3,1,3), plot(CriticWeightsArray(3,:),'k'), hold on, ... 
        %plot(AverageCriticWeights(3,3)*ones([1,nIterations]),'r--');
    

    OutputStruct.CriticWeightsArray = CriticWeightsArray;
    OutputStruct.ActionMatrixArray = ActionMatrixArray;
    
    % Calculate the progression of probabilities of choosing left at each location
    % Location A
    A_ActionVecs = squeeze(OutputStruct.ActionMatrixArray(:,1,:));
    % Location B
    B_ActionVecs = squeeze(OutputStruct.ActionMatrixArray(:,2,:));
    % Location C
    C_ActionVecs = squeeze(OutputStruct.ActionMatrixArray(:,3,:));
    
    ProbA = NaN*ones([2,nIterations]);
    ProbB = NaN*ones([2,nIterations]);
    ProbC = NaN*ones([2,nIterations]);
    
    for i = 1:(nIterations-1)
        
        ProbA(:,i) = SoftMaxProbabilities(A_ActionVecs(:,i), beta);
        ProbB(:,i) = SoftMaxProbabilities(B_ActionVecs(:,i), beta);
        ProbC(:,i) = SoftMaxProbabilities(C_ActionVecs(:,i), beta);
        
    end
    
    figure; hold on;
    subplot(3,1,1), plot(ProbA(1,:));
    subplot(3,1,2), plot(ProbB(1,:));
    subplot(3,1,3), plot(ProbC(1,:));
    
    
end