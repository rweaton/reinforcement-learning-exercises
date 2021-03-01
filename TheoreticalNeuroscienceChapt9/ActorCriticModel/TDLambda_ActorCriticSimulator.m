function OutputStruct = TDLambda_ActorCriticSimulator(StatesList, ActionsList, nIterationsPerEpisode, nEpisodes, params)

    %%%%%%%%%%%%%% Description of arguments %%%%%%%%%%%%%%%%%
    % StatesList:  Each column of this array is a vector describing
    % a particular state of the animal's environs/experience.
    
    % ActionsList: A vector listing the possible actions available to the
    % animal.  Actions guide transitions to new states.
    
    % nIterationsPerEpisode: The number of iterative steps (timesteps) 
    % within each reinforcement episode (trial)
    
    % nEpisodes: The number of reinforcement episodes (trials) during 
    % the simulation
 
    %%%%%%%%%%%%% Unpack Model Parameters %%%%%%%%%%%%%%%%%%%%%
    epsilon_Critic = params(1);
    epsilon_Actor = params(2);
    lambda_Critic = params(3);
    lambda_Actor = params(4);
    beta_SoftMax = params(5);
    gamma_discount = params(5);
    
    
    %%%%%%% Evaluate Sizes of Arguments for Array Initialization %%%%%%%
    [nElements, nStates] = size(StatesList);
    nActions = length(ActionsList);
    
    %%%%%%%%%%%%% Initialize Progression Record %%%%%%%%%%%%%%%%
    StateIndices = NaN*ones([nIterationsPerEpisode, nEpisodes]);
    DeltaArray = NaN*ones([nIterationsPerEpisode, nEpisodes]);
    
    %%%%%%%%%%%%% Initialize Weight Arrays %%%%%%%%%%%%%%%%%%%%%%%
    CriticWeightsArray = NaN*ones([nElements, nIterationsPerEpisode, nEpisodes]);
    ActionMatrixArray = NaN*ones([nActions, nElements, nIterationsPerEpisode, nEpisodes]);
    
    %%%%%%%%%%%%%% Initialize Reward Arrays %%%%%%%%%%%%%%%%%%%%
    ExpectedRewardsArray = NaN*ones([nIterationsPerEpisode, nEpisodes]);
    RewardsArray = NaN*ones([nIterationsPerEpisode, nEpisodes]);
    
    %%%%%%%%%%%%%% Initialize Choice Probabilities Array %%%%%%%
    ProbabilitiesArray = NaN*ones([nActions, nIterationsPerEpisode, nEpisodes]);
    ChoicesArray = NaN*ones([nIterationsPerEpisode, nEpisodes]);
    
    %%%%%%%%%%%%%% Initialize Eligibility Traces %%%%%%%%%%%%%%%%
    CriticEligibilityTraces = NaN*ones([nElements, nIterationsPerEpisode, nEpisodes]);
    StateActionPairEligibilityTraces = NaN*ones([nActions, nStates, nIterationsPerEpisode, nEpisodes]);
    
    
    % k is the EPISODE index
    % j is the intra-episode ITERATION index
    % i is the ACTION CHOICE index
    
    UpdateCritic = 1;
    
    
    for k = 1:nEpisodes
        
        if (k == 1)
            
            % Initialize variables for first iteration
            CriticWeightsArray(:,1,1) = zeros([nElements,1,1]);
            ActionMatrixArray(:,:,1,1) = zeros([nActions, nElements,1,1]);
            %ActionMatrixArray(:,:,1,1)
            StateIndices(1,1) = 1; 
            ExpectedRewardsArray(1,1) = ...
                squeeze(CriticWeightsArray(:,1,1))'*StatesList(:,StateIndices(1,1));
            CriticEligibilityTraces(:,1,1) = zeros([nElements,1,1]);
            StateActionPairEligibilityTraces(:,:,1,1) = zeros([nActions, nStates,1,1]);
            
        else
            
            % Set initial values for this episode to final values from last episode
            CriticWeightsArray(:,1,k) = CriticWeightsArray(:,nIterationsPerEpisode,k-1);
            ActionMatrixArray(:,:,1,k) = ActionMatrixArray(:,:,nIterationsPerEpisode,k-1);
            StateIndices(1,k) = StateIndices(nIterationsPerEpisode,k-1);
            %ExpectedRewardsArray(j,k) = ...
                %squeeze(CriticWeightsArray(StateIndices(j,k),:,j,k))*StatesList(:,StateIndices(j,k));   
                
            % Reinitialize EligibilityTraces
            CriticEligibilityTraces(:,1,k) = zeros([nElements,1,1]);
            StateActionPairEligibilityTraces(:,:,1,k) = zeros([nActions,nStates,1,1]); 
            
        end
    
        for j = 1:(nIterationsPerEpisode - 1)
            
            % Recompute ExpectedRewards after CriticWeights has been
            % updated during previous iteration
            ExpectedRewardsArray(j,k) = ...
                CriticWeightsArray(:,j,k)'*StatesList(:,StateIndices(j,k));  
            %ExpectedRewardsArray(j,k)
            
            %%%%%%%%%%%%%%%%%%%%% ACTOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Make Choice. 
            %squeeze(ActionMatrixArray(:,:,j,k))
            %StatesList(:,StateIndices(j,k))
            ActionValuesVec = squeeze(ActionMatrixArray(:,:,j,k))*StatesList(:,StateIndices(j,k));

            ProbabilitiesArray(:,j,k) = SoftMaxProbabilities(ActionValuesVec, beta_SoftMax);

            ChoiceIndex = Chooser(ProbabilitiesArray(:,j,k));
            ChoicesArray(j,k) = ActionsList(ChoiceIndex);
            
            % PERFORM ACTION.  Specific to the particular task the program
            % is simulating (i.e. maze-task, EMG-contingent stimulation, etc.)
            %NewStateStruct = StateTransformer(StatesList, ActionsList, StateIndices(j,k), ChoicesArray(j,k),...);
            NewStateStruct = MazeTaskEvaluator(StatesList, StateIndices(j,k), ChoicesArray(j,k));
            
            StateIndices(j+1,k) = NewStateStruct.NextStateIndex;
            RewardsArray(j,k) = NewStateStruct.Reward;
            EndOfTrial = NewStateStruct.EndOfTrial;
            
            %ActionMatrixArray(:,:,j,k) = ActionMatrix;
            
            %%%%%%%%%%%%%%%%%%%%%% CRITIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Evaluate expected reward for choice made at the end of the
            % previous iteration
            if (EndOfTrial == 0)
                
                ExpectedRewardsArray(j+1,k) = CriticWeightsArray(:,j,k)'...
                    *StatesList(:,StateIndices(j+1,k));
                %ExpectedRewardsArray(j+1,k)
                
            elseif (EndOfTrial == 1)
                
                ExpectedRewardsArray(j+1,k) = 0;
                
            end
            
            %%%%%%%%%% UPDATE CRITIC %%%%%%%%%%%%%%%
            % Calculate the scale of change to be made to critic weights using
            % temporal difference rule incorporating
            % CriticEligibilityTraces
            
            % Temporal difference learning rule
            DeltaArray(j,k) = RewardsArray(j,k) + gamma_discount*ExpectedRewardsArray(j+1,k) - ExpectedRewardsArray(j,k);
            %DeltaArray(j,k)
            
            CriticEligibilityTraces(StateIndices(j,k),j,k) = ...
                CriticEligibilityTraces(StateIndices(j,k),j,k) + ones([1,1,1]);
            
            % Update CriticWeights
            CriticWeightsArray(:,j+1,k) = CriticWeightsArray(:,j,k)...
                + epsilon_Critic*DeltaArray(j,k)*CriticEligibilityTraces(:,j,k);
            
            % Calcuate CriticEligibilityTrace on next iteration
            CriticEligibilityTraces(:,j+1,k) = ...
                gamma_discount*lambda_Critic*CriticEligibilityTraces(:,j,k);
            
            
            %%%%%%%%% UPDATE ACTOR %%%%%%%%%%%%%%%%% 
            % Calculate the updated ActionMatrix
            
            %%%% ReWork for TDLambda!!!
            for i = 1:nActions
        
                ActionMatrixArray(i,:,j+1,k) = ActionMatrixArray(i,:,j,k) + ...
                    epsilon_Actor*(KroneckerDelta(ChoiceIndex,i) - ProbabilitiesArray(i,j,k))...
                    *DeltaArray(j,k)*StateActionPairEligibilityTraces(i,:,j,k);
                
        
            end
            
            %ActionMatrixArray(:,:,j+1,k)
            
            % Add one to eligibility trace of current state-action pair 
            StateActionPairEligibilityTraces(ChoiceIndex,StateIndices(j,k),j,k) = ones([1,1,1,1]) +...
                StateActionPairEligibilityTraces(ChoiceIndex,StateIndices(j,k),j,k);
            
            % Calculate StateActionPairEligibilityTraces for next iteration
            StateActionPairEligibilityTraces(:,:,j+1,k) = lambda_Actor*gamma_discount*...
                StateActionPairEligibilityTraces(:,:,j,k);
            
        end
    end
    
    OutputStruct.StateIndices = StateIndices;
    OutputStruct.DeltaArray = DeltaArray;
    OutputStruct.CriticWeightsArray = CriticWeightsArray;
    OutputStruct.ActionMatrixArray = ActionMatrixArray;
    OutputStruct.ExpectedRewardsArray = ExpectedRewardsArray;
    OutputStruct.RewardsArray = RewardsArray;
    OutputStruct.ProbabilitiesArray = ProbabilitiesArray;
    OutputStruct.ChoicesArray = ChoicesArray;
    OutputStruct.CriticEligibilityTraces = CriticEligibilityTraces;
    OutputStruct.StateActionPairEligibilityTraces = StateActionPairEligibilityTraces;
    
            
end