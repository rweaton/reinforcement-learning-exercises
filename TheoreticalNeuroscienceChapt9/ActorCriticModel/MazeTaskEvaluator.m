function OutputStruct = MazeTaskEvaluator(StatesList, StateIndex, Choice)

%StateVec = StateVec(:);

% MoveLeftStateXform = [0, 1, 1; 1, 0, 0; 0, 0, 0];
% MoveRightStateXform = [0, 1, 1; 0, 0, 0; 1, 0, 0];

% MoveLeftRewardXform = zeros([3,3]); MoveLeftRewardXform(3,3) = 2;
% MoveRightRewardXform = zeros([3,3]); MoveRightRewardXform(2,2) = 5;

%UpdateCriticTransform = [1, 1, 1];

switch StateIndex
    case  1

        if Choice == 'l'
           NextStateIndex = 2;
           Reward = 0;
           EndOfTrial = 0;
        end

        if Choice == 'r'
           NextStateIndex = 3;
           Reward = 0;
           EndOfTrial = 0;
        end

    case 2

        if Choice == 'l'
           NextStateIndex = 1;
           Reward = 0;
           EndOfTrial = 1;
        end

        if Choice == 'r'
           NextStateIndex = 1;
           Reward = 5;
           EndOfTrial = 1;
        end

    case 3

        if Choice == 'l'
            NextStateIndex = 1;
            Reward = 2;
            EndOfTrial = 1;
        end

        if Choice == 'r'
            NextStateIndex = 1;
            Reward = 0;
            EndOfTrial = 0;
        end
    
end
        

OutputStruct.PreviousChoice = Choice;
OutputStruct.NextStateVec = StatesList(:,NextStateIndex);
OutputStruct.NextStateIndex = NextStateIndex;
OutputStruct.Reward = Reward;
OutputStruct.EndOfTrial = EndOfTrial;


end