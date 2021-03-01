function OutputStruct = StateEvaluator(StateVec, Choice)

StateVec = StateVec(:);

MoveLeftStateXform = [0, 1, 1; 1, 0, 0; 0, 0, 0];
MoveRightStateXform = [0, 1, 1; 0, 0, 0; 1, 0, 0];

MoveLeftRewardXform = zeros([3,3]); MoveLeftRewardXform(3,3) = 2;
MoveRightRewardXform = zeros([3,3]); MoveRightRewardXform(2,2) = 5;

%UpdateCriticTransform = [1, 1, 1];

    if Choice == 'l'
       NextStateVec = MoveLeftStateXform*StateVec;
       NextStateIndex = find(NextStateVec == 1);
       Reward = sum(MoveLeftRewardXform*StateVec);
    end

    if Choice == 'r'
        NextStateVec = MoveRightStateXform*StateVec;
        NextStateIndex = find(NextStateVec == 1);
        Reward = sum(MoveRightRewardXform*StateVec);
    end
    
    EndOfTrial = 0;
    if ([1, 0, 0]*StateVec ~= 1)
        EndOfTrial = 1;
    end
    
%UpdateCritic = UpdateCriticTransform*NextStateVec;

OutputStruct.PreviousChoice = Choice;
OutputStruct.NextStateVec = NextStateVec;
OutputStruct.NextStateIndex = NextStateIndex;
OutputStruct.Reward = Reward;
%OutputStruct.UpdateCritic = UpdateCritic;
OutputStruct.EndOfTrial = EndOfTrial;


end