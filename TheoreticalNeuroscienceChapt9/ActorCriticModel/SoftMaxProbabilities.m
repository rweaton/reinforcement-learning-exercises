function ProbVec = SoftMaxProbabilities(ActionValuesVec, Beta)

ProbVec = exp(Beta*ActionValuesVec)/sum(exp(Beta*ActionValuesVec));

end
    
     