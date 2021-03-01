function OutputArray = StochasticRewardsGenerator(ProbVec,nIterations)

nActions = length(ProbVec);

OutputArray = NaN*ones([nIterations, nActions]);

for i = 1:nActions
    
    OutputArray(:,i) = UnaryMonteCarlo(ProbVec(i)*ones([nIterations,1]));
    
end

end