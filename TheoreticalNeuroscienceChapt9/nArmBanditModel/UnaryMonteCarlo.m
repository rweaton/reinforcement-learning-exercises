function OutputVec = UnaryMonteCarlo(ProbVec)

SeedVec = rand(size(ProbVec));

OutputVec = (SeedVec < ProbVec);

end