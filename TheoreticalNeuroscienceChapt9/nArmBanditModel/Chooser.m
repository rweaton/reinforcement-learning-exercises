function Output = Chooser(ProbVec)

% NOTE: The sum of ProbVec must equal 1.

ProbVec = ProbVec(:);
nChoices = length(ProbVec);

ChoiceDomains = cumsum(ProbVec);
ChoiceDomains = [0; ChoiceDomains];

Seed = rand(1);

for i = 1:nChoices
   
    if (Seed > ChoiceDomains(i))&(Seed <= ChoiceDomains(i+1))
        Output = i;
    end
end
