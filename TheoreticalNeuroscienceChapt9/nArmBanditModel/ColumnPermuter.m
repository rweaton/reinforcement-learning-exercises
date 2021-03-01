function OutputArray = ColumnPermuter(InputArray, nRowsPerChunk)

[nIterations, nColumns] = size(InputArray);

OutputArray = NaN*ones([nIterations, nColumns]);

nChunks = floor(nIterations/nRowsPerChunk);

%nRowsInLastChunk = mod(nIterations, nRowsPerChunk);

for i = 1:nChunks

    OutputArray(((i-1)*nRowsPerChunk+1):(i*nRowsPerChunk), randperm(nColumns)) = ...
        InputArray(((i-1)*nRowsPerChunk+1):(i*nRowsPerChunk),:);
    
end

if (nChunks ~= 0)
    OutputArray((nChunks*nRowsPerChunk+1):nIterations, randperm(nColumns)) = ...
        InputArray((nChunks*nRowsPerChunk+1):nIterations, :);
    
end

end