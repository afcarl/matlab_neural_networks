function [bias, wta, err] = learn_wta_bias(output, target, INCREMENT, MAX_C, VERBOSE)
% finds a good bias value to make the wta part work
% assumes wta bias is linearly related to the output (which it should be if
% the neural network classified the data well)
% output = output of last layer of network
% target = desired output (after wta)
% INCREMENT = amount to change bias on each iteration
% MAX_C = max count, maximum number of times to run the algorithm

% first, sort by increasing values of "target" (for the WTA part)
[~, id] = sort(target);
output = output(id);
target = target(id);

% initialize bias and delta
bias = 0;
wta = compute_wta(output', bias)';
delta = sum(target - wta);
last_delta = abs(delta) + 1;
err = abs(delta);

% calculate new delta
for i=1:MAX_C
    if (sum(abs(delta)) >= sum(abs(last_delta)) || sum(delta) == 0)
        break;
    end
    
    if (nargin > 4 && VERBOSE)
        disp('delta:');
        disp(abs(delta));
    end
    
    last_delta = delta;
    bias = bias + sum(delta) * INCREMENT;
    wta = compute_wta(output', bias)';
    delta = sum(target - wta);
    err = abs(delta);
end

end