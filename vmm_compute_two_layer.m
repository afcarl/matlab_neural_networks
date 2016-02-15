function [z1,z2,a1,a2] = vmm_compute_two_layer(x,W1,W2,bias1,bias2)
% gist: perform VMM on a1, rectify output, then VMM with result and second
% set of weights. the number of neurons is dependent on the choice of
% weight. i think this is feasible on hardware... the use of a rectified
% unit instead of a tanh transfer function is semi-common in the neural
% network community. rectified linear unit -> "relu" = max(in, 0).
% apparently it tends to produce sparse weights although i don't know much
% about the theory behind it

% if you change the transfer function here, be sure to use the appropriate
% transfer function in "weight_update_two_layer" as well, otherwise it
% won't learn correctly

% first layer
a1 = x * W1;
a1 = a1 + repmat(bias1, [size(a1,1), 1]); % bias

% activation function
z1 = max(a1, 0); % rectified output of a1
% z1 = tanh(a1);
% z1 = 1 ./ (1 + exp(-a1));

% second layer
a2 = z1 * W2;
a2 = a2 + repmat(bias2, [size(a2,1), 1]); % bias

% activation function
% z2 = max(a2, 0); % rectified output of a2
z2 = tanh(a2);
% z2 = 1 ./ (1 + exp(-a2));

end