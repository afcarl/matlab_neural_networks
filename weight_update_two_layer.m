function [W1,W2,bias1,bias2] = weight_update_two_layer(x,W1,W2,bias1,bias2,a1,a2,z1,z2,yhat,LEARNING_RATE)
% basically, we want to make z2 as close to yhat as possible
% x = feature vector
% W1, W2 = first and second weight matrices
% bias1, bias2 = first and second bias vectors
% z1, z2 = output of first and second layers (post-transfer function)
% yhat = target (desired) values, e.g. error deriv = yhat - z2
% LEARNING_RATE = scalar learning rate

d_error = yhat - z2; % derivative of l2 error

% d_error

% gradients (uncomment based on desired transfer function in
% vmm_compute_two_layer)
% =========

% ***** tanh *****
% z2_grad = 1 - z2 .* z2;
% z1_grad = 1 - z1 .* z1;

% ***** sigmoid *****
% z2_grad = z2 .* (1 - z2);
% z1_grad = z1 .* (1 - z1);

% ***** relu (max) *****
% z2_grad = (z2 > 0);
% z1_grad = (z1 > 0);

% ***** sinh-1 *****
% z2_grad = 1 ./ sqrt(a2 .* a2 + 1);
% z1_grad = 1 ./ sqrt(a1 .* a1 + 1);

% ***** neuron-like *****
z2_grad = 1 ./ (a2 .* a2 + 1);
z1_grad = 1 ./ (a1 .* a1 + 1);

% updates
l2_delta = d_error .* z2_grad;
l1_delta = (l2_delta * W2') .* z1_grad;

% update the weights
W2 = W2 + z1' * l2_delta * LEARNING_RATE / size(x, 1);
W1 = W1 + x' * l1_delta * LEARNING_RATE;

% update the biases
bias2 = bias2 + sum(l2_delta) * LEARNING_RATE / size(x, 1);
bias1 = bias1 + sum(l1_delta) * LEARNING_RATE;

end

