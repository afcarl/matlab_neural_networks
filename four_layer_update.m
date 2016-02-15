function [output,W1,W2,W3,W4,bias1,bias2,bias3,bias4] = four_layer_update(x,yhat,W1,W2,W3,W4,bias1,bias2,bias3,bias4,LEARNING_RATE,LEARN)

% first layer
a1 = x * W1;
a1 = a1 + repmat(bias1, [size(a1,1), 1]); % bias
z1 = max(a1, 0); % rectified output of a1

% second layer
a2 = z1 * W2;
a2 = a2 + repmat(bias2, [size(a2,1), 1]); % bias
z2 = max(a2, 0); % rectified output of a2

% third layer
a3 = z2 * W3;
a3 = a3 + repmat(bias3, [size(a2,1), 1]); % bias
z3 = max(a3, 0); % rectified output of a2

% fourth layer
a4 = z3 * W4;
a4 = a4 + repmat(bias4, [size(a2,1), 1]); % bias
z4 = max(a4, 0); % rectified output of a2

output = z4;

if LEARN
    % d_error
    d_error = yhat - z4; % derivative of l2 error

    % gradients (uncomment based on desired transfer function in
    % vmm_compute_two_layer)
    % =========

    % ***** relu (max) *****
    z4_grad = (z4 > 0);
    z3_grad = (z3 > 0);
    z2_grad = (z2 > 0);
    z1_grad = (z1 > 0);

    % updates
    l4_delta = d_error .* z4_grad;
    l3_delta = (l4_delta * W4') .* z3_grad;
    l2_delta = (l3_delta * W3') .* z2_grad;
    l1_delta = (l2_delta * W2') .* z1_grad;
    
    % clip gradients
    GRAD_CLIP = 10;
    l4_delta = max(min(l4_delta, GRAD_CLIP), -GRAD_CLIP);
    l3_delta = max(min(l3_delta, GRAD_CLIP), -GRAD_CLIP);
    l2_delta = max(min(l2_delta, GRAD_CLIP), -GRAD_CLIP);
    l1_delta = max(min(l1_delta, GRAD_CLIP), -GRAD_CLIP);

    % update the weights
    W4 = W4 + z3' * l4_delta * LEARNING_RATE / size(x, 1);
    W3 = W3 + z2' * l3_delta * LEARNING_RATE / size(x, 1);
    W2 = W2 + z1' * l2_delta * LEARNING_RATE / size(x, 1);
    W1 = W1 + x' * l1_delta * LEARNING_RATE / size(x, 1);

    % update the biases
    bias4 = bias4 + sum(l4_delta) * LEARNING_RATE / size(x, 1);
    bias3 = bias3 + sum(l3_delta) * LEARNING_RATE / size(x, 1);
    bias2 = bias2 + sum(l2_delta) * LEARNING_RATE / size(x, 1);
    bias1 = bias1 + sum(l1_delta) * LEARNING_RATE / size(x, 1);
end

end

