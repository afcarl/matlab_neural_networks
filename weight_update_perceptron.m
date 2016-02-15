function [new_W,new_bias,error] = weight_update_perceptron(x,yout,yhat,W,bias)
% perceptron learning rule: useful for multi-class classification problems
% given some binary classifications

LEARNING_RATE = 0.01;

% perceptron learning rule: gradient decent
deltas = yhat-yout;
new_W = W + x' * (deltas * LEARNING_RATE) / size(deltas, 1);
new_bias = bias + sum(deltas) * LEARNING_RATE / size(deltas, 1);

% error term: single number
error = sum(sum(abs(deltas))) / size(deltas,1);

end

