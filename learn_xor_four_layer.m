%% parameters for the model

n_epochs = 200;     % number of epochs to train
n_neurons = [19,17,13];     % number of neurons
n_outputs = 1;      % one output (binary classification problem)

%% generate our training set

% learn a complex function (like a circle with a hole in it)
N_POINTS = 1000;
x_data = rand(N_POINTS, 2); % 2-dimensional data
y_hat = x_data(:,1) + x_data(:,2) > 0.3 ...
        & x_data(:,1) + x_data(:,2) < 1.7 ...
        & x_data(:,1) - x_data(:,2) < 0.7 ...
        & x_data(:,2) - x_data(:,1) < 0.7 ...
        & (x_data(:,2) - x_data(:,1) > 0.3 | x_data(:,1) - x_data(:,2) > 0.3 | x_data(:,1) + x_data(:,2) < 0.7 | x_data(:,1) + x_data(:,2) > 1.3);
% y_hat = (x_data(:,1)-0.5) .* (x_data(:,1)-0.5) + (x_data(:,2)-0.5) .* (x_data(:,2)-0.5) < 0.25 ...
%         & (x_data(:,1)-0.5) .* (x_data(:,1)-0.5) + (x_data(:,2)-0.5) .* (x_data(:,2)-0.5) > 0.05;
% y_hat = x_data(:,1) + x_data(:,2) < 0.5 ...
%         | x_data(:,1) + x_data(:,2) > 1.5;

% positive and negative training examples
x_pos = x_data(y_hat,:);
x_neg = x_data(~y_hat,:);

% plot the data: red is positive, blue is negative
PLOT = 1;
if PLOT
    figure;
    plot(x_pos(:,1), x_pos(:,2), 'ro');
    hold on;
    plot(x_neg(:,1), x_neg(:,2), 'bo');
    hold off;
end

%% construct the network itself

% Initialize the two sets of weights (two-layer neural network)
W1 = rand(size(x_data,2), n_neurons(1)) * 2 - 1;
bias1 = rand(1, n_neurons(1)) * 2 - 1;

W2 = rand(n_neurons(1), n_neurons(2)) * 2 - 1;
bias2 = rand(1, n_neurons(2)) * 2 - 1;

W3 = rand(n_neurons(2), n_neurons(3)) * 2 - 1;
bias3 = rand(1, n_neurons(3)) * 2 - 1;

W4 = rand(n_neurons(3), n_outputs) * 2 - 1;
bias4 = rand(1, n_outputs) * 2 - 1;

%  train the network

LEARNING_RATE = 0.01;
batch_error = numel(y_hat);

% can adjust this value based on how the model seems to be converging
while batch_error > 0.05
    batch = randperm(N_POINTS, 200);
    x_batch = x_data(batch, :);
    y_batch = y_hat(batch, :);
    
    for l=1:n_epochs
        [~,W1,W2,W3,W4,bias1,bias2,bias3,bias4] = four_layer_update(x_batch,y_batch,W1,W2,W3,W4,bias1,bias2,bias3,bias4,LEARNING_RATE,1);
    end
    
    [output] = four_layer_update(x_data,y_hat,W1,W2,W3,W4,bias1,bias2,bias3,bias4,LEARNING_RATE,0);
    
    % l2 error
    batch_error = sum(sum((y_hat - output) .* (y_hat - output))) / numel(y_hat);
    fprintf('total error: %d learning rate: %d\n', batch_error, LEARNING_RATE);
    
    LEARNING_RATE = LEARNING_RATE * 0.99;
end

%% wta part (sort of works... wta's time component is a bit hairy)

% basically learns what bias will allow the wta part to work correctly
% since the wta part is time-dependent it sorts the z2's in increasing
% order (uncomment last line to graph result idea)

[output] = four_layer_update(x_data,y_hat,W1,W2,W3,W4,bias1,bias2,bias3,bias4,LEARNING_RATE,0);
[bias, wta, batch_error] = learn_wta_bias(output, y_hat, 0.001, 10000, 1);
disp(batch_error);

plot(sort(output)); hold on; plot(wta);

%% graph result of classification for comparison

figure;
clear ax;

% plot the original data: red is positive, blue is negative
ax(1) = subplot(2,3,1);
plot(x_pos(:,1), x_pos(:,2), 'ro');
hold on;
plot(x_neg(:,1), x_neg(:,2), 'bo');
hold off;
title('Desired classifications');

[output] = four_layer_update(x_data,y_hat,W1,W2,W3,W4,bias1,bias2,bias3,bias4,LEARNING_RATE,0);

% plot the learned data: red is positive, blue is negative (decision
% boundary is 0.5)
pos = output > 0.5;
train_pos = x_data(pos,:);
train_neg = x_data(~pos,:);

ax(2) = subplot(2,3,2);
plot(train_pos(:,1), train_pos(:,2), 'ro');
hold on;
plot(train_neg(:,1), train_neg(:,2), 'bo');
hold off;
title('Learned classifications');

% plot the differences
diff_pos = pos == 1 & y_hat == 0;
diff_neg = pos == 0 & y_hat == 1;
x_diff_pos = x_data(diff_pos,:);
x_diff_neg = x_data(diff_neg,:);

ax(3) = subplot(2,3,4);
plot(x_diff_pos(:,1), x_diff_pos(:,2), 'ro');
hold on;
plot(x_diff_neg(:,1), x_diff_neg(:,2), 'bo');
hold off;
title('Incorrect classifications');

% plot the contour
VERT = 100;
HORI = 100;
x_test = [repmat(linspace(0,1,VERT),[1,HORI]); sort(repmat(linspace(0,1,HORI),[1,VERT]))]';
[output] = four_layer_update(x_test,y_hat,W1,W2,W3,W4,bias1,bias2,bias3,bias4,LEARNING_RATE,0);
pos = output > 0.5;
train_pos = x_test(pos,:);
train_neg = x_test(~pos,:);

ax(4) = subplot(2,3,5);
plot(train_pos(:,1), train_pos(:,2), 'r.', 'markersize', 15);
hold on;
plot(train_neg(:,1), train_neg(:,2), 'b.', 'markersize', 15);
hold off;
title('Decision boundaries');

linkaxes(ax, 'xy');

subplot(1,3,3);
surf(x_test(:,1), x_test(:,2), output);