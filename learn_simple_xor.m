%% parameters for the model

n_epochs = 100;     % number of epochs to train
n_neurons = 10;     % number of neurons
n_outputs = 1;      % one output (binary classification problem)

%% generate our training set

% learn an xor function
N_POINTS = 1000;a 
x_data = rand(N_POINTS, 2); % 2-dimensional data
y_hat = x_data(:,1) + x_data(:,2) > 0.7 & x_data(:,1) + x_data(:,2) < 1.3;

% positive and negative training examples
x_pos = x_data(y_hat,:);
x_neg = x_data(~y_hat,:);

% plot the data: red is positive, blue is negative
PLOT = 0;
if PLOT
    figure;
    plot(x_pos(:,1), x_pos(:,2), 'ro');
    hold on;
    plot(x_neg(:,1), x_neg(:,2), 'bo');
    hold off;
end

%% construct the network itself

% Initialize the two sets of weights (two-layer neural network)
W1 = rand(size(x_data,2), n_neurons) * 2 - 1;           % randomly initialize weights between -0.5 and +0.5 (to learn)
bias1 = rand(1, n_neurons) * 2 - 1;     % set the bias randomly as well

W2 = rand(n_neurons, n_outputs) * 2 - 1;
bias2 = rand(1, n_outputs) * 2 - 1;

%% train the network

LEARNING_RATE = 0.01;
batch_error = numel(y_hat);

% can adjust this value based on how the model seems to be converging
while batch_error > 30
    batch = randperm(N_POINTS, 200);
    x_batch = x_data(batch, :);
    y_batch = y_hat(batch, :);
    
    for l=1:n_epochs
        [z1,z2,a1,a2] = vmm_compute_two_layer(x_batch,W1,W2,bias1,bias2);
        [W1,W2,bias1,bias2] = weight_update_two_layer(x_batch,W1,W2,bias1,bias2,a1,a2,z1,z2,y_batch,LEARNING_RATE);
    end
    
    [~,z2] = vmm_compute_two_layer(x_data,W1,W2,bias1,bias2);
    
    % l2 error without dividing by the number of samples
    batch_error = sum((y_hat - z2) .* (y_hat - z2));
    fprintf('total error: %d\n', batch_error);
end

%% wta part (sort of works... wta's time component is a bit hairy)

% basically learns what bias will allow the wta part to work correctly
% since the wta part is time-dependent it sorts the z2's in increasing
% order (uncomment last line to graph result idea)

[~,z2] = vmm_compute_two_layer(x_data,W1,W2,bias1,bias2);
[bias, wta, batch_error] = learn_wta_bias(z2, y_hat, 0.001, 10000, 1);
disp(batch_error);

plot(sort(z2)); hold on; plot(wta);

%% graph result of classification for comparison

figure;
clear ax;

% plot the original data: red is positive, blue is negative
ax(1) = subplot(3,2,1);
plot(x_pos(:,1), x_pos(:,2), 'ro');
hold on;
plot(x_neg(:,1), x_neg(:,2), 'bo');
hold off;
title('Desired classifications');

[z1,z2] = vmm_compute_two_layer(x_data,W1,W2,bias1,bias2);

% plot the learned data: red is positive, blue is negative (decision
% boundary is 0.5)
pos = z2 > 0.5;
train_pos = x_data(pos,:);
train_neg = x_data(~pos,:);

ax(2) = subplot(3,2,3);
plot(train_pos(:,1), train_pos(:,2), 'ro');
hold on;
plot(train_neg(:,1), train_neg(:,2), 'bo');
hold off;
title('Learned classifications');

% plot the differences
diff = pos ~= y_hat;
x_diff = x_data(diff,:);

ax(3) = subplot(3,2,5);
plot(x_diff(:,1), x_diff(:,2), 'go');
title('Incorrect classifications');

linkaxes(ax, 'xy');

% plot 3d
subplot(1,2,2);
plot3(x_data(:,1), x_data(:,2), z2, 'o');
