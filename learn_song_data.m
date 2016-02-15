%% parameters for the model

n_batches = 10;      % number of minibatches to train on
n_epochs = 5;     % number of epochs to train
n_train = 10;        % number of cycles to repeat
n_neurons = [50, 50];     % number of hidden neurons
n_outputs = 3;      % three outputs (three song types)

%% generate our training set

% need to change directories to parent directory to access these functions
load('classifier_data_train_01.mat')
[time,x_data,y_hat] = create_training_dataset01(quiet_input01,generator_input01,car_idle_input01,truck_idle_input01);

% classes
x_c1 = x_data(y_hat(:,1) ~= 0, :);
x_c2 = x_data(y_hat(:,2) ~= 0, :);
x_c3 = x_data(y_hat(:,3) ~= 0, :);
x_neg = x_data(y_hat(:,1) == 0 & y_hat(:,2) == 0 & y_hat(:,3) == 0, :);

%% visualize training set

% plot the data: red is positive, blue is negative
PLOT = 1;
if PLOT
    figure;
    plot(time, x_data);
    hold on;
    plot(time, y_hat);
    hold off;
end

%% construct the network itself

% Initialize the two sets of weights (two-layer neural network)
W1 = rand(size(x_data,2), n_neurons(1)) * 2 - 1;           % randomly initialize weights between -0.5 and +0.5 (to learn)
bias1 = rand(1, n_neurons(1)) * 2 - 1;     % set the bias randomly as well

W2 = rand(n_neurons(1), n_neurons(2)) * 2 - 1;
bias2 = rand(1, n_neurons(2));

W3 = rand(n_neurons(2), n_outputs) * 2 - 1;
bias3 = rand(1, n_outputs) * 2 - 1;

% train the network

% some thoughts on this: the training doesn't always converge, so
% reinitializing the weights and running again might help, although the
% song data has generally been tricky. can stop at any time, but will stop
% automatically when the batch error goes below a certain threshold (which
% it probably won't, given previous behavior)

LEARNING_RATE = 0.001;
batch_error = numel(y_hat);

while sum(batch_error) > 4
    batch = randperm(size(x_data, 1), 200);
    x_batch = x_data(batch, :);
    y_batch = y_hat(batch, :);
    
    for l=1:n_epochs
        [z1,z2] = vmm_compute_two_layer(x_batch,W1,W2,bias1,bias2);
        [W1,W2,bias1,bias2] = weight_update_two_layer(x_batch,W1,W2,bias1,bias2,z1,z2,y_batch,LEARNING_RATE);
    end
    
    [z1,z2,a1,a2] = vmm_compute_two_layer(x_batch,W1,W2,bias1,bias2);
    
    batch_error = sum((y_batch - z2) .* (y_batch - z2));
    disp(batch_error);
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

% plot the original data: red is positive, blue is negative
ax(1) = subplot(3,1,1);
plot(time, x_data);
hold on;
plot(time, y_hat);
hold off;
title('Desired classifications');

[z1,z2] = vmm_compute_two_layer(x_data,W1,W2,bias1,bias2);

% plot the learned data: red is positive, blue is negative (decision
% boundary is 0.5)
pos = z2 > 0.5;
ax(2) = subplot(3,1,2);
plot(time, z2);
title('Learned classifications');

% plot the differences
diff = pos ~= y_hat;
ax(3) = subplot(3,1,3);
plot(time, diff);
title('Incorrect classifications');

linkaxes(ax, 'xy');
