%%%% DEPRECATED %%%%
% see learn_simple_xor or learn_song_data for more up-to-date versions

% parameters for the model
n_batches = 1;      % number of minibatches to train on
n_epochs = 100;     % number of epochs to train
n_train = 1;
n_neurons = 300;     % number of neurons
n_outputs = 1;

% This bit is specific to the sound dataset
idx = [find(yhat(:,1) ~= 0, 1000); find(yhat(:,1) == 0 & yhat(:,2) == 0 & yhat(:,3) == 0, 1000)];
x_data = x(idx, :);
yhat_data = yhat(idx, 1);

% Initialize the two sets of weights (two-layer neural network)
W1 = rand(size(x_data,2), n_neurons) - 0.5;           % randomly initialize weights between -0.5 and +0.5 (to learn)
bias1 = rand(1, n_neurons) - 0.5;     % set the bias randomly as well (really small though)

W2 = rand(n_neurons, n_outputs) - 0.5;
bias2 = rand(1, n_outputs) - 0.5;

% track the error in each batch over time
clear errorTrack;
errorTrack = zeros(n_epochs,n_batches);

LEARNING_RATE = 1;

for k=1:n_train;
    for l=1:n_epochs
        % randomly shuffle data between trials
        perm = randperm(size(x_data,1));
        x_data = x_data(perm,:);
        yhat_data = yhat_data(perm,:);

        [a1,z1,a2,z2,wta] = vmmwta_compute_two_layer(x_data,W1,W2,bias1,bias2);

        % train as minibatches
        inc = floor(size(x_data,1)/n_batches)-1;
        for j=1:n_batches
            i = inc * (j-1) + 1;

            x_batch = x_data(i:i+inc, :);
            a1_batch = a1(:, i:i+inc);
            z1_batch = z1(:, i:i+inc);
            a2_batch = a2(:, i:i+inc);
            z2_batch = z2(:, i:i+inc);
            wta_batch = wta(:, i:i+inc);
            yhat_batch = yhat_data(i:i+inc, :);

            [W1,W2,bias1,bias2] = weight_update_two_layer(x_batch,W1,W2,bias1,bias2,z1_batch,z2_batch,wta_batch,yhat_batch,LEARNING_RATE);

%             LEARNING_RATE = LEARNING_RATE * 0.9999;
        end
    end
    
    [a1,z1,a2,z2,wta] = vmmwta_compute_two_layer(x,W1,W2,bias1,bias2);
        
    figure;
    a(1) = subplot(4,1,1);
    plot(yhat);
    a(2) = subplot(4,1,2);
    plot(x);
    a(3) = subplot(4,1,3);
    plot(z2');
    a(4) = subplot(4,1,4);
    plot(wta');
    linkaxes(a,'xy');
    zoom xon;

    d_error = yhat - z2';
    batch_error = sum(sum((d_error .* d_error) / 2)); % this is the error we're trying to minimize

    disp(batch_error);
end;

% figure;
% plot(errorTrack);