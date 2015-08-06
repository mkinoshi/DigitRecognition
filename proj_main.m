%%initialization
clear ; clear all; clc

%%set up the parameter
input_layer_size = 784; %28 x 28
hidden_layer_size = 25; %25 hidden units
num_label = 10;


%%===========================Loading and visualizing============================
%load training set
fprintf('loading and Visualizing Data ....\n')
data = load('train2.csv');
X = data(:,2:end);
y = data(:,1);
m = size(X,1);
%%set up y data(change label of 0 to 10)
Y = zeros(m,1);
for i =1:m
	if y(i,1) == 0
		Y(i,1) = 10;
	else
		Y(i,1) = y(i,1);
	end
end


%Randomly select 100 data to display
sel = randperm(size(X,1));
sel = sel(1:100);

displayData(X(sel, :));
fprintf('Program paused. Press enter to countinue\n');
pause;

%%=======================randomized weights==========================
Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_label);
initial_nn_params = [Theta1(:) ; Theta2(:)];
%%=======================Loading parameters==========================
#fprintf('\n feedforward using nural network\n')
#fprintf('\n cost function without regularization')
#lambda = 0;

#J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, #num_label, X, Y, lambda);

#fprintf('\n cost function without regularization')
#lambda = 1;
#J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, #num_label, X, Y, lambda);

%%========================Sigmoid Gradient =======================
#fprintf('\n evaluating sigmoid gradient')

#g = sigmoidGradient([1 -0.5 0 0.5 1])

%%========================train neural network=====================
fprintf('\n training neural network')

options = optimset('MaxIter', 10);
lambda = 1;

costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_label, X, Y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params(hidden_layer_size * (input_layer_size + 1) + 1:end), num_label, (hidden_layer_size + 1));

fprintf('\n Visualizing neural network w 10 iterations...')
displayData(Theta1(:, 2:end));
pause;

options = optimset('Maxiter', 40);
lambda = 1;
[nn_params, cost] = fmincg(costFunction, nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params(hidden_layer_size * (input_layer_size + 1) + 1:end), num_label, (hidden_layer_size + 1));
fprintf('\n Visualizing neural network w 50 iterations...')
displayData(Theta1(:, 2:end));
pause;

options = optimset('Maxiter', 50);
lambda = 1;
[nn_params, cost] = fmincg(costFunction, nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params(hidden_layer_size * (input_layer_size + 1) + 1:end), num_label, (hidden_layer_size + 1));
fprintf('\n Visualizing neural network w 50 iterations...')
displayData(Theta1(:, 2:end));
pause;

%%so far I iterated 100 times

%%========================================analyzing test data =====================================================
fprintf('\nloading test data set')
X_test = load('test2.csv');
m_test = size(X_test,1);
X_forJ_test = [ones(m_test,1), X_test];
z2_forJ_test = X_forJ_test * Theta1';
a2_forJ_test = sigmoid(z2_forJ_test);
a2_forJ_test = [ones(m_test,1),a2_forJ_test];
z3_forJ_test = a2_forJ_test * Theta2';
a3_forJ_test = sigmoid(z3_forJ_test);

n_test = size(a3_forJ_test,2)
y_test = zeros(m_test,1);
for i=1:m_test
	max_val = a3_forJ_test(i,1);
	m_col = 1;
	for j=1:n_test
		if a3_forJ_test(i,j) > max_val
			max_val = a3_forJ_test(i,j);
			m_col = j;
		end
	end
	if m_col == 10
		m_col = 0;
	end
	y_test(i,1) = m_col;
end
save train_result.txt y_test;