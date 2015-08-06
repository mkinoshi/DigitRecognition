function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))): end), num_labels, (hidden_layer_size + 1));

%%set up parameters
m = size(X,1);
y_forDr = zeros(m,num_labels);
for i = 1:m
	y_forDr(i,Y(i,1)) = 1;
end
y_forJ = y_forDr;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
size(X);
size(Y);
size(Theta1);
size(Theta2);
X_forJ = [ones(m,1), X];
z2_forJ = X_forJ * Theta1';
a2_forJ = sigmoid(z2_forJ);
a2_forJ = [ones(m,1),a2_forJ];
z3_forJ = a2_forJ * Theta2';
a3_forJ = sigmoid(z3_forJ);
J = (1/m) * sum(sum((-y_forJ).* log(a3_forJ) - (1 - y_forJ).* log(1 - a3_forJ)));

%%regularized version
J = J + (lambda/(2 * m)) * (sum(sum(Theta1(:,2:(input_layer_size+1)).^2)) + sum(sum(Theta2(:,2:(hidden_layer_size+1)).^2)));

%%======================implement backpropagation==========================

y_forD = y_forDr;
for i = 1:m
	a1_forD  = X(i,:)';
	a1_forD = [1;a1_forD];
	y_forD = y_forDr(i,:)';
	z2_forD = Theta1 * a1_forD;
	a2_forD = sigmoid(z2_forD);
	a2_forD = [1;a2_forD];
	z3_forD = Theta2 * a2_forD;
	a3_forD = sigmoid(z3_forD);
	delta3 = a3_forD - y_forD;
	delta2 = Theta2' * delta3 .* sigmoidGradient([1;z2_forD]);
	dt2 = delta3 * a2_forD';
	dt1 = delta2(2:end) * a1_forD';
	Theta2_grad = Theta2_grad + dt2;
	Theta1_grad = Theta1_grad + dt1;
end
Theta1_grad = (1/m)*Theta1_grad;
Theta2_grad = (1/m)*Theta2_grad;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
grad = [Theta1_grad(:) ; Theta2_grad(:)];