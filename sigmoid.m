function z = sigmoid(value)
z = 1.0 ./ (1.0 + exp(-value));
end