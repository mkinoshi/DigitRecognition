function W = randInitializeWeights(L_in, L_out)

W = zeros(L_out, L_in + 1);
eplison = sqrt(6) / sqrt(L_in + L_out)
W = rand(L_out, L_in + 1) * 2 * eplison - eplison;

end;