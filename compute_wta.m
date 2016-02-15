function [wta] = compute_wta(output, bias)
% computes winner-take-all winner (solves ODE given output and bias)
% output = output of vmm part (input into wta part)
% bias = some linear addition to output (should be a scalar)

% bias is just added to output
output = output + bias;

[wid, len] = size(output);
all_ones = ones(1,wid);

% initialize some components
a = zeros(wid,len); 
wta = zeros(wid,len);
a(:,1) = all_ones;
z = wid;

r = 0.001;                  % sets exp of the bias voltage for Va
athreshold = 0.51;          % set for one winning node

% compute ODE solution for WTA part
for k=1:len;
    q1 = a(:,k) .* (output(:,k) - (z(k) * all_ones'));
    q2 = z(k) * r * all_ones';
    
    a(:,k+1) = a(:,k) + 0.1 * (q1 + q2);
    z(k+1) = 0.9 * z(k) + 0.1 * sum(a(:,k+1));
    
    wta(:,k+1) = 0.5 * sign((a(:,k) / z(k)) - athreshold) + 0.5;
end;

% chop off the first bit
wta = wta(:,2:end);

end