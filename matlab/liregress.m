x_fid = fopen('x.dat', 'rb');
y_fid = fopen('y.dat', 'rb');
theta_fid = fopen('theta.dat', 'rb');

x = fread(x_fid, [10, 8], 'float');
y = fread(y_fid, [10, 8], 'float');
theta = fread(theta_fid, [8, 8], 'float');

fclose(x_fid);
fclose(y_fid);
fclose(theta_fid);

alpha = 0.5;
epoch = 2;

for(i = 1:epoch)
  error = x * theta - y
  theta = theta - alpha * x' * error;
end

theta
