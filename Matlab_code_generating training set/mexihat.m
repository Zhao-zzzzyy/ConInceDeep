function [out1,out2] = mexihat(LB,UB,N,flagGUI)
% Compute values of the Lorentz4 wavelet.

out2 = linspace(LB,UB,N);     
out1 = out2.^2;
out1 = 24*(5*(out1.*out1)-10*out1+1)./((out1+1).*(out1+1).*(out1+1).*(out1+1).*(out1+1));

