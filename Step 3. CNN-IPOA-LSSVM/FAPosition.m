% ”©ª≥Ê»≈∂Ø
function [OutPosition]=FAPosition(dim,lb, ub,position,BestPosition)
dmax = (lb-ub)*sqrt(dim);
%æ‡¿Î◊Ó”≈Œª÷√æ‡¿Î
rij=norm(position - BestPosition)./dmax;

alpha=0.2;
beta0=2; 
m=2;
gamma=1;
beta=beta0*exp(-gamma*rij.^m);  
e=rand(1,dim)-1/2;
OutPosition= position+ beta.*(position- BestPosition)+ alpha.*e;

end

