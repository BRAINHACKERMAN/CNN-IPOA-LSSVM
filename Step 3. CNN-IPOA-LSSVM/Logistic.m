function X=Logistic(N,dim,ub,lb)
X=rand(N,dim);
k=4;  %»ìãç²ÎÊý
for i=1:N
    for j=1:dim-1
            X(i,j+1)=k*(1-X(i,j))*X(i,j);
    end
end
X = X.*(ub-lb)+lb;
end