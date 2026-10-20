%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function error = fun(x,inputn_train,output_train,inputn_test,output_test)

bestgam = x(1);
bestsig2 = x(2);
model = initlssvm(inputn_train,output_train,'c',[],[],'RBF_kernel');
%实现多分类，要对下列参数进行设置！
model.code = 'changed';
model.codetype = 'code_OneVsOne';
model.gam = bestgam*ones(1,45);  %这里的45是原始LSSVM在寻优gam和kernel_pars时，自己设定的参数，这个可以不必在意！
model.kernel_pars = bestsig2*ones(1,45);
model = trainlssvm(model);
an = simlssvm(model,inputn_test);
accuracy=sum(an==output_test)/length(output_test);   %计算预测的确率
error=1-accuracy;   
end





