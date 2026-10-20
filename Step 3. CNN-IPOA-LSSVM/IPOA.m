% 鹈鹕优化算法(Pelican Optimization Algorithm,POA)是2022年由Pavel Trojovsk?和Mohammad Dehghani 提出的，
% 该算法模拟了鹈鹕在狩猎过程中的自然行为。
function[Best_score,Best_pos,POA_curve]=IPOA(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,fitness)

if size(lowerbound,2)==1
    lowerbound=ones(1,dimension).*(lowerbound);   % 当所用的边界参量为1维时，将其扩展成dim维
    upperbound=ones(1,dimension).*(upperbound);   % 寻优上界和下界
end

%% 改进点1 种群初始化
X=Logistic(SearchAgents,dimension,upperbound,lowerbound);   %% 种群初始化

%% L、fit用于保存当前的鹈鹕的位置和适应度
for i =1:SearchAgents
    L=X(i,:);
    fit(i)=fitness(L);
end

%% 主循环
for t=1:Max_iterations
    t  
    %% 改进点2：引入反向差分进化机制
    [X,fit]=ODE(X,fit',upperbound,lowerbound,fitness);
    fit=fit';
    %% 更新当前最优解的情况
    [best , location]=min(fit);
    if t==1
        Xbest=X(location,:);
        fbest=best;
    elseif best<fbest  %best：全局最优解   fbest：表示局部最优解
        fbest=best;
        Xbest=X(location,:);
    end

    %% 更新当前食物的位置信息
    
    X_FOOD=[];
    k=randperm(SearchAgents,1); % 从种群中随机抽选一只作为食物源（这里可以考虑采用灰狼+轮盘法选取优势种群）
    X_FOOD=X(k,:);
    F_FOOD=fit(k);
    %%
    for i=1:SearchAgents
        
        %% 第一阶段：逼近猎物( 探索阶段 )
        I=round(1+rand(1,1)); %为1或者2的随机整数
        if fit(i)> F_FOOD
            X_new=X(i,:)+ rand(1,1).*(X_FOOD-I.* X(i,:));
        else
            X_new=X(i,:)+ rand(1,1).*(X(i,:)-1.*X_FOOD);
        end
        % 防止参数越界
        X_new= max(X_new,lowerbound);
        X_new = min(X_new,upperbound);
        
        % 更新当前解的情况
        f_new = fitness(X_new);
        if f_new <= fit (i)
            X(i,:) = X_new;
            fit (i)=f_new;
        end
        
        %% 第二阶段：水面飞行（开发阶段）
        % R为0或者2的随机整数，但本代码取0.2
        X_new=X(i,:)+0.2*(1-t/Max_iterations).*(2*rand(1,dimension)-1).*X(i,:);
        % 防止参数越界
        X_new= max(X_new,lowerbound);X_new = min(X_new,upperbound);
        % 位置更新
        f_new = fitness(X_new);
        if f_new <= fit (i)
            X(i,:) = X_new;
            fit (i)=f_new;
        end
    end
    %% 改进点3 萤火虫扰动
    for j = 1:SearchAgents
        Xhf = FAPosition(dimension,lowerbound, upperbound,X(j,:),Xbest);
        % 边界处理
        Flag4ub = Xhf>upperbound;
        Flag4lb = Xhf<lowerbound;
        Xhf=Xhf.*(~(Flag4ub+Flag4lb))+upperbound.*Flag4ub+lowerbound.*Flag4lb;
        temf_f=fitness(Xhf);
        if(temf_f<fit(j))
            X(j,:) = Xhf;
            fit(j) = temf_f;
        end
    end
    % 记录当前最优解和平均解
    best_so_far(t)=fbest;
    
end
Best_score=fbest;
Best_pos=Xbest;
POA_curve=best_so_far;
end

