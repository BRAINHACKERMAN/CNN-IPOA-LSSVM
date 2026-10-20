function [Moth_pos,Moth_fitness]=ODE(Moth_pos,Moth_fitness,ub,lb,fobj)
    [N,D] = size(Moth_pos);
    O_moth_fitness = zeros(1,N); %反向种群适应度
    Moth_new = zeros(2*N,D);  
    V_pop = zeros(2*N,D);    
    O_Moth_pos=ub+lb-Moth_pos;  %反向种群位置 
    for i=1:N
        O_moth_fitness(i)=fobj(O_Moth_pos(i,:));   %初始适应度计算
    end
    O_moth_fitness=reshape(O_moth_fitness,N,1);     %转换格式
    G_OBL_fitness=[O_moth_fitness;Moth_fitness];   %差分种群适应度 
    G_OBL_population=[O_Moth_pos;Moth_pos];   %差分种群位置 
    [~,I]=sort(G_OBL_fitness);         %排序
    Best_flame_pos=G_OBL_population(I(1),:);    
    %% DE
    for i=1:2*N
        
        index = randperm(2*N,2);    
        Moth_new(i,:) = G_OBL_population(i,:) + rand(1,D).*(G_OBL_population(index(1),:)-G_OBL_population(index(2),:));%当前解与领域解的差异
        
        V_pop(i,:) = Moth_new(i,:);
        for j = 1:D
            if rand > 0.5  %rand产生0-1随机数 随机位置更新
                V_pop(i,j) = G_OBL_population(i,j);
            end
        end
        
        Flag4ub=V_pop(i,:)>ub; %越界处理
        Flag4lb=V_pop(i,:)<lb; 
        V_pop(i,:)=(V_pop(i,:).*(~(Flag4ub+Flag4lb)))+Best_flame_pos.*(Flag4ub+Flag4lb);
        
        f_new = fobj(V_pop(i,:));
        if f_new < G_OBL_fitness(i)  
            G_OBL_population(i,:) = V_pop(i,:);%更新位置
            G_OBL_fitness(i) = f_new; %更新适应度
        end
    end
    [~,I]=sort(G_OBL_fitness);  %适应度排序
    Moth_pos=G_OBL_population(I(1:N),:);  %取前N个种群
    Moth_fitness = G_OBL_fitness(I(1:N));%前N个种群的适应度
end