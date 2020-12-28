clc;%CORAL+TCA+xianzaideselection
clear all
addpath support_files;
% addpath Public;
% Problem='DTLZ2';%????
% Algorithm = 'RVEA';
Pop_num=100;
objective=2;
Evaluations=200;
RunNum =30;
alpha=2;
Ratio=5;


NewNum=3;
%%using the uncertainty of GP2 to select some samples provided by GP3(final version)
%%%?????%%%%%%%%%%%%
Problems={'DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7','DTLZ8','DTLZ9'...
    'UF1','UF2','UF3','UF4','UF5','UF6','UF7','ZDT1','ZDT2','ZDT3','ZDT4','ZDT6','OneMax','UF7','DTLZ8'};%we don't use WFG and ZDT since the two objectives are separetable.
% for Prob = 1:length(Problems)
for Prob =1：21
    
    Problem=Problems{Prob};
    tic
    for Run=1:RunNum
        Run
        %% Generate random population
        M=objective;
        ArchiveAll=[];
        [D,p1,p2] = P_settings('RVEA',Problem,M);
        upboundry=ones(1,D);
        lpboundry=zeros(1,D);
        Boundary=[upboundry;lpboundry];
%         L    = 11*D+24;
        L=500;
        N=Pop_num;
        THETA = 5.*ones(M+1,D+1);
        Model = cell(1,M+1);
        Population = lhsamp(N,D);
        FunctionValue = P_objective1('value',Problem,M,Population);
        FunctionValue2 = FunctionValue(:,2);
        Population2=Population;
        FunctionValue3 = FunctionValue(:,2);
        Population3=Population;
        NumberNew1=0;
        NumberNew2=0;
        NumberNew3=0;
        FE = size(Population,1);
        wmax=20;
        iter=1;
        Best_pos=0.5*ones(1,D);
        PopNew=[];
        Achx2=[];
        Achf2=[];
        MeanPredict=[];
        MeanTrue=[];
        MeanValue=[];
        MeanNewValue=[];
        
        %% Optimization
        while FE<=Evaluations
            %%train GP for expensive one
            if mod(iter, Ratio) ~= 0
                [PopDec2,PopObj2]=TrainingSelect(Population2,FunctionValue2,L,NumberNew2);
                [PopDec3,PopObj3]=TrainingSelect(Population3,FunctionValue3,L,NumberNew3);
            else
                [PopDec2,PopObj2]=TrainingSelect(Population,FunctionValue(:,2),L,NumberNew2);
                [PopDec3,PopObj3]=TrainingSelect(Population,FunctionValue(:,2),L,NumberNew3);
            end
            dmodel     = fitrgp(PopDec2,PopObj2,'KernelFunction','ardsquaredexponential','KernelParameters',THETA(2,:));
            Model{2}   = dmodel;
            THETA(2,:) = dmodel.KernelInformation.KernelParameters;
            %%%%%co-training%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            dmodel     = fitrgp(PopDec3,PopObj3,'KernelFunction','ardmatern52','KernelParameters',THETA(3,:));
            Model{3}   = dmodel;
            THETA(3,:) = dmodel.KernelInformation.KernelParameters;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             
               
            %%%%%cheap one evaluate more solutions
            if iter==1
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%GA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                [X_nex,F_nex] = optimize_least_expensive(Population,Boundary,Ratio,Problem,M,1);
                Population1=X_nex;
                ArchiveAll=X_nex;
                FunctionValue1=F_nex;                
                %%%%%train GP for the first time
                [~,index]  = unique(Population1,'rows');
                PopDec1 = Population1(index,:);
                PopObj1 = FunctionValue1(index,1);
                dmodel     = fitrgp(PopDec1,PopObj1,'KernelFunction','ardsquaredexponential','KernelParameters',THETA(1,:));
                Model{1}   = dmodel;
                THETA(1,:) = dmodel.KernelInformation.KernelParameters;
            else
                [PopDec1,PopObj1]=TrainingSelect(Population1,FunctionValue1,L,NumberNew1);
                dmodel     = fitrgp(PopDec1,PopObj1,'KernelFunction','ardsquaredexponential','KernelParameters',THETA(1,:));
                Model{1}   = dmodel;
                THETA(1,:) = dmodel.KernelInformation.KernelParameters;
            end
            %%RVEA optimization
            [V0,Pop_num] = UniformPoint(Pop_num,M);
            V    = V0;
            w=1;
            V1    = V0;
            cal=0;
            Pop=PopDec2;
%             Pop = lhsamp(N,D);
            while w < 20
                [MatingPool] = F_mating(Pop,100);
                OffDec = P_generator(MatingPool,Boundary,'Real',100);
                Pop = [Pop;OffDec];
%                 N=size(Pop,1);
                [Popmean,MSE] = FitPre(Pop,M,Model);
%                 ObjTRUE= P_objective1('value',Problem,M,Pop);
%                 Plot2D(ObjTRUE,Popmean,'ro')
                PopObj=Popmean;
                Selection = FSelection(PopObj,V,(w/wmax)^alpha);
                Pop = Pop(Selection,:);
                Popmean=Popmean(Selection,:);
                PopObj = PopObj(Selection,:);
                MSE=MSE(Selection,:);
                if(mod(w, ceil(wmax*0.1)) == 0)
                    V = V0.*repmat(max(PopObj,[],1)-min(PopObj,[],1),size(V0,1),1);
                end;
                cal=cal+size(Pop,1);
                w=w+1;
            end
            if(mod(FE, ceil(200*0.1)) == 0)
                V1 = V0.*repmat(max(FunctionValue,[],1)-min(FunctionValue,[],1),size(V0,1),1);
            end;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [PopNew0,PopNew0Obj,PopNew1,PopNewObj1]=IndividualSelect(Pop,PopObj,MSE,FE,V1,NewNum);
            New0Obj=P_objective1('value',Problem,M,PopNew0);
            New1Obj=P_objective1('value',Problem,M,PopNew1);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Xs=FunctionValue(:,1);
            Xt=FunctionValue(:,2);
            cov_src = cov(Xs) + eye(size(Xs,2));
            cov_tar = cov(Xt) + eye(size(Xt,2));
            A_coral = cov_src^(-1/2) * cov_tar^(1/2);
            Xs_new = Xs * A_coral;
            Xt_new = Xt * A_coral;
            
            mu = 0.5;
            lambda = 'unused';
            dim = 20;           % Deduced dimension
            kind = 'Gaussian';  % The dimension of Gaussian Kernel feature space is inifinite, so the deduced dimension can be 20.
            p1 = 1;
            p2 = 'unused';
            p3 = 'unused';
            
            W = getW(Xs_new', Xt_new', mu, lambda, dim, kind, p1, p2, p3);
            POF_deduced = getNewY(Xs_new', Xt_new',(New1Obj(:,1)* A_coral)', W, kind, p1, p2, p3);
            CostFunction = @(x)PreFit2(x,Model);
            % Get initial population by POF_deduced
            dis_px = @(p, x)sum((getNewY(Xs_new', Xt_new', (CostFunction(x)* A_coral)', W, kind, p1, p2, p3) - p).^2);
            initn = size(POF_deduced, 2);
            init_population = zeros(initn, D);
            for i = 1:initn
                init_population(i,:) = fmincon(@(x)dis_px(POF_deduced(:,i), x), rand(1,D), ...
                    [], [], [], [], zeros(1,D), ones(1,D), [],optimset('display', 'off'));
            end
%             
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            PopNew1_TL= init_population;
            [Semi1Pop,Semi1Obj,Semi2Pop,Semi2Obj]=Cotraining1(PopNew1,PopNew1_TL,Model,FE);

            
            
            NumberNew1=size([PopNew0;PopNew1],1);
            NumberNew2=size([PopNew0;Semi1Pop],1);
            NumberNew3=size([PopNew0;Semi2Pop],1);
            Population1=[PopNew0;PopNew1;Population1];
            FunctionValue1=[New0Obj(:,1);New1Obj(:,1);FunctionValue1];
            Population2=[PopNew0;Semi1Pop;Population2];
            Population3=[PopNew0;Semi2Pop;Population3];
            FunctionValue2=[New0Obj(:,2);Semi1Obj;FunctionValue2];
            FunctionValue3=[New0Obj(:,2);Semi2Obj;FunctionValue3];
            Population=[Population;PopNew0];
            FunctionValue=[FunctionValue;New0Obj];
            
            
            FE = FE+size(PopNew0,1)
            P_output2(Population,toc,'TrSAEA',Problem,M,Run);
            Problem
            iter=iter+1;
        end
        Run=Run+1;
    end
end

