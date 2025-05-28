clc;clear;format compact;close all;
addpath 'E:\Desktop\network\DeepLearnToolbox-master\DBN'
addpath 'E:\Desktop\network\DeepLearnToolbox-master\NN'
addpath 'E:\Desktop\network\DeepLearnToolbox-master\util'
addpath 'E:\Desktop\network\MTL-master'
%% 加载数据。
Data = table2array(readtable('USA-data-15min.xlsx','Range','A2:B17521'));  % 105121
% 对风速数据进行四舍五入取整
Data(:,1) = round(Data(:,1));
% 归一化到 [0, 1] 范围或者给定范围
[data(:,1),parameter_wind]  = normalization(Data(:,1));
[data(:,2),parameter_power] = normalization(Data(:,2));

%% 划分数据
hPower = 15;
num_nwp = 1;
if  1

    power_data = data(:,2)';
    wind_data = data(:,1)';
    window_size = hPower;    % 窗口大小（历史数据长度）
    X = []; % 存储输入特征
    Y = []; % 存储输出标签
    window_data = [];
    N = size(power_data,2); 
    for i = 1:N-window_size
        window_data = power_data(i:i+window_size-1); % 获取当前窗口内的数据
        window_data = [wind_data(i+window_size),window_data];
        target = power_data(i+window_size); % 下一个时间步长的值作为输出标签
        
        X = [X; window_data]; % 将当前窗口数据添加到输入特征中
        Y = [Y; target]; % 将输出标签添加到输出标签中
    end
    % 将变量 X 和 Y 保存到 MATLAB 工作空间
%     save('data_set.mat', 'X', 'Y');
end

% load data_set.mat
% 训练和测试数量
num_train = 15000;
num_test  = 144;

%% 确定线性网络参数
% 创建线性神经网络模型1
net = fitnet(5); % 1表示输入特征的数量
% 设置训练参数
net.trainParam.epochs = 500; % 迭代次数
net.trainParam.lr = 0.001; % 学习率
net.trainParam.showWindow = false; % 关闭训练窗口
% 创建线性神经网络模型2
net2 = fitnet(4); % 1表示输入特征的数量
% 设置训练参数
net2.trainParam.epochs = 500; % 迭代次数
net2.trainParam.lr = 0.00001; % 学习率
net2.trainParam.showWindow = false; % 关闭训练窗口
%% bilstm
% 设置模型参数
inputSize = 5;  % 输入特征的维度
hiddenSize = 32;  % 隐层大小
outputSize = 1;  % 输出维度为1

% 创建BIGRU模型
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(hiddenSize)
    fullyConnectedLayer(outputSize)
    regressionLayer];

% 设置训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 120, ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 20, ...
    'L2Regularization', 0.01, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'none');
key = 0;  % 控制训练的方式
%% 进行多轮循环，训练线性网络整合性能
% 添加高斯噪音
mu    = 0;   % 随机数据的均值和
sigma = 0.0001; % 方差 0.00001
rand_matrix = randnData(length(X), hPower, mu, sigma);
% 将变量rand_matrix保存到 MATLAB 工作空间
% save('US_rand_matrix.mat', 'rand_matrix');
for cycle_number = 1:1   % 循环训练的次数
    
    if  1
             % 训练数据
        X_train{1} = X(1: num_train +(cycle_number-1)*num_test, :);
        Y_train{1} = Y(1: num_train +(cycle_number-1)*num_test, :);
         
        for i = 2: hPower+1
            % 训练数据
        X_train{i} = X(1: num_train +(cycle_number-1)*num_test,:);
        X_train{i}(1:num_train +(cycle_number-1)*num_test, end-( (i-1)-1):end) = ...
        X(1: num_train +(cycle_number-1)*num_test, end-( (i-1)-1):end)+ rand_matrix(1: num_train +(cycle_number-1)*num_test, 1:i-1);
    
        % 将大于1的值强制为1
        X_train{i}(X_train{i} > 1) = 1;
        % 将小于0的值强制为0
        X_train{i}(X_train{i} < 0) = 0;
    
        Y_train{i} = Y(1: num_train +(cycle_number-1)*num_test,:);
            
        end
        % 将变量 X 和 Y 保存到 MATLAB 工作空间
    %     save('data_set_MTL.mat', 'X_train', 'Y_train', 'X_test', 'Y_test');
%         save('cycle_data_set_MTL.mat', 'X_train', 'Y_train');
    end
    
%     load cycle_data_set_MTL.mat

    %% 模型学习
    for i = 1: hPower+1
    
        model{i} = fun_function_model(X_train{i}, Y_train{i});
        disp(['训练轮数:',num2str(cycle_number),';   完成训练的次数：',num2str(i)])
    end
%     save('US_model.mat', 'model');
%     
%     load US_model.mat
    %% 预测
    X_test_start  = X(num_train+1+(cycle_number-1)*num_test:num_train +cycle_number*num_test,:);
    Y_test        = Y(num_train+1+(cycle_number-1)*num_test:num_train +cycle_number*num_test,:);
    % 为第一步预测准备数据
    %     X_test.bilstm = X_test_start(1,:);
    %     X_test.cnngru = X_test_start(1,:);
    %     X_test.dnn = X_test_start(1,:);
    %     X_test.dbn = X_test_start(1,:);
    
        X_test.elm = X_test_start(1,:);
        X_test.svm = X_test_start(1,:);
        X_test.decisiontree = X_test_start(1,:);
        X_test.bp = X_test_start(1,:);
        X_test.mlp = X_test_start(1,:);
    
    for i = 1: num_test
    
        if i < hPower+1
            result = fun_function_MTLpredict(model{i}, X_test);
        else
            result = fun_function_MTLpredict(model{hPower+1}, X_test);
        end
        if  i < num_test
        % 输入当前时刻的nwp数据，将前一步输入功率的第2列至最后一列保留，并与最新的结果合并，更新输入
    %     X_test.bilstm = [X_test_start(i+1, 1:num_nwp),X_test.bilstm(1,end-(hPower-2):end),result.bilstm];  
    %     X_test.cnngru = [X_test_start(i+1, 1:num_nwp),X_test.cnngru(1,end-(hPower-2):end),result.cnngru];
    %     X_test.dnn = [X_test_start(i+1, 1:num_nwp),X_test.dnn(1,end-(hPower-2):end),result.dnn];
    %     X_test.dbn = [X_test_start(i+1, 1:num_nwp),X_test.dbn(1,end-(hPower-2):end),result.dbn];
    
        X_test.elm = [X_test_start(i+1, 1:num_nwp),X_test.elm(1,end-(hPower-2):end),result.elm];
        X_test.svm = [X_test_start(i+1, 1:num_nwp),X_test.svm(1,end-(hPower-2):end),result.svm];
        X_test.decisiontree = [X_test_start(i+1, 1:num_nwp),X_test.decisiontree(1,end-(hPower-2):end),result.decisiontree];
        X_test.bp = [X_test_start(i+1, 1:num_nwp),X_test.bp(1,end-(hPower-2):end),result.bp];
        X_test.mlp = [X_test_start(i+1, 1:num_nwp),X_test.mlp(1,end-(hPower-2):end),result.mlp];
    
        end
    %     pred.bilstm(1,i) = result.bilstm;
    %     pred.cnngru(1,i) = result.cnngru;
    %     pred.dnn(1,i)    = result.dnn;
    %     pred.dbn(1,i)    = result.dbn;
    
        pred.elm(1,i)    = result.elm;
        pred.svm(1,i)    = result.svm;
        pred.decisiontree(1,i) = result.decisiontree;
        pred.bp(1,i)     = result.bp;
        pred.mlp(1,i)    = result.mlp;
    end

    %% 线性神经网络
    % 将数据转换为网络的输入格式
    inputs = [pred.elm;pred.svm;pred.decisiontree;pred.bp;pred.mlp];
    targets = Y_test';
    
    % 训练线性神经网络
    net = train(net, inputs, targets);
    % 使用训练好的模型进行预测
    predictions = sim(net,inputs);
    
    result_data = result_k_means(inputs);
    % 训练线性神经网络
    net2 = train(net2, result_data, targets);
    % 使用训练好的模型进行预测
    predictions2 = sim(net2,result_data);
    

    % bilstm
    % 训练模型

    if key == 0
        net3 = trainNetwork(inputs, targets, layers, options);
        key = 1;
    else
        net3 = trainNetwork(inputs, targets, net3.Layers, options);
    end
    % 使用训练好的模型进行预测
    predictions3 = predict(net3,inputs);
%     rr = (pred.elm+pred.svm+pred.decisiontree+pred.bp+pred.mlp)./5;  % 平均
%     rr1 = 0.1.*pred.elm+0.1.*pred.svm+0.6.*pred.decisiontree+0.1.*pred.bp+0.1.*pred.mlp;  % 加权系数
    %% 可视化结果
    figure;
    plot(Y_test,'-','LineWidth',2)
    hold on
    % plot(pred.bilstm)
    % hold on
    % plot(pred.cnngru)
    % % hold on
    % plot(pred.dbn,'--')
    % hold on
    % plot(pred.dnn,'--')
    
    plot(pred.elm)
    hold on
    plot(pred.svm)
    hold on
    plot(pred.decisiontree,'-.')
    hold on
    plot(pred.bp)
    hold on
    plot(pred.mlp,'--')
    hold on
%     plot(rr,'--','LineWidth',1.5)
%     hold on
%     plot(rr1,'-.','LineWidth',1.5)
%     hold on
    plot(predictions,'-^','LineWidth',1)
    hold on
    plot(predictions2,'-*','LineWidth',1)
    hold on
    plot(predictions3,'-o','LineWidth',1)
    legend('实际值', 'elm预测值', 'svm预测值','decisiontree预测值','bp预测值','mlp预测值','5联合预测值','4联合预测值','bilstm联合')
    % legend('实际值', 'bilstm预测值', 'elm预测值', 'svm预测值','decisiontree预测值','bp预测值','cnngru预测值','mlp预测值','dnn预测值')

end
%% 预测
i_test = 15826;  % 16546
X_test_start  = X(i_test:i_test+(num_test-1),:);  
Y_test        = Y(i_test:i_test+(num_test-1),:);
% 为第一步预测准备数据
%     X_test.bilstm = X_test_start(1,:);
%     X_test.cnngru = X_test_start(1,:);
%     X_test.dnn = X_test_start(1,:);
%     X_test.dbn = X_test_start(1,:);

    X_test.elm = X_test_start(1,:);
    X_test.svm = X_test_start(1,:);
    X_test.decisiontree = X_test_start(1,:);
    X_test.bp = X_test_start(1,:);
    X_test.mlp = X_test_start(1,:);

for i = 1: num_test

    if i < hPower+1
        result = fun_function_MTLpredict(model{i}, X_test);
    else
        result = fun_function_MTLpredict(model{hPower+1}, X_test);
    end
    if  i < num_test
    % 输入当前时刻的nwp数据，将前一步输入功率的第2列至最后一列保留，并与最新的结果合并，更新输入
%     X_test.bilstm = [X_test_start(i+1, 1:num_nwp),X_test.bilstm(1,end-(hPower-2):end),result.bilstm];  
%     X_test.cnngru = [X_test_start(i+1, 1:num_nwp),X_test.cnngru(1,end-(hPower-2):end),result.cnngru];
%     X_test.dnn = [X_test_start(i+1, 1:num_nwp),X_test.dnn(1,end-(hPower-2):end),result.dnn];
%     X_test.dbn = [X_test_start(i+1, 1:num_nwp),X_test.dbn(1,end-(hPower-2):end),result.dbn];

    X_test.elm = [X_test_start(i+1, 1:num_nwp),X_test.elm(1,end-(hPower-2):end),result.elm];
    X_test.svm = [X_test_start(i+1, 1:num_nwp),X_test.svm(1,end-(hPower-2):end),result.svm];
    X_test.decisiontree = [X_test_start(i+1, 1:num_nwp),X_test.decisiontree(1,end-(hPower-2):end),result.decisiontree];
    X_test.bp = [X_test_start(i+1, 1:num_nwp),X_test.bp(1,end-(hPower-2):end),result.bp];
    X_test.mlp = [X_test_start(i+1, 1:num_nwp),X_test.mlp(1,end-(hPower-2):end),result.mlp];

    end
%     pred.bilstm(1,i) = result.bilstm;
%     pred.cnngru(1,i) = result.cnngru;
%     pred.dnn(1,i)    = result.dnn;
%     pred.dbn(1,i)    = result.dbn;

    pred.elm(1,i)    = result.elm;
    pred.svm(1,i)    = result.svm;
    pred.decisiontree(1,i) = result.decisiontree;
    pred.bp(1,i)     = result.bp;
    pred.mlp(1,i)    = result.mlp;
end

%% 线性神经网络
% 将数据转换为网络的输入格式
inputs = [pred.elm;pred.svm;pred.decisiontree;pred.bp;pred.mlp];
targets = Y_test';
% 使用训练好的模型进行预测
predictions = sim(net,inputs);

result_data = result_k_means(inputs);
% 使用训练好的模型进行预测
predictions2 = sim(net2,result_data);

% 使用训练好的模型进行预测
predictions3 = predict(net3,inputs);
% rr = (pred.elm+pred.svm+pred.decisiontree+pred.bp+pred.mlp)./5;  % 平均
% rr1 = 0.1.*pred.elm+0.1.*pred.svm+0.6.*pred.decisiontree+0.1.*pred.bp+0.1.*pred.mlp;  % 加权系数
% save('US_result.mat', 'Y_test','pred',"predictions","predictions2","predictions3");
% load US_result.mat
%% 可视化结果
figure;
plot(Y_test,'-','LineWidth',2)
hold on
plot(pred.elm)
hold on
plot(pred.svm)
hold on
plot(pred.decisiontree,'-.')
hold on
plot(pred.bp)
hold on
plot(pred.mlp,'--')
hold on
% plot(rr,'--','LineWidth',1.5)
% hold on
% plot(rr1,'-.','LineWidth',1.5)
% hold on
plot(predictions,'-^','LineWidth',1)
hold on
plot(predictions2,'-*','LineWidth',1)
hold on
plot(predictions3,'-o','LineWidth',1)
legend('实际值', 'elm预测值', 'svm预测值','decisiontree预测值','bp预测值','mlp预测值','5联合预测值','4联合预测值','bilstm联合')
save('MTL_pred.mat',"pred")
p_result(:,1) = pred.bp';
p_result(:,2) = pred.decisiontree';
p_result(:,3) = pred.elm';
p_result(:,4) = pred.mlp';
p_result(:,5) = pred.svm';
p_result(:,6) = predictions';
p_result(:,7) = predictions3';
p_result(:,8) = Y_test;

l = 4;
USevaluat(1,1:l) = evaluat(p_result(:,8), p_result(:,1));
USevaluat(2,1:l) = evaluat(p_result(:,8), p_result(:,2));
USevaluat(3,1:l) = evaluat(p_result(:,8), p_result(:,3));
USevaluat(4,1:l) = evaluat(p_result(:,8), p_result(:,4));
USevaluat(5,1:l) = evaluat(p_result(:,8), p_result(:,5));
USevaluat(6,1:l) = evaluat(p_result(:,8), p_result(:,6));
USevaluat(7,1:l) = evaluat(p_result(:,8), p_result(:,7));

