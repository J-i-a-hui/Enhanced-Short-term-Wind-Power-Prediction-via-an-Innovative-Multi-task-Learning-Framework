%% 数据清洗主程序
clear;close all;clc
c1 = [168,218,219]; % 浅色
c2 = [069,123,157]; % 中
c3 = [029,053,087]; % 深
c4 = [231,056,071]; % 红
c5 = [042,157,142]; % 绿
c6 = [230,111,081]; % 绿
% 读取数据
% original_data = table2array(readtable('USA-data.xlsx','Range','A2:B15121'));
% data = original_data;
load outlier_result5.mat   % 导入'random_wind', 'random_power', 'data','OriginalData'
figure
subplot(2, 1, 1);
plot(data(:,1),Color= c2./255,LineWidth = 1)
hold on
plot(random_wind(1:20),data(random_wind(1:20),1),'o', 'MarkerFaceColor', c4./255, 'MarkerEdgeColor', 'none')
hold on
plot(random_wind(21:90),data(random_wind(21:90),1),'o', 'MarkerFaceColor', c5./255, 'MarkerEdgeColor', 'none')
hold on
plot(random_wind(91:100),OriginalData(random_wind(91:100),1),'o', 'MarkerFaceColor', c3./255, 'MarkerEdgeColor', 'none')
title('风速异常数据添加前后');
legend('正常数据','全局异常数据','局部异常值','缺失值', 'Location', 'northwest')
grid on
% ylim([0.05, 0.4]);   
ylabel("风速(m/s)");xlabel("风速数据")
title("(a)" )
set(gca,'FontSize',12,'FontName','SimSun','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 0.5, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...     
    'FontName', 'SimSun', 'FontSize', 12, 'FontWeight', 'normal');
subplot(2, 1, 2);
plot(data(:,2), Color= c2./255,LineWidth = 1)
hold on
plot(random_power(1:20),data(random_power(1:20),2),'o', 'MarkerFaceColor', c4./255, 'MarkerEdgeColor', 'none')
hold on
plot(random_power(21:90),data(random_power(21:90),2),'o', 'MarkerFaceColor', c5./255, 'MarkerEdgeColor', 'none')
hold on
plot(random_power(91:100),OriginalData(random_power(91:100),2),'o', 'MarkerFaceColor', c3./255, 'MarkerEdgeColor', 'none')
title('功率异常数据添加前后')
legend('正常数据','全局异常数据','局部异常值','缺失值', 'Location', 'northwest')
grid on
% ylim([0.05, 0.4]);   
ylabel("功率(MW)");xlabel("风电功率数据")
title("(b)" )
set(gca,'FontSize',12,'FontName','SimSun','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 0.5, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...     
    'FontName', 'SimSun', 'FontSize', 12, 'FontWeight', 'normal');


Original.Data = OriginalData; % 原始数据备份
%% 绘制含有异常值的风速-功率的散点图
original_data = data;    % 将异常数据备份，便于后期检索
figure
% 绘制第1个子图
subplot(2, 2, 1);
plot(data(:,1),data(:,2),'.',Color= c2./255)
title('含有异常值的风速-功率的散点图')
legend('数据', '异常值', 'northwest')
grid on
% ylim([0.05, 0.4]);   
ylabel("功率(MW)");xlabel("风速(m/s)")
title("(a)" )
set(gca,'FontSize',12,'FontName','SimSun','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 0.5, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...     
    'FontName', 'SimSun', 'FontSize', 12, 'FontWeight', 'normal');
%% 查找缺失值
NaN_idx          = isnan(data);  % 使用isnan函数查找NaN值  % 其中，记为1的为缺失值所在的位置
[NaN_row, ~]     = find(NaN_idx ~= 0); % 使用 find 函数查找非零元素的位置
data(NaN_row,:)  = [];          % 将可能为异常的数据剔除
data_process.NaN = data;        % 记录缺失值处理后的结果
Original.Data(NaN_row,:)  = []; % 原始数据进行同样剔除处理
Original.NaN     = Original.Data;

%% 数据异常值检测（得到异常值所在位置）
iforest.index_wind = iForestfOrest(data(:,1), 0.999);        % 最终得分  得分越低越有可能成为异常值
bLdod.index_wind   = BLDOD(data(:,1), 1, 0.99 );            % 最终得分  得分越高越有可能成为异常值
index_wind         = union(iforest.index_wind,  bLdod.index_wind);  % 结果求并集

% 在原始数据中记录异常数据的位置  此段是将缺失值与异常检测数据位置进行整合
i_wind             = zeros(length(data),1);
i_wind(index_wind) = 1;
adress = 0;
for i =1:length(original_data)
    
    if ismember(i, NaN_row)
    orig_idx(i,1) = 0;
    else
    adress = adress+1;
    orig_idx(i,1) = i_wind(adress);
    end
end
n = nnz(orig_idx); % 确定数组中非零元素的个数

iforest.index_power  = iForestfOrest(data(:,2), 0.999); % 最终得分  得分越低越有可能成为异常值
bLdod.index_power    = BLDOD(data(:,2), 1, 0.99 );  % 最终得分  得分越高越有可能成为异常值
index_power          = union(iforest.index_power,  bLdod.index_power);  % 结果求并集

% % 在原始数据中记录异常数据的位置  此段是将缺失值与异常检测数据位置进行整合
i_power              = zeros(length(data),1);
i_power(index_power) = 1;
adress = 0;
for i =1:length(original_data)
    
    if ismember(i, NaN_row)
    orig_idx(i,2) = 0;
    else
    adress = adress+1;
    orig_idx(i,2) = i_power(adress);
    end
end
n = nnz(orig_idx);          % 确定数组中非零元素的个数

subplot(2, 2, 2);
plot(data(:,1),data(:,2),'.',Color= c2./255)
hold on
plot(data(index_wind,1),data(index_wind,2),'o',Color= c5./255)
hold on
plot(data(index_power,1),data(index_power,2),'o',Color= c5./255)
title('全局和局部异常值剔除后的风速-功率散点图')
legend('数据', '异常值','Location', 'northwest')
grid on
% ylim([0.05, 0.4]);   
ylabel("功率(MW)");xlabel("风速(m/s)")
title("(b)" )
set(gca,'FontSize',12,'FontName','SimSun','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 0.5, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...     
    'FontName', 'SimSun', 'FontSize', 12, 'FontWeight', 'normal'); 

index         = union(index_wind,  index_power);  % 求异常数据位置的并集
data(index,:) = [];         % 将异常的数据剔除

Original.Data(index,:)    = []; % 原始数据进行同样剔除处理
Original.iforestbLdod     = Original.Data;
data_process.iforestbLdod = data;  % 记录第一次异常值剔除后的数据
                                                                                                                                                                              
%% 基于密度的异常检测 DBSCAN方法
% 设置参数
epsilon = 0.1; % 邻域半径
minPts  = 5;   % 最小邻域样本点数目
% 执行 DBSCAN
[DBSCAN_labels, ~] = dbscan(data,epsilon,minPts);
index_DBSCAN       = find(DBSCAN_labels == -1);    % 标记为-1的为异常数据
    
% figure % 绘图 
% 绘制第1个子图
subplot(2, 2, 3);
plot(data(:,1),data(:,2),'.',Color= c2./255)
hold on
plot(data(index_DBSCAN ,1),data(index_DBSCAN ,2),'o',Color= c4./255)
title('将密度方法检测的异常值标记的风速-功率散点图')
legend('数据', '异常值', 'Location', 'northwest')
grid on
% ylim([0.05, 0.4]);   
ylabel("功率(MW)");xlabel("风速(m/s)")
title("(c)" )
set(gca,'FontSize',12,'FontName','SimSun','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 0.5, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...     
    'FontName', 'SimSun', 'FontSize', 12, 'FontWeight', 'normal');

data(index_DBSCAN ,:) = []; % 将可能为异常的数据剔除

Original.Data(index_DBSCAN,:) = []; % 原始数据进行同样剔除处理
Original.DBSCAN       = Original.Data;
data_process.DBSCAN   = data;

% figure  % 绘图
subplot(2, 2, 4);
plot(data(:,1),data(:,2),'.',Color= c2./255)
title('密度方法检测异常值剔除后的风速-功率散点图')
legend('数据', '异常值', 'northwest')
grid on
ylim([0, 16.2]);   
ylabel("功率(MW)");xlabel("风速(m/s)")
title("(d)" )
set(gca,'FontSize',12,'FontName','SimSun','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 0.5, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...     
    'FontName', 'SimSun', 'FontSize', 12, 'FontWeight', 'normal');
%%  将正常数据还原到原始位置
result_data3     = recover_data(data_process.iforestbLdod, index_DBSCAN , data_process.DBSCAN);  % DBSCAN处理数据还原到孤立森林
% 使用isnan函数查找NaN值 % 使用 find 函数查找非零元素的位置
[result_idx3, ~] = find( (isnan(result_data3)) ~= 0);

result_data2     = recover_data(data_process.NaN, index , result_data3);  % 在孤立森林还原的数据基础上，还原到缺失值处理的结果部分
% 使用isnan函数查找NaN值 % 使用 find 函数查找非零元素的位置
[result_idx2, ~] = find( (isnan(result_data2)) ~= 0);

result_data1     = recover_data(original_data, NaN_row , result_data2);   % 缺失值处理部分还原到原始数据部分
% 使用isnan函数查找NaN值 % 使用 find 函数查找非零元素的位置
[result_idx, ~]  = find( (isnan(result_data1)) ~= 0);

%% 互相关拟合关系
% 风速->功率
net_1 = BP_train(50, data(:,1)', data(:,2)');
% 功率->风速
net_2 = BP_train(100, data(:,2)', data(:,1)');
model = ELM(100, data(:,2), data(:,1));

% 回填修复数据
indexes_wind  = find(orig_idx(:,1)); % 查找数组中非零元素的位置
indexes_power = find(orig_idx(:,2)); % 查找数组中非零元素的位置

% 将预测数据填补进对应位置
result_data1(indexes_wind,1)  = pred_wind;
result_data1(indexes_wind,2)  = data_process.NaN( index_wind, 2);
result_data1(indexes_power,2) = pred_power;
result_data1(indexes_power,1) = data_process.NaN( index_power, 1);

% 使用逻辑索引找出大于30的元素所在的行 第二个参数2表示针对每一行进行判断。
rows = any(result_data1 > 30, 2);
% 将对应行的元素置为NaN值
result_data1(rows, :) = NaN;
% result_data1 将估计值填到对应位置上

%% 自相关拟合关系
num_data = 11; % 自相关数据集的构建，1：num_data-1列是历史数据，num_data列是目标数据
for hData = 1:num_data
C_DataWind(:,hData)  = data(hData:end-(num_data-hData),1);
C_DataPower(:,hData) = data(hData:end-(num_data-hData),2);
end
% 风速自相关
net_3 = BP_train(100, C_DataWind(:,1:end-1)', C_DataWind(:,end)');
% 功率自相关
net_4 = BP_train(100, C_DataPower(:,1:end-1)', C_DataPower(:,end)');
% x1 = [];
% y1 = [];
j = 0;
for i = 1:length(result_data1)
    if isnan(result_data1(i))
        j = j+1;
        result_data1(i,1) = sim(net_3,result_data1(i-(num_data-1):i-1,1));
        x1(j) = result_data1(i,1);
        y1(j) = OriginalData(i,1);
        result_data1(i,2) = sim(net_4,result_data1(i-(num_data-1):i-1,2));
        x2(j) = result_data1(i,2);
        y2(j) = OriginalData(i,2);

    end
end

%% 结果评价
figure
subplot(2, 1, 1);
w_true = OriginalData(random_wind,1);
w_pred = result_data1(random_wind,1);

stem(w_true,'o',Color= c4./255,LineWidth = 2)
hold on
stem(w_pred,'.',Color= c2./255,LineWidth = 2)
cell_array = num2cell(random_wind);
xticklabels(cell_array);
xticks(1:100);
% 将横坐标刻度标签旋转45度
xtickangle(65)
legend('实际数据','修正数据', 'Location', 'northwest');
legend('NumColumns', 2);  % 设置图例的列数
grid on
% ylim([0.05, 0.4]);   
ylabel("风速(m/s)");xlabel("风速数据索引")
title("(a)" )
set(gca,'FontSize',16,'FontName','SimSun','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 0.5, 'Layer', 'top',...
     'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...     
    'FontName', 'SimSun', 'FontSize', 12, 'FontWeight', 'normal');

subplot(2, 1, 2);
p_true = OriginalData(random_power,2);
p_pred = result_data1(random_power,2);

stem(p_true,'o',Color= c4./255,LineWidth = 2)
hold on
stem(p_pred,'.',Color= c2./255,LineWidth = 2)

cell_array = num2cell(random_power);
xticklabels(cell_array);
xticks(1:100);
% 将横坐标刻度标签旋转45度
xtickangle(65)
legend('实际数据','修正数据', 'Location', 'northwest');
legend('NumColumns', 2);  % 设置图例的列数
grid on
% ylim([0.05, 0.4]);   
ylabel("功率(MW)");xlabel("功率数据索引")
title("(b)" )
set(gca,'FontSize',16,'FontName','SimSun','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 0.5, 'Layer', 'top',...
     'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...     
    'FontName', 'SimSun', 'FontSize', 12, 'FontWeight', 'normal');


%% BP用于补全空值
function net = BP_train(epochs, train_x, train_y)

% 此程序利用BP网络拟合数据从而得到映射关系
% train_x train_y 确定映射关系的输入和输出

% 节点个数
hiddennum = 25;    % 隐含层节点数量

% 构建BP神经网络
%"train_x"与"train_y"需要对应，行数为数据维数，列数为数据个数
net = newff(train_x,train_y,hiddennum,{'tansig','purelin'},'trainlm');
% 建立模型，传递函数使用purelin，采用梯度下降法训练

%  网络参数配置（ 训练次数，学习速率，训练目标最小误差等）
net.trainParam.epochs = epochs;         % 训练次数，
net.trainParam.lr     = 0.01;       % 学习速率，
net.trainParam.goal   = 0.00001;      % 训练目标最小误差
% 设置训练窗口关闭
net.trainParam.showWindow = false;
%  BP神经网络训练
net = train(net,train_x,train_y);  % 开始训练

end

%% 
function matrix = ELM(hidden_neurons, X_train, Y_train)
    
    % 设置极限学习机参数
    % hidden_neurons = 100; % 隐层神经元数量
    
    % 构建随机权重矩阵和偏置向量
    matrix.input = rand(hidden_neurons, size(X_train, 2)) * 2 - 1;
    matrix.bias = rand(hidden_neurons, 1);
    
    % 计算隐层输出
    H_train = 1./(1+exp(-(X_train * matrix.input' + repmat(matrix.bias', size(X_train, 1), 1))));
    
    % 使用最小二乘法求解输出权重
    matrix.output = pinv(H_train) * Y_train;

end

function pred  = ELM_prediction(matrix, x_test)
    H_test = 1./(1+exp(-(x_test * matrix.input' + repmat(matrix.bias', size(x_test, 1), 1))));
    % 预测
    pred = H_test * matrix.output;
end
%% Boundary-aware Local density-based Outlier Detection (BLDOD 局部异常值检测)
function index = BLDOD( X, lambda ,threshold)
    
    % Compute the number of points
    m = size(X, 1);
    
    % Compute the standard deviation of the points（每一列的标准差）
    stdDev = std(X, 'omitnan');
    
    N = zeros(m, 1);
    
    % Multiply the norm of the standard deviation of the points with lambda
    deviation = lambda * norm(stdDev);
    
    for i = 1 : m
        % Move the considered point to the center (0, 0)
        D = X - X(i, :);
        % The number of points that are as far away as the deviation length value from the considered point
        N(i) = sum(sqrt(sum(D.^2, 2)) <= deviation);
    end
    
    % Normalize the values between 0-1
    score = 1 - normalize(N,"range");
    [aa, index] = sort(score);  % 排序并返回原始索引
    index = index(round(length(index)*threshold):end);
end

%% 孤立森林全局异常值检测
function index1 = iForestfOrest(Data, threshold)
addpath('E:\Desktop\pygcn-master\data\iForest-master')
% Run iForest
    % general parameters
    rounds = 1; % rounds of repeat
    
    % parameters for iForest
    NumTree = 100; % number of isolation trees
    NumSub = 256; % subsample size
    NumDim = size(Data, 2); % do not perform dimension sampling 
     
    mtime = zeros(rounds, 2);
    rseed = zeros(rounds, 1);

    for r = 1:rounds
        disp(['rounds ', num2str(r), ':']);
        
        rseed(r) = sum(100 * clock);
        Forest = IsolationForest(Data, NumTree, NumSub, NumDim, rseed(r));
        mtime(r, 1) = Forest.ElapseTime;
        [Mass, ~] = IsolationEstimation(Data, Forest);
        Score = - mean(Mass, 2);  % 最终得分  得分越高越有可能成为异常值   
    end
    [aa, index] = sort(Score);  % 排序并返回原始索引
%     index1 = index(1:round(length(index)*(1-threshold)));
    index1 = index(round(length(index)*threshold):end);
end
%% 还原数据位置
function result_data = recover_data(original_data, NaN_row, data)
    %  此函数是还原处理过的数据，并将标记为异常的数据置为空
    %  original_data：进行处理操作的原始数据
    %  NaN_row：处理数据时记录的原始数据索引
    %  data： 处理后的数据
    original_label = ones(length(original_data), 1);
    original_label(NaN_row) = 0;
    adress = 0;
    for i =1:length(original_label)
        
        if original_label(i) == 1
            adress = adress+1;
            result_data(i,:) = data(adress,:);
        else
            result_data(i,:) = NaN;
        end
    end
end

