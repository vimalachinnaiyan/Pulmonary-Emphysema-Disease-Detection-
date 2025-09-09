function [] = Plot_Results()


Plot_Result()
Image_Result()
Plot_segment()
plot_confution()
Actual_pred()
end


function [] = Plot_Result()
clc;
clear all;
close all;
warning off;


load Eval_all

Eval_all1 = load('./Paper 1/Evaluate_all.mat');
Eval_all2 = load('./Paper 2/Eval_all.mat');

Perf_Terms = ["Accuracy"; "Sensitivity"; "Specificity"; "Precision"; "FPR"; "FNR"; "NPV"; "FDR"; "F1-Score"; "MCC"];
Terms = [6, 7, 9, 10];
Learn_per = [35, 45, 55, 65, 75];



%% Plot Results
for m = 1: 2
    for n = 1:length(Terms)
        for i = 1:size(Eval_all, 1)
            for j = 1:size(Eval_all, 2)
                for k = 1:size(Eval_all{i, j}, 1)
                    if n == length(Terms) % Perf_Terms
                        val_f(j, k) = Eval_all{i, j}(k, Terms(n) + 4);
                        val_1(j, k) = Eval_all1.Evaluate_all(j, k, Terms(n) + 4);
                        val_2(j, k) = Eval_all2.Eval_all{i, j}(k, Terms(n) + 4);
                    else
                        val_f(j, k) = Eval_all{i, j}(k, Terms(n) + 4) * 100;
                        val_1(j, k) = Eval_all1.Evaluate_all(j, k, Terms(n) + 4) * 100;
                        val_2(j, k) = Eval_all2.Eval_all{i, j}(k, Terms(n) + 4) * 100;
                    end
                end
            end
            val_for_line = val_f(:, 1:5)';
            val_for_bar = [val_f(:, 6)'; val_f(:, 7)' ;val_f(:, 8)'; val_f(:, 9)'; val_1(:, 10)'; val_2(:, 5)'; val_f(:, 10)'] ;

            figure,
            plot(Learn_per, val_for_line(1, :), 'r', 'LineWidth', 2, 'marker', 's', 'markersize', 10)
            hold on;
            plot(Learn_per, val_for_line(2, :), 'g', 'LineWidth', 2, 'marker', 's', 'markersize', 10)
            plot(Learn_per, val_for_line(3, :), 'b', 'LineWidth', 2, 'marker', 's', 'markersize', 10)
            plot(Learn_per, val_for_line(4, :), 'm', 'LineWidth', 2, 'marker', 's', 'markersize', 10)
            plot(Learn_per, val_for_line(5, :), 'k', 'LineWidth', 2, 'marker', 's', 'markersize', 10)
            set(gca, 'fontsize', 12);
            grid on;
            xlabel('Learning Percentage', 'fontsize', 12);
            xticks([35, 45, 55, 65, 75])
            xticklabels({'35','45','55','65','75'}')

            ylabel(char(Perf_Terms{Terms(n)}), 'fontsize', 12);
            h = legend('CO-ARes-DNet', 'AOA-ARes-DNet', 'WSA-ARes-DNet', 'MOA-ARes-DNet', 'IMOA-ARes-DNet');
            set(h, 'fontsize', 12, 'Location', 'NorthEastOutside')
            print('-dtiff','-r300',['.\Results\', 'alg-',char(Perf_Terms(Terms(n))),'-',num2str(i)])

            figure,
            bar(Learn_per, val_for_bar', 'Linewidth', 1.5)
            set(gca, 'fontsize', 12);
            grid on;
            xlabel('Learning Percentage', 'fontsize', 12);
            ylabel(char(Perf_Terms{Terms(n)}), 'fontsize', 12);
            h = legend('Resnet', 'CNN', 'DBN', 'Residual DenseNet ','IHBA-AMSD-RN-BiLSTM', 'H-MRTSA-A-ResNet', 'IMOA-ARes-DNet');
            set(h,'fontsize', 12, 'Location', 'NorthEastOutside')
            print('-dtiff','-r300',['.\Results\', 'mtd-',char(Perf_Terms(Terms(n))),'-',num2str(i)])
        end
    end
end


load Eval_all_1;

Perf_Terms = ["Accuracy"; "Sensitivity"; "Specificity"; "Precision"; "FPR"; "FNR"; "NPV"; "FDR"; "F1-Score"; "MCC"];
Terms_1 = [1, 2, 3, 4];
epoch = [50, 100, 150, 200];

%% Plot  Results
for n = 1:length(Terms_1) % Perf_Terms
    for i = 1:size(Eval_all_1, 1)
        for j = 1:size(Eval_all_1, 2)
            for k = 1:size(Eval_all_1{i, j}, 1)
                if n == length(Terms_1) % Perf_Terms
                    val_f_1(j, k) = Eval_all_1{i, j}(k, n + 4);
                else
                    val_f_1(j, k) = Eval_all_1{i, j}(k, n + 4) * 100;
                end
            end
        end
        val_for_line_1 = val_f_1(:, 1:5)';
        val_for_bar_1 = val_f_1(:, 6:end)';
        
        figure,
        plot(epoch, val_for_line_1(1, :),'Color',[0.4660 0.6740 0.1880], 'LineWidth', 2, 'marker', 's', 'markersize', 10)
        hold on;
        plot(epoch, val_for_line_1(2, :),'Color',[0.7 0 1], 'LineWidth', 2, 'marker', 's', 'markersize', 10)
        plot(epoch, val_for_line_1(3, :),'Color',[1 .0 .1], 'LineWidth', 2, 'marker', 's', 'markersize', 10)
        plot(epoch, val_for_line_1(4, :), 'm', 'LineWidth', 2, 'marker', 's', 'markersize', 10)
        plot(epoch, val_for_line_1(5, :), 'k', 'LineWidth', 2, 'marker', 's', 'markersize', 10)
        set(gca, 'fontsize', 12);
        grid on;
        xlabel('Epoch', 'fontsize', 12);
        ylabel(char(Perf_Terms{Terms_1(n)}), 'fontsize', 12);
        h = legend('CO-ARes-DNet', 'AOA-ARes-DNet', 'WSA-ARes-DNet', 'MOA-ARes-DNet', 'IMOA-ARes-DNet');
        set(h, 'fontsize', 12, 'Location', 'NorthEastOutside')
        print('-dtiff','-r300',['.\Results\', 'alg-',char(Perf_Terms(Terms_1(n))),'-',num2str(i)])
        
        figure,
        bar(epoch, val_for_bar_1', 'Linewidth', 1.5)
        set(gca, 'fontsize', 12);
        grid on;
        xlabel('Epoch', 'fontsize', 12);
        ylabel(char(Perf_Terms{Terms_1(n)}), 'fontsize', 12);
        h = legend('Resnet', 'CNN', 'DBN', 'Residual DenseNet ', 'IMOA-ARes-DNet');
        set(h,'fontsize', 12, 'Location', 'NorthEastOutside')
        print('-dtiff','-r300',['.\Results\', 'mtd-',char(Perf_Terms(Terms_1(n))),'-',num2str(i)])
    end
end
end



%% Image Results
function [] = Image_Result()
% clc;
% close all;
% clear all;

load Images
load prep
load Seg_Image
load Segment_unet
load Segment_Resuet
load Segment_mask
load Segment_unet3
img = [18,568, 570, 606, 611];  %i[4, 5, 7, 8, 9];
for n = 1:5
    %     disp(n)
    subplot(1, 5, 1), imshow(Images{img(n)});
    title('Original');
    subplot(1, 7, 2), imshow(prep{img(n)});
    title('Preprocess');
    subplot(1, 7, 3), imshow(images1{img(n)});
    title('UNET');
    
    subplot(1, 7, 4), imshow(images2{img(n)});
    title('ResUNET');
    subplot(1, 7, 5), imshow(images3{img(n)});
    title('Mask Rcnn');
    subplot(1, 7, 6), imshow(images4{img(n)});
    title('UNET3+');
    
    subplot(1, 7, 7), imshow(Seg_Image{img(n)});
    title('Segmented');
    
    pause(1)
    print('-dtiff', '-r300', ['./Results/IMG/Images-', num2str(n)]);
    s1 = strcat('.\Results\IMG\', 'Original-', num2str(n), '.png');
%     imwrite(Images{img(n)},s1)
    s2 = strcat('.\Results\IMG\', 'Preprocess-', num2str(n), '.png');
    imwrite(prep{img(n)},s2)
    s3 = strcat('.\Results\IMG\', 'Unet-', num2str(n), '.png');
%     imwrite(images1{img(n)},s3)
    
    s4 = strcat('.\Results\IMG\', 'ResUnet-', num2str(n), '.png');
%     imwrite(images2{img(n)},s4)
    s5 = strcat('.\Results\IMG\', 'Mask Rcnn-', num2str(n), '.png');
%     imwrite(images3{img(n)},s5)
    s6 = strcat('.\Results\IMG\', 'Unet3+-', num2str(n), '.png');
%     imwrite(images4{img(n)},s6)
    
    s7 = strcat('.\Results\IMG\', 'Segmented-', num2str(n), '.png');
%     imwrite(Seg_Image{img(n)},s7)
end
end



function [] = Conver()
Node  = [1];
Dataset = [];
% Convergence
Algms = {'Co', 'Aoa', 'Wsa', 'Moa', 'Prop'};
for var = 1:1 % For all  variations
    for i = 1 : 5 % For all agorithms
        eval(['load ', char(Algms{i})])
        eval(['val(i, :) = ',char(Algms{i}),'(var).fit;'])
    end
    ind = find(val(:, end) == min(val(:, end)));
    t = val(5, :);
    val(5, :) = val(ind(1), :);
    val(ind(1), :) = t;
    for i = 1 : 5
        stat(i, :) = Statistical_Analysis(val(i, :));
        a1{i} = Statistical_Analysis(val(i, :));
    end

    fprintf('Statistical Analysis :')
    ca = {'Best','Worst','Mean','Median','Std_dev'};
    T = table(a1{1}',a1{2}',a1{3}',a1{4}',a1{5}','Rownames',ca);
    T.Properties.VariableNames = {'CO-ARes-DNet', 'AOA-ARes-DNet', 'WSA-ARes-DNet', 'MOA-ARes-DNet', 'IMOA-ARes-DNet'};
    disp(T)

    fig = figure;
    set(fig, 'Name', 'Convergence');
    plot(val(1, :), 'r', 'LineWidth', 2)
    hold on;
    plot(val(2, :), 'g', 'LineWidth', 2)
    plot(val(3, :), 'b', 'LineWidth', 2)
    plot(val(4, :), 'm', 'LineWidth', 2)
    plot(val(5, :), 'c', 'LineWidth', 2)
    set(gca, 'fontsize', 12);
    grid on;
    xlabel('No of Iteration', 'fontsize', 12);

    ylabel('Cost Function', 'fontsize', 12);
    h = legend('CO-ARes-DNet', 'AOA-ARes-DNet', 'WSA-ARes-DNet', 'MOA-ARes-DNet', 'IMOA-ARes-DNet');
    set(h, 'fontsize', 12, 'Location', 'NorthEastOutside')
    print('-dtiff', '-r300', ['./Results/Convergence_', num2str(var)])
end
end

function [] = Plot_segment()
load Eval_all_Dice

Error_Terms = {'Dice Coefficient', 'Jacard', 'Accuracy', 'MSE', 'PSNR', 'IOU'};
Algms = {'DO-AMTUnet++-ASPP', 'EOO-AMTUnet++-ASPP', 'RSA-AMTUnet++-ASPP', 'TFMOA-AMTUnet++-ASPP', 'ARI-TFMOA-AMTUnet++-ASPP'};

for i = 1:size(Eval_all_Dice{1}, 2)
    for j = 1:size(Eval_all_Dice{1}, 1)
        for k = 1:length(Eval_all_Dice)
            value(j, k) = Eval_all_Dice{k}(j, i);
        end
    end
    for j = 1:size(value, 1)
        Value(j, 1) = min(value(j, :));
        Value(j, 2) = max(value(j, :));
        Value(j, 3) = mean(value(j, :));
        Value(j, 4) = median(value(j, :));
        Value(j, 5) = std(value(j, :));
    end
    
    X = 1:size(Value, 2);
    val_for_alg = Value(1:5, :);    
    figure();
    b = bar(X, val_for_alg', 'LineWidth', 2);
    b(1).FaceColor = [.0 .5 .5];
    b(2).FaceColor = [0 0 1];
    b(3).FaceColor = [1 .0 .1];
    b(4).FaceColor = [0.7 .0 1];
    b(5).FaceColor = [0 .0 .0];

    set(gca, 'fontsize', 12);
    grid on;
    xlabel('STATISTICAL ANALYSIS', 'fontsize', 10);
    ylabel(char(Error_Terms{i}), 'fontsize', 12);
    xticklabels({'BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'});
    h = legend('Unet', 'ResUnet', 'Mask-RCNN', 'Unet3+', 'TR2Unet');
    set(h, 'fontsize', 7, 'Location', 'NorthEastOutside')
    print('-dtiff', '-r300', ['./Results/', 'seg_mtd-', char(Error_Terms{i})])
end

end


function [] = Actual_pred()
load Eval_actual;
load Eval_Pred;
% Example data
actual = Eval_actual;
predicted = Eval_Pred;

% Create a table
T = table(actual{1,1}, predicted{1,1}, 'VariableNames', {'ActualValue', 'PredictedValue'});

% Write the table to an Excel file
filename = 'actual_vs_predicted.xlsx';
writetable(T, filename);

disp(['Data written to ', filename]);
end


function[] = plot_confution()
load Eval_actual;
load Eval_Pred;
Actual = Eval_actual;
Predict = Eval_Pred;
classNames = {'CLE', 'NORMAL', 'PLE','PSE'};
for n = 1:1
    figure;
    cm = confusionmat(vec2ind(Actual{n}'), vec2ind(Predict{n}'));
    heatmap(cm, 'Colormap', jet, 'ColorbarVisible', 'on', 'FontSize', 10);
    xlabel('Predicted');
    ylabel('Actual');
    ax = gca;
    ax.XDisplayLabels = classNames;
    ax.YDisplayLabels = classNames;
%     title(sprintf('Confusion Matrix'));
    path = sprintf('./Results/Confusion_%d.png', n);
    saveas(gcf, path);
end

end

function[a] = Statistical_Analysis(val)
a(1) = min(val);
a(2) = max(val);
a(3) = mean(val);
a(4) = median(val);
a(5) = std(val);
end


