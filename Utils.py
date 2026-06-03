import configparser
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.sparse.linalg import eigs
from tensorflow import keras

# 配置matplotlib以优化内存管理
plt.rcParams['figure.max_open_warning'] = 30  # 增加警告阈值
plt.ioff()  # 关闭交互模式，避免在后台保留图形



##########################################################################################
# Print score between Ytrue and Ypred ####################################################

def PrintScore(true, pred, fold=-1, savePath=None, average='macro', model_name="model"):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + f"Result_{model_name}.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores for fold ", str(fold))
    print('Acc\tF1S\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2], F1[3], F1[4]),
          file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred,
                                        target_names=['Wake','N1','N2','N3','REM'],
                                        digits=4), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true,pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t',metrics.accuracy_score(true,pred), file=saveFile)
    print(' Cohen Kappa\t',metrics.cohen_kappa_score(true,pred), file=saveFile)
    print('    F1-Score\t',metrics.f1_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    print('   Precision\t',metrics.precision_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    print('      Recall\t',metrics.recall_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    if savePath != None:
        saveFile.close()
    return

##########################################################################################
# Print confusion matrix and save ########################################################

def ConfusionMatrix(y_true, y_pred, classes, savePath, fold=-1, model_name="model", title=None, cmap=plt.cm.Blues):
    if not title:
        title = f'Confusion matrix for fold {str(fold)}'
    
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n=cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(5, 4))
    
    try:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j]*100,'.2f')+'%\n'+format(cm_n[i, j],'d'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(savePath + f"{model_name}_ConfusionMatrix_fold_{fold}.png", 
                   bbox_inches='tight', dpi=150)  # 添加bbox_inches和dpi参数优化保存
        plt.show()
    finally:
        # 确保图形被关闭，即使出现异常
        plt.close(fig)
    
    return ax

##########################################################################################
# Draw ACC / loss curve and save #########################################################

def VariationCurve(fit,val,yLabel,savePath,figsize=(9, 6)):
    # 显式创建figure对象
    fig = plt.figure(figsize=figsize)
    
    try:
        plt.plot(range(1,len(fit)+1), fit,label='Train')
        plt.plot(range(1,len(val)+1), val, label='Val')
        plt.title('Model ' + yLabel)
        plt.xlabel('Epochs')
        plt.ylabel(yLabel)
        plt.legend()
        plt.tight_layout()  # 添加tight_layout优化布局
        plt.savefig(savePath + 'Model_' + yLabel + '.png', 
                   bbox_inches='tight', dpi=150)  # 添加bbox_inches和dpi参数
        plt.show()
    finally:
        # 确保图形被关闭，即使出现异常
        plt.close(fig)
    
    return

##########################################################################################
# 生成概率分布热力图 #####################################################################

def plot_probability_heatmap(probabilities, true_labels=None, save_path=None, 
                           fold=-1, model_name="MFE", max_samples=1000, 
                           figsize=(12, 8), title_prefix=""):
    """
    生成模型分类概率分布热力图
    
    参数:
    - probabilities: 形状为 (n_samples, n_classes) 的概率矩阵
    - true_labels: 真实标签，用于颜色编码 (可选)
    - save_path: 保存路径
    - fold: fold编号
    - model_name: 模型名称
    - max_samples: 最大显示样本数（避免图像过大）
    - figsize: 图像尺寸
    - title_prefix: 标题前缀
    
    返回:
    - fig: matplotlib图像对象
    """
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    n_samples, n_classes = probabilities.shape
    
    # 如果样本太多，进行采样
    if max_samples is not None and n_samples > max_samples:
        indices = np.random.choice(n_samples, max_samples, replace=False)
        indices = np.sort(indices)  # 保持时序
        prob_subset = probabilities[indices]
        true_subset = true_labels[indices] if true_labels is not None else None
        print(f"采样显示 {max_samples}/{n_samples} 个样本")
    else:
        prob_subset = probabilities
        true_subset = true_labels
        indices = np.arange(n_samples)
        if max_samples is None:
            print(f"显示所有 {n_samples} 个样本")
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    try:
        # 上图：概率热力图 - 每个epoch的概率分布
        im = axes[0].imshow(prob_subset.T, aspect='auto', cmap='viridis', 
                           interpolation='nearest', vmin=0, vmax=1)
        
        # 设置y轴标签
        axes[0].set_yticks(range(n_classes))
        axes[0].set_yticklabels(class_names)
        axes[0].set_ylabel('Sleep Stages')
        axes[0].set_xlabel('Epochs (30-second windows)')
        
        # 设置x轴刻度，显示时间信息
        n_display = len(prob_subset)
        if n_display > 20:
            epoch_ticks = np.arange(0, n_display, max(1, n_display//10))
            time_labels = [f'{int(epoch*0.5)}min' for epoch in epoch_ticks]
            axes[0].set_xticks(epoch_ticks)
            axes[0].set_xticklabels(time_labels)
        
        axes[0].set_title(f'{title_prefix}Probability Heatmap - Fold {fold} ({n_display} epochs)')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[0])
        cbar.set_label('Probability')
        cbar.ax.tick_params(labelsize=10)
        
        # 下图：真实标签和预测标签对比（如果提供了真实标签）
        if true_subset is not None:
            pred_labels = np.argmax(prob_subset, axis=1)
            
            # 创建标签序列对比图
            epoch_range = range(len(true_subset))
            axes[1].plot(epoch_range, true_subset, 'b-', 
                        alpha=0.8, linewidth=2, label='True Labels', marker='o', markersize=2)
            axes[1].plot(epoch_range, pred_labels, 'r--', 
                        alpha=0.8, linewidth=2, label='Predicted Labels', marker='s', markersize=2)
            
            axes[1].set_yticks(range(n_classes))
            axes[1].set_yticklabels(class_names)
            axes[1].set_ylabel('Sleep Stages')
            axes[1].set_xlabel('Epochs (30-second windows)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title(f'Sleep Stage Sequence - {len(true_subset)} epochs ({len(true_subset)*0.5:.1f} min)')
            
            # 设置x轴刻度
            if len(true_subset) > 20:
                axes[1].set_xticks(epoch_ticks)
                axes[1].set_xticklabels(time_labels)
            
            # 计算并显示准确率
            accuracy = np.mean(true_subset == pred_labels)
            axes[1].text(0.02, 0.98, f'Accuracy: {accuracy:.3f}', 
                        transform=axes[1].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        verticalalignment='top', fontsize=10)
        else:
            # 如果没有真实标签，只显示预测标签
            pred_labels = np.argmax(prob_subset, axis=1)
            epoch_range = range(len(pred_labels))
            axes[1].plot(epoch_range, pred_labels, 'g-', 
                        alpha=0.8, linewidth=2, label='Predicted Labels', marker='o', markersize=2)
            
            axes[1].set_yticks(range(n_classes))
            axes[1].set_yticklabels(class_names)
            axes[1].set_ylabel('Sleep Stages')
            axes[1].set_xlabel('Epochs (30-second windows)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title(f'Predicted Sleep Stage Sequence - {len(pred_labels)} epochs')
            
            if len(pred_labels) > 20:
                axes[1].set_xticks(epoch_ticks)
                axes[1].set_xticklabels(time_labels)
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            filename = f"{save_path}{model_name}_probability_heatmap_fold_{fold}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"概率热力图已保存到: {filename}")
        
        plt.show()
        
        return fig
        
    finally:
        # 确保图形被关闭
        if 'fig' in locals():
            plt.close(fig)


def plot_class_probability_distribution(probabilities, save_path=None, 
                                      fold=-1, model_name="MFE", 
                                      figsize=(15, 4)):
    """
    生成各类别概率分布箱线图
    
    参数:
    - probabilities: 形状为 (n_samples, n_classes) 的概率矩阵
    - save_path: 保存路径
    - fold: fold编号
    - model_name: 模型名称
    - figsize: 图像尺寸
    
    返回:
    - fig: matplotlib图像对象
    """
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        # 创建箱线图数据
        box_data = [probabilities[:, i] for i in range(probabilities.shape[1])]
        
        # 绘制箱线图
        bp = ax.boxplot(box_data, labels=class_names, patch_artist=True)
        
        # 设置颜色
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Probability')
        ax.set_xlabel('Sleep Stages')
        ax.set_title(f'Class Probability Distribution - Fold {fold}')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        for i, class_name in enumerate(class_names):
            mean_prob = np.mean(probabilities[:, i])
            ax.text(i+1, mean_prob, f'{mean_prob:.3f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            filename = f"{save_path}{model_name}_prob_distribution_fold_{fold}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"概率分布图已保存到: {filename}")
        
        plt.show()
        
        return fig
        
    finally:
        # 确保图形被关闭
        if 'fig' in locals():
            plt.close(fig)


##########################################################################################
# 生成HMM对比热力图 #####################################################################

def plot_hmm_comparison_heatmap(probabilities, true_labels, hmm_predictions, original_predictions,
                               save_path=None, fold=-1, model_name="HMM", max_samples=1000, 
                               figsize=(15, 12), title_prefix=""):
    """
    生成HMM优化前后对比的概率热力图
    
    参数:
    - probabilities: 平滑后的概率矩阵 (n_samples, n_classes) 
    - true_labels: 真实标签
    - hmm_predictions: HMM优化后的预测标签
    - original_predictions: 原始平滑预测标签
    - save_path: 保存路径
    - fold: fold编号
    - model_name: 模型名称
    - max_samples: 最大显示样本数
    - figsize: 图像尺寸
    - title_prefix: 标题前缀
    
    返回:
    - fig: matplotlib图像对象
    """
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    n_samples, n_classes = probabilities.shape
    
    # 如果样本太多，进行采样
    if max_samples is not None and n_samples > max_samples:
        indices = np.random.choice(n_samples, max_samples, replace=False)
        indices = np.sort(indices)  # 保持时序
        prob_subset = probabilities[indices]
        true_subset = true_labels[indices]
        hmm_subset = hmm_predictions[indices] 
        orig_subset = original_predictions[indices]
        print(f"采样显示 {max_samples}/{n_samples} 个样本")
    else:
        prob_subset = probabilities
        true_subset = true_labels
        hmm_subset = hmm_predictions
        orig_subset = original_predictions
        indices = np.arange(n_samples)
        if max_samples is None:
            print(f"显示所有 {n_samples} 个样本")
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[3, 1, 1])
    
    try:
        # 上图：概率热力图（HMM输入的平滑概率）
        im = axes[0].imshow(prob_subset.T, aspect='auto', cmap='viridis', 
                           interpolation='nearest', vmin=0, vmax=1)
        
        # 设置y轴标签
        axes[0].set_yticks(range(n_classes))
        axes[0].set_yticklabels(class_names)
        axes[0].set_ylabel('Sleep Stages')
        axes[0].set_title(f'{title_prefix}Smoothed Probabilities (HMM Input) - Fold {fold}')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[0])
        cbar.set_label('Probability')
        
        # 中图：真实标签 vs HMM预测对比
        axes[1].plot(range(len(true_subset)), true_subset, 'b-', 
                    alpha=0.8, linewidth=2, label='True Labels', marker='o', markersize=2)
        axes[1].plot(range(len(hmm_subset)), hmm_subset, 'r-', 
                    alpha=0.8, linewidth=2, label='HMM Predictions', marker='s', markersize=2)
        
        axes[1].set_yticks(range(n_classes))
        axes[1].set_yticklabels(class_names)
        axes[1].set_ylabel('Sleep Stages')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('True vs HMM Optimized Predictions')
        
        # 下图：平滑预测 vs HMM预测对比
        axes[2].plot(range(len(orig_subset)), orig_subset, 'g--', 
                    alpha=0.7, linewidth=2, label='Smoothed Predictions', marker='^', markersize=2)
        axes[2].plot(range(len(hmm_subset)), hmm_subset, 'r-', 
                    alpha=0.8, linewidth=2, label='HMM Predictions', marker='s', markersize=2)
        
        axes[2].set_yticks(range(n_classes))
        axes[2].set_yticklabels(class_names)
        axes[2].set_ylabel('Sleep Stages')
        axes[2].set_xlabel('Time Points (Epochs)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Smoothed vs HMM Optimized Predictions')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            filename = f"{save_path}{model_name}_hmm_comparison_fold_{fold}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"HMM对比热力图已保存到: {filename}")
        
        plt.show()
        
        return fig
        
    finally:
        # 确保图形被关闭
        if 'fig' in locals():
            plt.close(fig)


##########################################################################################

