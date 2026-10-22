import numpy as np
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings('ignore')

def parse_metric_for_print(metric):
    if isinstance(metric, float):
        return f"{metric:.4f}"
    return str(metric)

def get_test_metrics(y_pred, y_true, img_names):
    """计算测试指标"""
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    return {'auc': auc, 'ap': ap, 'eer': eer}

def get_video_metrics(image, pred, label):
    result_dict = {}
    new_label = []
    new_pred = []
    for item in np.transpose(np.stack((image, pred, label)), (1, 0)):
        s = item[0]
        if '\\' in s:
            parts = s.split('\\')
        else:
            parts = s.split('/')
        a = parts[-2]
        if a not in result_dict:
            result_dict[a] = []
        result_dict[a].append(item)
    
    for key, value in result_dict.items():
        pred_sum = 0
        label_sum = 0
        leng = 0
        for frame in value:
            pred_sum += frame[1]
            label_sum += int(frame[2])
            leng += 1
        new_pred.append(pred_sum / leng)
        new_label.append(int(label_sum / leng))
    
    new_pred = np.array(new_pred)
    new_label = np.array(new_label)
    
    if np.isnan(new_pred).any():
        print(f"警告: new_pred 包含 NaN，替换为 0")
        new_pred = np.nan_to_num(new_pred, nan=0.0)
    
    fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
    v_auc = metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return v_auc, v_eer
