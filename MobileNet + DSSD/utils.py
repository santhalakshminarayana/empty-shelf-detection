import torch
import torch.nn as nn
import 
def xcycwh_to_xyminmax(a):
    # a is in shape [N,4] - (xc, yc, w, h)
    x_min = a[:, 0] - a[:, 2]/2
    y_min = a[:, 1] - a[:, 3]/2
    x_max = a[:, 0] + a[:, 2]/2
    y_max = a[:, 1] + a[:, 3]/2
    res = torch.stack([x_min, y_min, x_max, y_max], dim = 1)
    return res.clamp(min = 0.0, max = 1.0)

def xyminmax_to_xcycwh(a):
    # a is in shape [N,4] - (xmin, ymin, xmax, ymax)
    w = a[:, 2] - a[:, 0]
    h = a[:, 3] - a[:, 1]
    xc = (a[:, 0] + a[:, 2])/2
    yc = (a[:, 1] + a[:, 3])/2
    res = torch.stack([xc, yc, h, w], dim = 1)
    return res.clamp(min = 0.0, max = 1.0)

def iou(a, b):
    # a,b in format (x_min, y_min, x_max, y_max)
    x_int_max = torch.max(a[:, 0], b[:, 0])
    y_int_max = torch.max(a[:, 1], b[:, 1])
    x_int_min = torch.min(a[:, 2], b[:, 2])
    y_int_min = torch.min(a[:, 3], b[:, 3])
    int_area = (x_int_max - x_int_min) * (y_int_max - y_int_min)
    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = (int_area) / (a_area + b_area - int_area)
    return iou

def match_boxes(gb, db, th = 0.4):
    '''
    Param:
    ------
        gb : ground truth boxes in format (xc, yc, w, h)
                        shape - (n, 4) , n = no.of boxes in image
        db : default boxes in format (xc, yc, w, h)
                        shape - (N, 4), N = no.of default boxes
    '''
    is_box_matched = torch.zeros(default_boxes.shape[0])
    box_matched_index = torch.zeros(default_boxes.shape[0])

    n = ground_boxes.shape[0]
    N = default_boxes.shape[0]
    # expand ground and default boxes feasible to apply vector operations
    #gb = gb.repeat(N, 1)
    #db = db.repeat_interleave(n, 0)
    
    #scores = iou(xcycwh_to_xyminmax(gb),
    #             xcycwh_to_xyminmax(db))
    #print(scores)

    for i in range(n):
        xc, yc, w, h = gb[i][0], gb[i][1], gb[i][2], gb[i][3]
        x1_min = xc - w/2
        y1_min = yc - h/2
        x1_max = xc + w/2
        y1_max = yc + h/2

        x1_min.clamp_(0.0, 1.0)
        y1_min.clamp_(0.0, 1.0)
        x1_max.clamp_(0.0, 1.0)
        y1_max.clamp_(0.0, 1.0)

        x2_min = db[:, 0] - db[:, 2]/2
        y2_min = db[:, 1] - db[:, 3]/2
        x2_max = db[:, 0] + db[:, 2]/2
        y2_max = db[:, 1] + db[:, 3]/2

        x2_min.clamp_(0.0, 1.0)
        y2_min.clamp_(0.0, 1.0)
        x2_max.clamp_(0.0, 1.0)
        y2_max.clamp_(0.0, 1.0)

        x_max = torch.max(x1_min, x2_min)
        y_max = torch.max(y1_min, y2_min)
        x_min = torch.min(x1_max, x2_max)
        y_min = torch.min(y1_max, y2_max)

        int_area = (x_max - x_min) * (y_max - y_min)
        a_area = (x1_max - x1_min) * (y1_max - y1_min)
        b_area = (x2_max - x2_min) * (y2_max - y2_min)

        iou = (int_area) / (a_area + b_area - int_area)

        iou_ind_gt = (iou > 1.0).nonzero().squeeze()
        iou[iou_ind_gt] = 0
        iou_ind = (iou > th).nonzero().squeeze()
        is_box_matched[iou_ind] = 1
        box_matched_index[iou_ind] = i 

    return is_box_matched, box_matched_index

def encode_offset(gb, db, bmi):
    '''
    param:
    ------
        gb = ground boxes - shape (n, 4) (xc, yc, w, h)
        db = default boxes - shape(N, 4) (xc, yc, w, h)
        bmi = box matched index - 
            indicies of ground truth boxes maps to default boxes
            shape (N,)

    others:
    -------
        variance used is 0.1 for centers and 0.2 for height and width
    '''
    gb = gb[bmi.long()]
    xc = (gb[:, 0] - db[:, 0]) / (db[:, 2])
    yc = (gb[:, 1] - db[:, 1]) / (db[:, 3])
    w = torch.log(gb[:, 2] / db[:, 2])
    h = torch.log(gb[:, 3] / db[:, 3])
    return torch.stack([xc/0.1, yc/0.1, w/0.2, h/0.2], dim = 1)

def decode_offset(pb, db):
    '''
    param:
    ------
        pb = predicted boxes - shape (n, 4) (xc, yc, w, h)
        db = default boxes - shape(N, 4) (xc, yc, w, h)

    others:
    -------
        variance used is 0.1 for centers and 0.2 for height and width
    '''
    xc = (pb[:, 0] * 0.1 * db[:, 2]) + db[:, 0]
    yc = (pb[:, 1] * 0.1 * db[:, 3]) + db[:, 1]
    w = torch.exp(pb[:, 2] * 0.2 ) * db[:, 2]
    h = torch.exp(pb[:, 3] * 0.2 ) * db[:, 3]
    return torch.stack([xc, yc, w, h], dim = 1)

def net_loss(gb, db, pred_labels, pred_offset, neg_ratio = 3, neg_samples = 5):
    '''
    Param:
    ------
        gb : ground truth boxes in shape (n, xc, yc, w, h)
        db : default boxes in shape (N, xc, yc, w, h)
        pred_labels: confidence scores of box offsets (N)
        pred_offsets: local offsets in shape (N, xc, yc, w, h)
        neg_ratio : ratio of negative to positive samples for classification loss
        neg_samples: no.of samples to take for classification loss if matched boxes are 0
    '''
    
    is_box_matched, box_matched_index = match_boxes(gb, db)
    matched_boxes = (is_box_matched == 1).nonzero().squeeze()
    non_matched_boxes = (is_box_matched == 0).nonzero().squeeze()
    num_matches = 0
    if len(matched_boxes.size()) != 0:
        num_matches = matched_boxes.shape[0]
    true_offset = encode_offset(gb, db, box_matched_index)

    # regression loss
    regression_loss = 0.0
    if num_matches != 0:
        regression_loss_criterion = nn.SmoothL1Loss(reduction = 'none')
        regression_loss = regression_loss_criterion(true_offset[matched_boxes], pred_offset[matched_boxes])
        regression_loss = regression_loss.sum() / num_matches

    # classification loss
    classification_loss_criterion = nn.BCELoss(reduction = 'none')
    classification_loss = classification_loss_criterion(pred_labels, is_box_matched.to(device))
    negative_predictions = classification_loss[non_matched_boxes]
    _, negative_prediction_ids = torch.sort(negative_predictions, descending = True)
    if num_matches != 0:
        positive_predictions = classification_loss[matched_boxes]
        negative_predictions_ids = negative_prediction_ids[: num_matches * neg_ratio]
        classification_loss = (positive_predictions.sum() + negative_predictions[negative_prediction_ids].sum()) / (num_matches)

    # if no.of mathces = 0 take neg_samples in classification loss 
    if num_matches == 0:
        negative_predictions = negative_prediction_ids[: neg_samples]
        classification_loss = negative_predictions.sum() / neg_samples

    return regression_loss, classification_loss, matched_boxes

def plot_img(img, gray = False):
    fig=plt.figure(figsize=(5,10))
    ax=fig.add_subplot(111)
    if gray == False:
        #img = img[:, :, :: -1]
        ax.imshow(img)
    else:
        ax.imshow(img, cmap = 'gray')
    
    plt.xticks([]),plt.yticks([])
    plt.show()

def non_max_supression(pred_boxes, confs, conf_threshold = 0.5, iou_threshold = 0.1):
    '''
    Param:
    ------
        pred_boxes : predicted bounding boxes in shape (N, 4) - (xc, yc, w, h)
        confs : probabilities of bounding boxes in shape (N)

    Return:
    -------
        boxes : bounding boxes final in shape (N, 4) - (xmin, ymin, xmax, ymax)
    '''

    confs_ids = (confs > conf_threshold).nonzero().squeeze()
    x_min = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    y_min = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    x_max = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    y_max = pred_boxes[:, 1] + pred_boxes[:, 3] / 2 

    x_min.clamp_(0.0, 1.0)
    y_min.clamp_(0.0, 1.0)
    x_max.clamp_(0.0, 1.0)
    y_max.clamp_(0.0, 1.0)
    
    #print(x_min, y_min, x_max, y_max)
    confs = confs[confs_ids]
    if confs.shape[0] == 0:
        return 
    print(confs.shape)
    _, prob_inds = torch.sort(confs, descending = True)

    boxes = [[x_min[prob_inds[0]], y_min[prob_inds[0]],
             x_max[prob_inds[0]], y_max[prob_inds[0]]]]
    
    for ind in prob_inds[1:]:
        x11, y11, x12, y12 = x_min[ind], y_min[ind], x_max[ind], y_max[ind]
        a_area = (x12 - x11) * (y12 - y11)
        reject = False
        for i in range(len(boxes)):
            x21, y21, x22, y22 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            
            x_int_min = torch.max(x11, x21)
            y_int_min = torch.max(y11, y21)
            x_int_max = torch.min(x12, x22)
            y_int_max = torch.min(y12, y22)
            int_area = (x_int_max - x_int_min) * (y_int_max - y_int_min)
            b_area = (x22 - x21) * (y22 - y21)
            iou_area = (int_area) / (a_area + b_area - int_area)

            #print(iou_area)
            if iou_area > iou_threshold:
                reject = True
                break
            
        if reject == False:
            boxes.append([x11, y11, x12, y12])

    return torch.Tensor(boxes).tolist()