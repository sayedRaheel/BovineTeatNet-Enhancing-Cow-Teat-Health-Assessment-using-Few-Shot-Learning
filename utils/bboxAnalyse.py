import pandas as pd
import numpy as np
import cv2
import networkx as nx
# return bbox and class
# the bbox form is xyxy
# the find_most_confident_bbox function is used to find the most confident bbox if there are bbox overlap area >0.8
def yolov8_detection(model, image):
    results = model(image,verbose=False)  # generator of Results objects

    boxes_list = []
    boxes = results[0].boxes  # Boxes object for bbox outputs
    class_id = results[0].boxes.cls.long().tolist()
    boxes_list.append(boxes.xyxy.tolist())

    bbox = [[int(i) for i in box] for boxes in boxes_list for box in boxes]
    # print(bbox, class_id, results[0].boxes.conf.tolist())
    bbox, class_id = find_most_confident_bbox(bbox, class_id, results[0].boxes.conf.tolist())
    return bbox, class_id, image,results

# input binary mask, bbox, class, output labeled mask
def label_mask(SAM_mask,bboxes, classes):
    labeled_mask = SAM_mask.copy().astype(np.uint8)
    for bbox, cls in zip(bboxes, classes):
        x1, y1, x2, y2 = bbox
        labeled_mask[y1:y2, x1:x2] = np.where(SAM_mask[y1:y2, x1:x2] == True, cls, labeled_mask[y1:y2, x1:x2])
        labeled_mask[labeled_mask==1]=0
    return labeled_mask


class_colors = {
    range(1, 61): (255, 0, 255),  # Purple
    61: (0, 0, 255),  # Red
    62: (0, 255, 0),  # Green
    63: (255, 0, 0),  # Blue
    64: (0, 255, 255)  # Yellow
}

# plot bbox on image
def plot_bbox(image, yolov8_boxex, yolov8_class_id, if_teat=False):
    # Conve
    for bbox, class_id in zip(yolov8_boxex, yolov8_class_id):
        if class_id <61:
            color = (255, 0, 255)
        elif if_teat:
            # Determine the color for this class
            color = class_colors[class_id]  # Default to white
        else: 
            continue
        # Draw the bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
        
        # Write the class id near the top-left corner of the box
        cv2.putText(image, str(class_id), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def yolo2xyxy(x,y,w,h,size):
    dw  = size[1]
    dh = size[0]
    x1 = int((x - w / 2) * dw)
    x2 = int((x + w / 2) * dw)
    y1 = int((y - h / 2) * dh)
    y2 = int((y + h / 2) * dh)
    
    if x1 < 0:
        x1 = 0
    if x2 > dw - 1:
        x2 = dw - 1
    if y1 < 0:
        y1 = 0
    if y2 > dh - 1:
        y2 = dh - 1
    return x1,y1,x2,y2


def calculate_iou(bbox1, bbox2):
    """
    This function calculates Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    bbox1, bbox2: list of coordinates [xmin, ymin, xmax, ymax]
    
    Returns:
    iou: Intersection over Union as a float
    """
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    
    # Determine the coordinates of the intersection rectangle
    xmin_inter = max(xmin1, xmin2)
    ymin_inter = max(ymin1, ymin2)
    xmax_inter = min(xmax1, xmax2)
    ymax_inter = min(ymax1, ymax2)
    
    # Calculate the area of intersection rectangle
    inter_area = max(0, xmax_inter - xmin_inter + 1) * max(0, ymax_inter - ymin_inter + 1)
    
    # Calculate the area of both bounding boxes
    bbox1_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    bbox2_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    
    # Calculate the area of union
    union_area = bbox1_area + bbox2_area - inter_area
    
    # Compute the IoU
    iou = inter_area / union_area if union_area != 0 else 0
    
    return iou

# because some bbox overlap area >0.8, we need to find the most confident bbox
def find_most_confident_bbox(bboxes, classes, confidence):
# Define bounding boxes, classes, and confidence

    # Create a graph where each node is a bounding box and each edge represents overlap > 0.9
    G = nx.Graph()
    # print(bboxes)
    for i in range(len(bboxes)):
        for j in range(i+1, len(bboxes)):
            if calculate_iou(bboxes[i], bboxes[j]) > 0.8:
                # print(bboxes[i], bboxes[j])
                G.add_edge(i, j)
    # print(G)
    # print(G.edges)
    # Find connected components, which correspond to sets of bounding boxes that should be merged
    connected_components = list(nx.connected_components(G))

    # For each group of bounding boxes that should be merged, replace them with the one with highest confidence
    merged_bboxes = []
    merged_classes = []
    merged_confidence = []
    for group in connected_components:
        # Find the bounding box in the group with the highest confidence
        max_confidence_idx = max(group, key=lambda idx: confidence[idx])
        merged_bboxes.append(bboxes[max_confidence_idx])
        merged_classes.append(classes[max_confidence_idx])
        merged_confidence.append(confidence[max_confidence_idx])

    # Add any bounding boxes that don't need to be merged
    for i in range(len(bboxes)):
        if i not in G:
            merged_bboxes.append(bboxes[i])
            merged_classes.append(classes[i])
            merged_confidence.append(confidence[i])
    return merged_bboxes, merged_classes


# return the location of teat LH,LF,RH,RF
def get_location(bboxes,classes,sn):
    df = pd.DataFrame(list(zip(bboxes, classes)), columns=['bbox', 'class'])

    df.sort_values(by=['bbox'], key=lambda x: x.str[1], inplace=True)

    # Separate top two and bottom two
    top_two, bottom_two = df.iloc[:2].copy(), df.iloc[2:4].copy()

    # Sort top two and bottom two by x1 coordinate
    top_two.sort_values(by=['bbox'], key=lambda x: x.str[0], inplace=True)
    bottom_two.sort_values(by=['bbox'], key=lambda x: x.str[0], inplace=True)

    # Create location dictionary
    location_dict = {}

    # Assign top-left and top-right
    location_dict['left-hind'] = (top_two.iloc[0]['bbox'], top_two.iloc[0]['class'])
    location_dict['right-hind'] = (top_two.iloc[1]['bbox'], top_two.iloc[1]['class'])

    # Assign bottom-left and bottom-right
    location_dict['left-front'] = (bottom_two.iloc[0]['bbox'], bottom_two.iloc[0]['class'])
    location_dict['right-front'] = (bottom_two.iloc[1]['bbox'], bottom_two.iloc[1]['class'])

    # Add stall number
    location_dict['stall'] = sn
    return location_dict

# if one image has two stall number, we need to judge does these two stall number is adjacent, such like 1 and 2, 2 and 3, 59 and 60, 
# if 56 and 58, we need to drop this image
def judge_stall_number(fs,ts):
    if fs ==1 and ts == 60:
        return True
    elif fs ==ts +1:
        return True
    else:
        return False

# input class_id and bbox, output the location of teat
def get_score(class_id, box):
    stall_numbers_id = []
    stall_numbers_box = []
    teat_id = []
    teat_box = []
    false_teat_id = []
    false_teat_box = []
    true_teat_id = []
    true_teat_box = []
    # collect stall number(1-60 are stall number)
    for i in range(len(class_id)):
        if class_id[i] < 61:
            stall_numbers_id.append(class_id[i])
            stall_numbers_box.append(box[i])
    # if there are none stall number or more than 2 stall number, drop this image
    if len(stall_numbers_id) < 1 or len(stall_numbers_id) > 2:
        return []
    # if there are two stall number at one image
    elif len(stall_numbers_id) == 2:
        # find which at left, which at right, first is right, second is left
        if stall_numbers_box[0][0] > stall_numbers_box[1][0]: # compare the x1 value
            first_stall = stall_numbers_id[0]
            first_box = stall_numbers_box[0]
            second_stall = stall_numbers_id[1]
            second_box = stall_numbers_box[1]
        else:
            first_stall = stall_numbers_id[1]
            first_box = stall_numbers_box[1]
            
            second_stall = stall_numbers_id[0]
            second_box = stall_numbers_box[0]
        # if two stall number is adjacent, drop this image
        if not judge_stall_number(second_stall,first_stall):
            return []
        # collect teat id and box, the teat between two stall number will be labeled as teat, the teat at left of second stall number will be labeled as false teat(actually don't need to care about false teat, we most use teat condition)
        for i in range(len(class_id)):
            if class_id[i]>60 and box[i][0] < first_box[0] and box[i][0] > second_box[0]:
                teat_id.append(class_id[i])
                teat_box.append(box[i])
            elif class_id[i]>60 and box[i][0] < second_box[0]:
                false_teat_id.append(class_id[i])
                false_teat_box.append(box[i])
        result = []
        # if there are 4 teat, we can get the location of teat, if not 4, drop this image
        if len(teat_id) ==4:
            location= get_location(np.array(teat_box),teat_id,first_stall)
            # if location["top-left"] is not None or location["top-right"] is  not None or location["bottom-left"] is not None or location["bottom-right"] is not None:
            if location is not None:
                result.append(location)
        if len(false_teat_id) == 4:
            location= get_location(np.array(false_teat_box),false_teat_id,second_stall)
            # if location["top-left"] is not None or location["top-right"] is  not None or location["bottom-left"] is not None or location["bottom-right"] is not None:
            if location is not None:
                result.append(location)
        return result
    # if there are one stall number at one image
    else:
        first_stall = stall_numbers_id[0]
        first_box = stall_numbers_box[0]
        # count the teat number at left side and right side
        for i in range(len(class_id)):
            if class_id[i]>60 and box[i][0] < first_box[0]:
                teat_id.append(class_id[i])
                teat_box.append(box[i])
            elif class_id[i]>60 and box[i][0] > first_box[0]:
                true_teat_box.append(box[i])
                true_teat_id.append(class_id[i])
        result = []
        # if there are 4 teat at one side, we can get the location of teat, if not 4, drop this image
        if len(teat_id) ==4:
            location= get_location(np.array(teat_box),teat_id,first_stall)
            # if location["top-left"] is not None or location["top-right"] is  not None or location["bottom-left"] is not None or location["bottom-right"] is not None:
            if location is not None:
                result.append(location)
        elif len(true_teat_id) == 4:
            if first_stall == 1:
                first_stall = 60
            else:
                first_stall = first_stall - 1
            location= get_location(np.array(true_teat_box),true_teat_id,first_stall)
            if location is not None:
                result.append(location)
        return result
    
# record score at scores
def record(scores,score_result):
    for i in score_result:
        scores[i['stall']][0][i['left-hind'][1]-61]+=1
        scores[i['stall']][1][i['left-front'][1]-61]+=1
        scores[i['stall']][2][i['right-hind'][1]-61]+=1
        scores[i['stall']][3][i['right-front'][1]-61]+=1


# fix load file problem
def convert_value(value):
    try:
        return int(value)
    except ValueError:
        if value in ['x','N/a']:
            return np.nan
        else:
            return value
# get ground truth label information, store at dict file        
def get_GT_label(video_name):
    df = pd.read_csv('label.csv')
    df = df[df['Video name']==video_name]
    df = df.drop(columns=['Tag ID', 'Group Number', 'Yield', 'Milking Time', 'Animal ID','Video name','Exit Time','Comments','Unnamed: 13'])
    df.set_index('Stall Number', inplace=True)
    df = df.applymap(convert_value)
# Convert DataFrame to dictionary
    result_dict = df.T.to_dict('list')
    return result_dict

# convert score to dict
def convert_scores(scores):
    scores = np.array(scores.copy()).astype(int)
    scores_table_nan = np.where(scores == 0, np.nan, scores)
    no_result_mask = np.isnan(scores_table_nan).all(axis=2)
    highest_count_scores = np.argmax(scores, axis=2)+1
    highest_count_scores = highest_count_scores.astype(float)
    highest_count_scores[no_result_mask] = np.nan
    dataset = highest_count_scores[1:, :]
    data_dict = {i: list(dataset[i-1]) for i in range(1, len(dataset) + 1)}
    return data_dict

# compare scores dict and GT dict
def cal_correct_count(result_dict,pred_dict):
    total_count = 0
    correct_count = 0
    class_total = np.zeros(4)
    correct_class_total = np.zeros(4)
    for i in result_dict:
        # print(result_dict[i])
        # print(data_dict[i])
        for j in range(4):
            if result_dict[i][j] >0 and result_dict[i][j] <5:
                # print(result_dict[i][j],data_dict[i][j])
                class_total[int(result_dict[i][j])-1] +=1
                if result_dict[i][j] == pred_dict[i][j]:
                    correct_class_total[int(result_dict[i][j])-1] +=1
        correct_count += np.sum(np.equal(result_dict[i],pred_dict[i]))
        total_count += len(result_dict[i])
    return total_count,correct_count,class_total,correct_class_total

def show_class_item(correct_class_total,class_total):
    print('score 1 correct:',correct_class_total[0], 'score 1 total:',class_total[0])
    print('score 2 correct:',correct_class_total[1], 'score 2 total:',class_total[1])
    print('score 3 correct:',correct_class_total[2], 'score 3 total:',class_total[2])
    print('score 4 correct:',correct_class_total[3], 'score 4 total:',class_total[3])