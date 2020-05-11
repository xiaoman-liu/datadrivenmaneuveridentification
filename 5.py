
class Solution:
    def help(self, arr) :
        return sorted(arr, key=lambda i: [bin(i).count('1'), i])


if __name__ == "__main__":
    a = Solution()
    arr = [0,1,2,3,4,5,6,7,8]
    res = a.help(arr)
    print(res) # res = [0, 1, 2, 4, 8, 3, 5, 6, 7]




import numpy as np
def nms(bbox,iou_threld):
    """
    :param bbox: np.array,shape = (n,6),n :number of box,6:(xmin,ymin,xmax,ymax,confidence,cls)
    :param iou_threld: int
    :return: bbox_nms
    """

    classes = bbox[:,5]
    unique_class = np.unique(classes)
    max_bbox = []

    for cls in unique_class:
        mask = classes == cls
        bbox_on_class = bbox[mask]
        score = bbox_on_class[:,4]
        max_bbox_index = []
        order = np.argsort(score)[::-1]
        x1, y1 = bbox_on_class[:, 0], bbox_on_class[:, 1]  # (n,)
        x2, y2 = bbox_on_class[:, 2], bbox_on_class[:, 3]  # (n,)
        area = (x2 - x1) * (y2 - y1)

        while order.size > 0:
            i = order[0]
            max_bbox_index.append(i)
            x1_max = np.maximum(x1[order[0]],x1[order[1:]]) #(n-1)
            y1_max = np.maximum(y1[order[0]],y1[order[1:]]) #(n-1)
            x2_min = np.minimum(x2[order[0]],x2[order[1:]]) #(n-1)
            y2_min = np.minimum(y2[order[0]],y2[order[1:]]) #(n-1)

            iw = max(0,x2_min - x1_max) #(n-1)
            ih = max(0,y2_min - y1_max) #(n-1)
            inter_area = iw * ih
            iou = area[i] + area[order[1:]] - inter_area
            keep_index = np.where(iou <= iou_threld)[0]
            order = order[keep_index+1]
        max_bbox.append(bbox[max_bbox_index])
    max_bbox = np.vstack(max_bbox)
    return max_bbox








