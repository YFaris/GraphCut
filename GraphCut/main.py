import cv2
from GraphCut import GraphCutMaster

if __name__ == '__main__':
    # read image
    image = cv2.imread("fig/CMS_livingroom.PNG")
    image = cv2.resize(image, (int(0.5 * image.shape[1]), int(0.5 * image.shape[0])), cv2.INTER_AREA)
    graph_cut = GraphCutMaster(image)
    mask = graph_cut.segmentation()