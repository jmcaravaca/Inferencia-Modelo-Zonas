# import the necessary packages
from random import randrange
import numpy as np
import time
import cv2
import os

direct_in = "img"
out = "result"
confianza = 0.5
umbral = 0.1
weightsPath = os.path.join(os.getcwd(), "zonas",  "zonas.weights")
configPath = os.path.join(os.getcwd(), "zonas" ,  "zonas.cfg")
inference_width = 416
inference_height = 608

confianza = float(confianza)
umbral = float(umbral)

# load the COCO class labels our YOLO model was trained on
LABELS = ["0", "1", "2", "3", "4", "5"]

# BGR Format
COLORS = [[0,0,255],[255,0,255],[255,0,0],[0,255,0],[128,0,128]]

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
id_general = 1000
print(direct_in)
for imagen in os.listdir(direct_in):
    
    if not (imagen.endswith(".jpg") or imagen.endswith(".JPG") or imagen.endswith(".png") or imagen.endswith(".PNG") or imagen.endswith(".jpeg") or imagen.endswith(".tiff") or imagen.endswith(".tif")):
        continue
    print(imagen)
    # load our input image and grab its spatial dimensions
    image = cv2.imread(os.path.join(direct_in, imagen))
    image_orig = image
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = np.stack((image,)*3, axis=-1)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    start = time.time()
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (int(inference_width),int(inference_height)),
        swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confianza:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confianza, umbral)

    # ensure at least one detection exists
    anchos = []
    altos = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0] , boxes[i][1] )
            (w, h) = (boxes[i][2] , boxes[i][3] )
            clase_str = [clase for clase in LABELS[classIDs[i]]]
            cajaConfianza = str(round(confidences[i], 2))
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            anchos.append(w)
            altos.append(h)
            
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            # ###################################
            # # IMAGEN
            # ###################################            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            text = "{}".format(clase_str[0])
            cv2.putText(image, text, (x-10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(image,str(cajaConfianza),(x, y + 10),cv2.FONT_HERSHEY_SIMPLEX ,0.5 ,(0,120,0) ,thickness=2)
            
    if not os.path.exists(out):
        os.makedirs(out)

    cv2.imwrite(os.path.join(out, imagen), image)
