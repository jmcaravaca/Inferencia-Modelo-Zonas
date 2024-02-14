# import the necessary packages
from random import randrange
import numpy as np
import time
import cv2
import os
from functools import cmp_to_key
import math



inference_width_zonas = 416
inference_height_zonas = 608    
inference_width_cajas = 416
inference_height_cajas = 416   

confianza_zonas = 0.5
umbral_zonas = 0.1
labels_zonas = ["Pregunta Test", "Pregunta Enunciado", "QR", "No Presentado/No Calificado", "Examinadores", "Barcode"]
confianza_cajas = 0.1
umbral_cajas = 0.1
labels_cajas = ["marca", "no_marca"]   


def infer_from_imgpath(imagen: str, net_zonas, net_cajas):
    if not (imagen.endswith(".jpg") or imagen.endswith(".JPG") or imagen.endswith(".png") or imagen.endswith(".PNG") or imagen.endswith(".jpeg") or imagen.endswith(".tiff") or imagen.endswith(".tif")):
        return ("Error")
    print("Procesando imagen " + imagen + "...")
    # load our input image and grab its spatial dimensions
    result = process_image(imagen, net_zonas, net_cajas)
    #pintar_resultados(imagen, result)
    return result["zonas"]  

def sort_cajas_fila_zona(boxes):
    matrix = []
    try:
        sorted_boxes = sorted(boxes, key=cmp_to_key(cajas_comparator(20)))        
        return sorted_boxes
    except Exception as ex:
        print("Error al ordenar las filas ")
        return matrix
    
# _________________________________________________________________________________________________________________________#
def cajas_comparator(close_value):
    def compare(box1, box2):
        if math.isclose(box1["y"], box2["y"], abs_tol = close_value):
            return (box1["x"] > box2["x"]) - (box1["x"] < box2["x"])
        else:
            return (box1["y"] > box2["y"]) - (box1["y"] < box2["y"])
    return compare

def process_image(imagen: str, net_zonas, net_cajas):
    image = cv2.imread(imagen)
    image_orig = image
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = np.stack((image,)*3, axis=-1)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net_zonas.getLayerNames()
    ln = [ln[i - 1] for i in net_zonas.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    start = time.time()
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (int(inference_width_zonas),int(inference_height_zonas)), swapRB=True, crop=False)
    net_zonas.setInput(blob)
    layerOutputs = net_zonas.forward(ln)
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
            if confidence > confianza_zonas:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes" width and height
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
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confianza_zonas, umbral_zonas)
    # ensure at least one detection exists
    zonas = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0] , boxes[i][1] )
            (w, h) = (boxes[i][2] , boxes[i][3] )
            clase_str = labels_zonas[classIDs[i]]
            cajaConfianza = str(round(confidences[i], 2))
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            zona = {"x": x, "y": y, "width": w, "height": h, "class": clase_str, "confidence": cajaConfianza}
            if classIDs[i] != 2 and classIDs[i] != 5: #Definit qué son clase 2 y clase 5
                cut_image = image[y:y+h, x:x+w]
                (H, W) = cut_image.shape[:2]

                # determine only the *output* layer names that we need from YOLO
                ln_cajas = net_cajas.getLayerNames()
                ln_cajas = [ln_cajas[i - 1] for i in net_cajas.getUnconnectedOutLayers()]

                # construct a blob from the input image and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes and
                # associated probabilities
                start = time.time()
                blob = cv2.dnn.blobFromImage(cut_image, 1 / 255.0, (int(inference_width_cajas),int(inference_height_cajas)), swapRB=True, crop=False)
                net_cajas.setInput(blob)
                layerOutputs = net_cajas.forward(ln_cajas)
                end = time.time()

                # show timing information on YOLO
                print("[INFO] YOLO took {:.6f} seconds".format(end - start))

                # initialize our lists of detected bounding boxes, confidences, and
                # class IDs, respectively
                boxes_cajas = []
                confidences_cajas = []
                classIDs_cajas = []

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
                        if confidence > confianza_cajas:
                            # scale the bounding box coordinates back relative to the
                            # size of the image, keeping in mind that YOLO actually
                            # returns the center (x, y)-coordinates of the bounding
                            # box followed by the boxes" width and height
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes_cajas.append([x, y, int(width), int(height)])
                            confidences_cajas.append(float(confidence))
                            classIDs_cajas.append(classID)

                # apply non-maxima suppression to suppress weak, overlapping bounding
                # boxes
                idxs_cajas = cv2.dnn.NMSBoxes(boxes_cajas, confidences_cajas, confianza_cajas, umbral_cajas)

                # ensure at least one detection exists
                cajas = []
                if len(idxs_cajas) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs_cajas.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes_cajas[i][0] , boxes_cajas[i][1] )
                        (w, h) = (boxes_cajas[i][2] , boxes_cajas[i][3] )
                        clase_str = labels_cajas[classIDs_cajas[i]]
                        cajaConfianza = str(round(confidences_cajas[i], 2))
                        if x < 0:
                            x = 0
                        if y < 0:
                            y = 0
                        cajas.append({"x": x, "y": y, "width": w, "height": h, "class": clase_str, "confidence": cajaConfianza})
                
                zona["cajas"] = sort_cajas_fila_zona(cajas)
            zonas.append(zona)           
    result_image = {"image": imagen, "zonas": sorted(zonas, key=lambda zona: (zona["y"], zona["x"]))}
    return result_image

