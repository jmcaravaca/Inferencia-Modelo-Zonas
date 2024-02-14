
import json
from yoloinference import infer
import cv2
import uuid
import os

class Main(object):
   def __init__(self):
      self.load_zonas()
      self.load_cajas()


   def load_cajas(self):
      print("[INFO] loading YOLOCAJAS from disk...")
      weightsPathCajas = os.path.join(os.getcwd(), "cajasmarca",  "cajasmarca.weights")
      configPathCajas = os.path.join(os.getcwd(), "cajasmarca" ,  "cajasmarca.cfg")
      self.yolonet_cajas = cv2.dnn.readNetFromDarknet(configPathCajas, weightsPathCajas)
      self.yolonet_cajas.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
      self.yolonet_cajas.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)            
      

   def load_zonas(self):
      print("[INFO] loading YOLOZONAS from disk...")
      weightsPathZonas = os.path.join(os.getcwd(), "zonas",  "zonas.weights")
      configPathZonas = os.path.join(os.getcwd(), "zonas" ,  "zonas.cfg")
      self.yolonet_zonas = cv2.dnn.readNetFromDarknet(configPathZonas, weightsPathZonas)
      self.yolonet_zonas.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
      self.yolonet_zonas.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)      
      

   def predict(self, skill_input):
      import io
      file_like = io.BytesIO(skill_input)
      # Write img to file
      with file_like as f:
         unique_id = uuid.uuid4()
         filepath = os.path.join("img", f"tempfile_{unique_id}.png")
         with open(filepath, 'wb') as output_file:
            output_file.write(f.read())         
      results = infer.infer_from_imgpath(imagen=filepath,
                                         net_zonas=self.yolonet_zonas, net_cajas=self.yolonet_cajas)
      os.remove(filepath)
      return json.dumps(results)
      
      
      
if __name__ == '__main__':
   a = Main()
   import os
   for filename in os.listdir("testdata"):
      try:
         print(f"....{filename}.....")
         with open(os.path.join("testdata", filename), 'rb') as f:
               results = a.predict(skill_input=f.read())
               print(results)
      except Exception as e:
         print(e)