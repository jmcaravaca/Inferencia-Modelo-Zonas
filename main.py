
import json
from yoloinference import infer
import cv2
import uuid
import os

class Main(object):
   def __init__(self):
      weightsPath = os.path.join(os.getcwd(), "zonas",  "zonas.weights")
      configPath = os.path.join(os.getcwd(), "zonas" ,  "zonas.cfg")
      self.yolonet = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
      self.yolonet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
      self.yolonet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

   def predict(self, skill_input):
      import io
      file_like = io.BytesIO(skill_input)
      # Write img to file
      with file_like as f:
         unique_id = uuid.uuid4()
         filepath = os.path.join("img", f"tempfile_{unique_id}.png")
         with open(filepath, 'wb') as output_file:
            output_file.write(file_like.read())         
      results = infer.infer_from_imgpath(filepath, self.yolonet)
      os.remove(filepath)
      return results
      
      
      
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