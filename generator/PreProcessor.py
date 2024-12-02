# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2024/11/29 Preprocessor.py

import os
import glob
import shutil
import numpy as np
import cv2

import scipy.io
import traceback

class PreProcessor:

  def __init__(self, size=512):
    self.RESIZE = (size, size)
    
  def preprocess(self, image_files_pattern, 
                 label_files_pattern, output_dir):
    
    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  = os.path.join(output_dir, "masks")

    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)

    image_files = glob.glob(image_files_pattern)
    mat_files   = glob.glob(label_files_pattern)
    image_files = sorted(image_files)
    mask_files  = sorted(mat_files)
    num_image_files = len(image_files)
    num_mask_files  = len(mask_files)
    if num_image_files != num_mask_files:
      raise Exception("Unmatched num_image_files and num_mask_file")

    for i in range(num_image_files):
      image_file = image_files[i]
      mat_file   = mat_files[i]
      mat = scipy.io.loadmat(mat_file)
      
      mask = mat["inst_map"]
      mask = np.array(mask, dtype=np.uint8)
      mask = cv2.resize(mask, self.RESIZE)
    
      filename = str(i+1) + ".jpg"

      output_image_file = os.path.join(output_images_dir, filename)
      output_mask_file  = os.path.join(output_masks_dir,  filename)
    
      image = cv2.imread(image_file)
      image = cv2.resize(image, self.RESIZE)
      cv2.imwrite(output_image_file, image)
      print("=== Saved {}".format(output_image_file))

      cv2.imwrite(output_mask_file, mask)
      print("=== Saved {}".format(output_mask_file))

if __name__== "__main__":
  try:
    category = "consep"

    image_files_pattern = "./Lizard_images1/Lizard_Images1/"      + category + "*.png"
    label_files_pattern = "./lizard_labels/Lizard_Labels/Labels/" + category + "*.mat"
    output_dir = "./Lizard-" + category + "-master"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)    

    preprocessor = PreProcessor()
    preprocessor.preprocess(image_files_pattern, 
                            label_files_pattern, output_dir)
  except:
    traceback.print_exc()
    
