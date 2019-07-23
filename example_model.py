# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
from PIL import Image
import numpy as np
import face_recognition

class FaceTracker():

    def __init__(self, options):
        self.known_faces = {}
        self.index = 0
        pass

    # Process an input_image
    def process(self, input_image):
        input_image = np.array(input_image)
        height, width = input_image.shape[0:2]

        face_locations = face_recognition.face_locations(input_image)
        face_encodings = face_recognition.face_encodings(input_image, face_locations)

        found_faces = []

        for face_location, face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces([self.known_faces[f] for f in self.known_faces ], face_encoding)

            # if there is a match
            if True in matches:
                face_index = matches.index(True)
            
            # if there is no match
            else:
                self.known_faces[self.index] = face_encoding
                face_index = self.index
                self.index += 1

            top, right, bottom, left = face_location
            box = [ float(left)/width, float(top)/height, float(right)/width, float(bottom)/height ]

            found_faces.append({"index": face_index, "box": box})

        return found_faces
