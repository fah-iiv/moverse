import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import cv2 as cv

class mZoomCarToCameraDistance:
    def __init__(self) -> None:
        self.config = { 'method': 'cv.TM_CCOEFF', # 'cv.TM_CCOEFF': cv.TM_SQDIFF_NORMED' or 'cv.TM_CCOEFF_NORMED'
                'start_scale': 0.2,  # start by searching for the template at its original size
                'end_scale': 1.0,    # search for the template at a reduced scale of 10% (0.1 times) of its original size.
                'scale_step': 20, # perform template matching at 20 different scales between 100% (1 times) and 10% (0.1 times) of the original template size, with incremental steps of 5%
                'distance_threshold': 200,
                'high_area_coverage_threshold': 10,
                'low_area_coverage_threshold': 6,
                'reduce_percentage': 0.30,
                'increase_percentage': 1.30,
                }

    def postprocess(self, predictions):
        results = []

        for pred in predictions:
            results.append({
                'distance': pred
            })

        return results
    
    def calculate_distance(self, distance,  detected_area=0, original_area=0):

        percentage_coverage = (detected_area / original_area) * 100

        is_high_distance = distance > self.config['distance_threshold']
        is_high_area_coverage = percentage_coverage > self.config['high_area_coverage_threshold']
        is_low_area_coverage = percentage_coverage < self.config['low_area_coverage_threshold']

        length_ratio = np.sqrt(detected_area / original_area)
        zoomed_distance = int(distance * length_ratio)

        # Adjust distance based on conditions
        if is_high_distance:
            print("High distance")
            if is_high_area_coverage:
                print("High area coverage")
                zoomed_distance *= self.config['reduce_percentage']  # Reducing the distance
            elif is_low_area_coverage:
                print("Low area coverage")
                zoomed_distance *= self.config['increase_percentage']  # Increasing the distance
        return zoomed_distance


    def template_matching(self, X, visualize):
        
        ls_distance = []

        method = eval(self.config['method'])
        scales = np.linspace(self.config['start_scale'],self.config['end_scale'], self.config['scale_step'])

        for x in X:
            img = cv.imread(x['original_image'], cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (3226, 2419)) # cv.resize(img, (0,0), fx=0.8, fy=0.8)

            img2 = img.copy()
            template = cv.imread(x['zoom_image'], cv.IMREAD_GRAYSCALE)
            template = cv.resize(template, (3226, 2419))


            best_score = -np.inf
            best_scale = None
            best_top_left = None
            best_w, best_h = 0, 0


            # Define scales with more granularity
            for scale in scales:
                resized_template = cv.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
                w, h = resized_template.shape[::-1]
                print(f"Scale: {scale:.2f}, Template size: {w}x{h}")

                # Skip if the template is larger than the image in either dimension
                if img.shape[0] < h or img.shape[1] < w:
                    continue

                res = cv.matchTemplate(img, resized_template, method)
                _, max_val, _, max_loc = cv.minMaxLoc(res)

                # Normalizing the score
                normalized_score = max_val / (w * h)

                if normalized_score > best_score:
                    best_score = normalized_score
                    best_scale = scale
                    best_top_left = max_loc
                    best_w, best_h = w, h

            # print(f"Best scale is: {best_scale:.2f} with normalized score of {best_score:.2f}")

            detected_area = best_w * best_h
            original_area = img2.shape[0] * img2.shape[1]

            distance = self.calculate_distance(x['distance'], detected_area, original_area)

            # print(f"Distance: {distance}")

            if visualize:
                bottom_right = (best_top_left[0] + best_w, best_top_left[1] + best_h)
                cv.rectangle(img2, best_top_left, bottom_right, 255, 2)
                cv.imshow('Best Match using Direct Multi-Scale', img2)
                cv.waitKey(0)

            if distance >= x['distance']:
                ls_distance.append(-1)
            else:
                ls_distance.append(distance)

        return ls_distance


    def predict(self, X = None, postprocess = True, visualize = False):
        
        Y = self.template_matching(X, visualize)

        if postprocess:
            Y = self.postprocess(Y)
        
        return Y

if __name__ == '__main__':

    X = [
        {'original_image': 'data/IIVMoverse_Component_mCarToCameraDistance/input/original/1_266.jpg', 
    'zoom_image':'data/IIVMoverse_Component_mCarToCameraDistance/input/zoom/1_73.jpg',
    'distance': 266},
    {'original_image': 'data/IIVMoverse_Component_mCarToCameraDistance/input/original/2_134.jpg', 
    'zoom_image':'data/IIVMoverse_Component_mCarToCameraDistance/input/zoom/2_45.jpg',
    'distance': 134},
    {'original_image': 'data/IIVMoverse_Component_mCarToCameraDistance/input/original/3_211.jpg', 
    'zoom_image':'data/IIVMoverse_Component_mCarToCameraDistance/input/zoom/3_72.jpg',
    'distance': 211},
    {'original_image': 'data/IIVMoverse_Component_mCarToCameraDistance/input/original/4_318.jpg', 
    'zoom_image':'data/IIVMoverse_Component_mCarToCameraDistance/input/zoom/4_147.jpg',
    'distance': 318},
    ]

    mZoomCarToCameraDistance = mZoomCarToCameraDistance()
    Y = mZoomCarToCameraDistance.predict(X[:], postprocess = True, visualize = True)
    print(Y)
