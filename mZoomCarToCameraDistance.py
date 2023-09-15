import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import cv2	
from sklearn.metrics import mean_absolute_percentage_error
class mZoomCarToCameraDistance:
    def __init__(self):
        self.config = { 'method': 'TM_CCOEFF_NORMED', # 'TM_SQDIFF_NORMED' or 'TM_CCOEFF_NORMED'
                       'start_scale': 0.1,  # or even smaller if necessary
                        'end_scale': 1.0,    # you might not need to go up to 1
                        'scale_step': 20,
                        'distance_threshold': 250,
                        'high_area_coverage_threshold': 10,
                        'low_area_coverage_threshold': 6,
                        'reduce_percentage': 0.50,
                        'increase_percentage': 1.50
                        }

    def load(self, pth_mdl = '' ):
        pass
    
    def preprocess(self, lst_image_links):
        images = []
        for link in lst_image_links:
            response = requests.get(link['image_link'])
            image = Image.open(BytesIO(response.content)).convert('RGB')
            img = np.asarray(image)
            images.append(img)
        return images

    def calculate_distance(self, distance,  detected_area=0, original_area=0):

        percentage_coverage = (detected_area / original_area) * 100

        is_high_distance = distance > self.config['distance_threshold']
        is_high_area_coverage = percentage_coverage > self.config['high_area_coverage_threshold']
        is_low_area_coverage = percentage_coverage < self.config['low_area_coverage_threshold']

        length_ratio = np.sqrt(detected_area / original_area)
        zoomed_distance = int(distance * length_ratio)

        # Adjust distance based on conditions
        if is_high_distance:
            if is_high_area_coverage:
                zoomed_distance *= self.config['reduce_percentage']  # Reducing the distance
            elif is_low_area_coverage:
                zoomed_distance *= self.config['increase_percentage']  # Increasing the distance

        return zoomed_distance

    def postprocess(self, predictions):
        results = []

        for pred in predictions:
            results.append({
                'distance': pred
            })

        return results
    
    def visualize(self, original_image, top_left, bottom_right, template_image=None, scale=None, overlay=False, wait_time=0):
        if overlay:
            resized_template_color = cv2.resize(template_image, (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))
            overlay_image = original_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv2.addWeighted(resized_template_color, 0.5, overlay_image, 0.5, 0, overlay_image)

        cv2.rectangle(original_image, top_left, bottom_right, 255, 2)
        title = f'Scale {scale}' if scale else 'Results'
        cv2.imshow(title, original_image)
        cv2.waitKey(wait_time)
        if wait_time == 0:  # close the window after the key press if wait_time is 0
            cv2.destroyAllWindows()


    def template_matching(self, X, visualize):

        ls_distance = []
        best_match_val = -1
        best_match_loc = None
        best_scale = None

        ls_pred = []
        ls_true = []

        for x in X:
            original_image = cv2.imread(x['original_image'])
            template_image = cv2.imread(x['zoom_image'])
            # here is same cv2.resize(original_image, (0,0), fx=0.5, fy=0.5)
            original_image = cv2.resize(original_image, (2016, 1512))
            template_image = cv2.resize(template_image, (2016, 1512))

            original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

            w, h = template_gray.shape[::-1]

            # Loop through scales
            for scale in np.linspace(self.config['start_scale'], self.config['end_scale'], self.config['scale_step'])[::-1]:

                resized_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))

                if resized_template.shape[0] > original_gray.shape[0] or resized_template.shape[1] > original_gray.shape[1]:
                    continue  # Skip this iteration

                if self.config['method'] == 'TM_CCOEFF_NORMED':
                    result = cv2.matchTemplate(original_gray, resized_template, cv2.TM_CCOEFF_NORMED)
                elif self.config['method'] == 'TM_SQDIFF_NORMED':
                    result = cv2.matchTemplate(original_gray, resized_template, cv2.TM_SQDIFF_NORMED)

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # if visualize:
                # 	self.visualize(original_image.copy(), max_loc, (max_loc[0] + int(w * scale), max_loc[1] + int(h * scale)), scale=scale, wait_time=2)

                if self.config['method'] == 'TM_CCOEFF_NORMED':
                    if max_val > best_match_val:
                        best_match_val = max_val
                        best_match_loc = max_loc
                        best_scale = scale
                if self.config['method'] == 'TM_SQDIFF_NORMED':
                    if min_val < best_match_val or best_match_val == -1:  # Change here
                        best_match_val = min_val  # Change here
                        best_match_loc = min_loc  # Change here
                        best_scale = scale

            top_left = best_match_loc
            bottom_right = (top_left[0] + int(w * best_scale), top_left[1] + int(h * best_scale))

            detected_area = int(w * best_scale) * int(h * best_scale)
            original_area = original_image.shape[0] * original_image.shape[1]

            if visualize:
                self.visualize(original_image, top_left, bottom_right, template_image, best_scale, overlay=True)

            distance = self.calculate_distance(x['distance'], detected_area, original_area)
            
            ls_distance.append(distance)
            ls_pred.append(distance)
            x['zoom_distance'] = int(os.path.basename(x['zoom_image']).split('_')[1].split('.')[0])
            ls_true.append(x['zoom_distance'])

        print(ls_true)
        print(ls_pred)
        print('MAPE: ', (mean_absolute_percentage_error(ls_true, ls_pred) * 100))
        return ls_distance

    def predict(self, X = None, postprocess = True, visualize = True):
        
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
    Y = mZoomCarToCameraDistance.predict(X[:], postprocess = True, visualize = False)
    print(Y)

