import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import cv2	

class mZoomCarToCameraDistance:
    def __init__(self):
        self.config = { 'method': 'TM_CCOEFF_NORMED', # 'TM_CCOEFF': TM_SQDIFF_NORMED' or 'TM_CCOEFF_NORMED'
                       'start_scale': 1.0,  # start by searching for the template at its original size
                        'end_scale': 0.1,    # search for the template at a reduced scale of 10% (0.1 times) of its original size.
                        'scale_step': 20, # perform template matching at 20 different scales between 100% (1 times) and 10% (0.1 times) of the original template size, with incremental steps of 5%
                        'distance_threshold': 250,
                        'high_area_coverage_threshold': 10,
                        'low_area_coverage_threshold': 6,
                        'reduce_percentage': 0.20,
                        'increase_percentage': 1.20,
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
        cv2.destroyAllWindows()


    def template_matching(self, X, visualize):

        ls_distance = []

        for x in X:
            best_match_val = -np.inf
            best_match_loc = None
            best_scale = None

            original_image = cv2.imread(x['original_image'])
            template_image = cv2.imread(x['zoom_image'])
            # here is same cv2.resize(original_image, (0,0), fx=0.5, fy=0.5)
            original_image = cv2.resize(original_image, (2016, 1512))
            template_image = cv2.resize(template_image, (2016, 1512))

            original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

            w, h = template_gray.shape[::-1]
            

            # Loop through scales
            for current_scale in np.linspace(self.config['start_scale'], self.config['end_scale'], num=self.config['scale_step']):

                resized_template = cv2.resize(template_gray, (int(w * current_scale), int(h * current_scale)))

                if resized_template.shape[0] > original_gray.shape[0] or resized_template.shape[1] > original_gray.shape[1]:
                    continue  # Skip this iteration

                if self.config['method'] == 'TM_CCOEFF':
                    result = cv2.matchTemplate(original_gray, resized_template, cv2.TM_CCOEFF)
                elif self.config['method'] == 'TM_CCOEFF_NORMED':
                    result = cv2.matchTemplate(original_gray, resized_template, cv2.TM_CCOEFF_NORMED)
                elif self.config['method'] == 'TM_SQDIFF_NORMED':
                    result = cv2.matchTemplate(original_gray, resized_template, cv2.TM_SQDIFF_NORMED)

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if visualize:
                	self.visualize(
                        original_image.copy(), 
                        max_loc, 
                        (max_loc[0] + int(w * current_scale), max_loc[1] + int(h * current_scale)), 
                        template_image=template_image,  # Assuming you have a variable named template_image holding your template
                        scale=current_scale, 
                        overlay=True, 
                        wait_time=2
                )

                if self.config['method'] == 'TM_CCOEFF' or self.config['method'] == 'TM_CCOEFF_NORMED':
                    if max_val > best_match_val:
                        best_match_val = max_val
                        best_match_loc = max_loc
                        best_scale = current_scale
                if self.config['method'] == 'TM_SQDIFF_NORMED':
                    if min_val < best_match_val or best_match_val == -1:
                        best_match_val = min_val 
                        best_match_loc = min_loc 
                        best_scale = current_scale
                
            top_left = best_match_loc
            bottom_right = (top_left[0] + int(w * best_scale), top_left[1] + int(h * best_scale))
            detected_area = int(w * best_scale) * int(h * best_scale)
            original_area = original_image.shape[0] * original_image.shape[1]

            if visualize:
                self.visualize(original_image, top_left, bottom_right, template_image, best_scale, overlay=True)

            distance = self.calculate_distance(x['distance'], detected_area, original_area)

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
    Y = mZoomCarToCameraDistance.predict(X[:], postprocess = True, visualize = False)
    print(Y)
