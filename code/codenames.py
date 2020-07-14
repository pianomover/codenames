import numpy as np
import cv2 as cv
import os
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\CHONGW2\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'

# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class codenamesClass():

    def __init__(self, image):
        self.image = image

    def redchannel(self):
        self.red_channel = self.image[:,:,2]

    def thresholdgen(self):
        # image needs to be grayscale
        self.redchannel()
        ret, self.thresh_red = cv.threshold(self.red_channel, int(self.red_channel.mean()), 255, cv.THRESH_BINARY)

    def contoursgen(self):
        # image needs to be binary
        self.thresholdgen()
        _, contours, _ = cv.findContours(self.thresh_red, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        return contours


    def clusteringgen(self, contours):
        areas = [cv.contourArea(arr) for arr in contours]
        X = np.array(areas).reshape((-1, 1))
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(X)

        return clusters

    def highlights(self, show):
        print('generating contours')
        contours = self.contoursgen()
        print('getting clusters')
        clusters = self.clusteringgen(contours)
        cluster_id, cluster_count = np.unique(clusters, return_counts=True)

        try:
            final_cluster = cluster_id[cluster_count == 25]
            # contours_new = [arr for arr in contours if cv.contourArea(arr) in X.reshape(-1)[clusters==final_cluster]]
            contours_new = np.array(contours)[clusters==final_cluster]
            contours_img = cv.drawContours(self.image, contours_new, -1, (0, 255, 0), 3)
            if show:
                cv.imshow('', contours_img)

            return contours_new
        except:
            print('something blew up')

    def croppedimage(self):
        contours_new = self.highlights(show=False)

        x_min, x_max, y_min, y_max = self.image.shape[1], 0, self.image.shape[0], 0

        for arr in contours_new:
            x_min = min(x_min, arr[:, :, 0].min())
            y_min = min(y_min, arr[:, :, 1].min())
            x_max = max(x_max, arr[:, :, 0].max())
            y_max = max(y_max, arr[:, :, 1].max())

        crop_img = self.image[y_min:y_max, x_min:x_max]
        # cv.imshow("cropped", crop_img)

        return crop_img


class extractWords():

    def __init__(self, image):
        '''

        :param image: cropped color image that will be processed
        '''
        self.image = image
        self.len_x = int(self.image.shape[1] / 5)
        self.len_y = int(self.image.shape[0] / 5)
        self.whites = np.floor(self.image.min(2) / 255) * 255
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        _, self.thresholded = cv.threshold(gray, int(self.whites.mean()), 255, cv.THRESH_BINARY)


    def gen_terms(self):
        '''
        generates a dictionary of terms
        :return: text_dict, dictionary of colors and terms
        '''
        text_dict = {'blue': [], 'red': [], 'black': [], 'neutral': []}

        for i in range(5):
            for j in range(5):
                square_crop = self.image[(self.len_y * i):(self.len_y * (i + 1)), (self.len_x * j):(self.len_x * (j + 1)), :]
                square_bw = self.thresholded[(self.len_y * i):(self.len_y * (i + 1)), (self.len_x * j):(self.len_x * (j + 1))]
                if square_bw.mean() < 210:  # get the black square
                    # using min image for black box
                    square_bw = 255 - self.whites[(self.len_y * i):(self.len_y * (i + 1)), (self.len_x * j):(self.len_x * (j + 1))]
                    colour = 'black'
                else:
                    if square_crop[:, :, 2].mean() < 150:  # check red channel
                        colour = 'blue'
                    elif square_crop[:, :, 0].mean() < 150:  # check blue channel
                        colour = 'red'
                    else:
                        colour = 'neutral'

                filename = os.path.join('square_bw.png')
                cv.imwrite(filename, square_bw)
                text = pytesseract.image_to_string(Image.open(filename))
                os.remove(filename)
                text_dict[colour].append(text)

        return text_dict


wd = os.getcwd()

if __name__ == '__main__':
    print('working directory:', wd)
    image = cv.imread(os.path.join(wd, 'data', 'codenames.jpg'))

    # create the class which allows you to preprocess the image
    haha = codenamesClass(image)

    # generate cropped image
    crop_img = haha.croppedimage()

    # write cropped image to disk
    filename = os.path.join(wd, 'data', 'crop.png')
    cv.imwrite(filename, crop_img)


    # extract words using extractWords class
    asdf = extractWords(crop_img)

    # extract terms
    terms = asdf.gen_terms()
    print(terms)