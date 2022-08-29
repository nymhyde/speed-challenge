import cv2
import os
import sys
import queue

import numpy as np
from sklearn import linear_model
from utils import *


class Solver():

    def __init__(self, params):
        self.vid = cv2.VideoCapture(params.train_vid)
        self.txt = params.train_txt
        self.vis = bool(params.vis)
        self.L = params.len_gt
        self.test_vid = cv2.VideoCapture(params.test_vid)
        self.lk_params = dict(winSize = (21, 21),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

        self.frame_idx = 0
        self.prev_pts = None
        self.detect_interval = 1
        self.temp_preds = np.zeros(int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        #self.gt = np.reshape(np.loadtxt(self.txt), (len(np.loadtxt(self.txt)),1))
        self.gt = np.loadtxt(self.txt)

        self.window = 100 
        self.prev_gray = None
        self.split = 0.9 
        self.xs = 35
        self.ys = 130


    def createMask(self, mask=None, test=False):
        '''
        Create a Mask to only see certain portion of the video.
        Here, it is part of the road that has next to none overlap with
        other vehicles or the ego vehicle itself.
        '''

        vid = self.test_vid if test else self.vid

        if mask is None :
            W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

            mask = np.zeros(shape= (H,W), dtype=np.uint8)
            mask.fill(255)

        else:
            W = mask.shape[1]
            H = mask.shape[0]

        cv2.rectangle(mask, (0,0), (W,H), (0,0,0), -1)
        poly_pts = np.array([[ [400, 250], [240, 250], [65, 350], [575, 350] ]], dtype=np.int32)
        cv2.fillPoly(mask, poly_pts, (255, 255, 255))

        return mask

    

    def processFrame(self, frame):
        '''
        takes an image frame --> apply Lucas Kanade Optical Flow
        '''
        frame = cv2.GaussianBlur(frame, (3,3), 0)
        curr_pts, _str , _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame, self.prev_pts, None, **self.lk_params)

        # store flow (x, y, dx, dy)
        flow = np.hstack((self.prev_pts.reshape(-1, 2), (curr_pts - self.prev_pts).reshape(-1, 2)))

        preds = []

        for x, y, u, v in flow:
            if abs(u) - abs(v) > 11:
                preds.append(0)
                preds.append(0)


            # translate points to center
            x -= frame.shape[1] / 2
            y -= frame.shape[0] / 2
           
            if (x and y) != 0:
                preds.append(u / (x*y))
                preds.append(v / (y*y))

            else:
                preds.append(0)
                preds.append(0)

        return [n for n in preds if n>=0]



    def getPts(self, offset_x=0, offset_y=0):
        '''
        return key points to track with offset
        '''
        if self.prev_pts is None:
            return None
        
        return [cv2.KeyPoint(x=p[0][0] + offset_x, y=p[0][1] + offset_y, _size=10) for p in self.prev_pts]



    def getFeatures(self, frame_gray, mask):
        return cv2.goodFeaturesToTrack(frame_gray, 10, 0.01, 10, blockSize=10, mask=mask)


    def stat(self):
        
        # split preds into train and validation
        split = int(round(self.frame_idx*self.split))
        train_preds = self.temp_preds[:split]
        val_preds = self.temp_preds[split:self.frame_idx]

        gt_train = self.gt[:len(train_preds)]
        gt_val = self.gt[len(train_preds):self.frame_idx]
        
        # fit to ground truth
        preds = movingAverage(train_preds, self.window)

        lin_reg = linear_model.LinearRegression(fit_intercept=False)
        lin_reg.fit(preds.reshape(-1, 1), gt_train)
        hf_factor = lin_reg.coef_[0]
        print(f'Estimated hf factor : {hf_factor:.4f}')

        # estimate train error
        pred_speed_train = train_preds * hf_factor
        pred_speed_train = movingAverage(pred_speed_train, self.window)
        mse = np.mean((pred_speed_train - gt_train)**2)
        print(f' MSE for train : {mse:.4f}')

        # plot(pred_speed_train, gt_train)

        # estimate val error
        pred_speed_val = val_preds * hf_factor
        pred_speed_val = movingAverage(pred_speed_val, self.window)
        mse = np.mean((pred_speed_val - gt_val)**2)
        print(f' MSE for val : {mse:.4f}')
        
        # plot(pred_speed_val, gt_val)

        return hf_factor



    def visualize(self, frame, mask_vis, prev_key_pts, speed=None):
        
        mask_vis = self.createMask(mask_vis)

        mask_vis = cv2.bitwise_not(mask_vis)
        frame_vis = cv2.addWeighted(mask_vis, 0.2, frame, 0.8, 0)
        key_pts = self.getPts(self.xs, self.ys)
        cv2.drawKeypoints(frame_vis, key_pts, frame_vis, color=(0,0,255))
        cv2.drawKeypoints(frame_vis, prev_key_pts, frame_vis, color=(0,255,0))

        if speed is not None:
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_vis, f' Speed : {speed}', (10,50), font, 1.2, (0, 0, 255))

        cv2.imshow('test', frame_vis)

        return key_pts


    
    def test(self, hf_factor, save_txt=False):
        mask = self.createMask(test=True)

        self.prev_gray = None
        test_preds = np.zeros(int(self.test_vid.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_idx = 0
        curr_estimate = 0
        prev_key_pts = None
        self.prev_pts = None


        while self.test_vid.isOpened():
            ret, frame = self.test_vid.read()
            if not ret:
                break

            # convert to B/W
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_gray = frame_gray[130:350, 35:605]
            mask_vis = frame.copy()

            pred_speed = 0
            if self.prev_pts is None:
                test_preds[frame_idx] = 0
            else:
                preds = self.processFrame(frame_gray)
                pred_speed = np.median(preds) * hf_factor if len(preds) else 0
                test_preds[frame_idx] = pred_speed


            self.prev_pts = self.getFeatures(frame_gray, mask[130:350, 35:605])
            self.prev_gray = frame_gray
            frame_idx += 1

            vis_pred_speed = computeAverage(test_preds, self.window//2, frame_idx)
            test_pred_speed = movingAverage(test_preds, self.window)
            prev_key_pts = self.visualize(frame, mask_vis, prev_key_pts, speed=vis_pred_speed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        self.test_vid.release()

        print(f'Saving predicted speeds in test.txt')
        np.savetxt('test.txt', test_pred_speed)



    def run(self):

        # construct mask
        mask = self.createMask()
        print(mask.shape)
        prev_key_pts = None

        while self.vid.isOpened() and self.frame_idx < len(self.gt):
            ret, frame = self.vid.read()

            if not ret:
                break
            # conver to black&white
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray[self.ys:480-self.ys, self.xs:640-self.xs]
            mask_vis = frame.copy()

            # process each frame
            if self.prev_pts is None:
                self.temp_preds[self.frame_idx] = 0
            else:
                # get preds for V/hf values
                preds = self.processFrame(frame_gray)
                # get median of these values
                self.temp_preds[self.frame_idx] = np.median(preds) if len(preds) else 0

            # extract features
            self.prev_pts = self.getFeatures(frame_gray, mask[self.ys:480-self.ys, self.xs:640-self.xs])
            self.prev_gray = frame_gray
            self.frame_idx += 1
        
            # for visualization <-- 'cause why not ??
            if self.vis:
                prev_key_pts = self.visualize(frame, mask_vis, prev_key_pts)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

        # self.video.release()
        self.vid.release()

        # get stats
        hf = self.stat()

        # test and save against test.mp4
        self.test(hf)



