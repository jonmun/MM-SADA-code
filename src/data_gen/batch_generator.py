import numpy as np
from random import randint
import random
import pandas as pd
import os
from random import shuffle
import itertools
# cv2
from itertools import groupby


class BatchGenerator:
    """
    Iterate over a video datasets, returning filenames of frames to laod.
    preprocessing.py can be used in combination with BatchGenerator to read and preprocess frames.

    (num_labels) number of labels in dataset.
    (filename) part of filename before _train.pkl or _test.pkl in annotations folder (e.g. Annotations/D1).
    (temporal_window) number of frames to be sampled in a single video segment.
    (flow_data_path) Path to folder containing flow frames.
    (rgb_data_path) Path to folder containing rgb frames.
    (random_sync) Half the batch with corresponding examples, half the batch with non-corresponding examples.
    (synchronised) flow and rgb frames are temporally synchronised for corresponding examples.

    nextBatch - returns rgb and flow filenames of frames to load in a training batch, labels, and correspondence labels
    nextBatchEval - returns rgb and flow filenames of frames to load in a training batch, labels, and correspondence labels
    """
    def __init__(self,num_labels,filename, temporal_window=16,
                 flow_data_path="flow_frames_parent/flow",
                 rgb_data_path="rgb_frames_parent/frames",
                 synchronised=False, random_sync=False):
        self.random_sync=random_sync
        self.synchronised = synchronised
        self.temporal_window = temporal_window
        self.flow_data_path = flow_data_path+"/"
        self.rgb_data_path = rgb_data_path+"/"
        self.filename = filename
        self.num_labels = num_labels

        dataset_data, dataset_labels = self._parse_inputs_df(
            filename+"_train.pkl")
        dataset_data_test, dataset_labels_test = self._parse_inputs_df(
            filename+"_test.pkl")

        dataset_data_total = np.arange(dataset_data.shape[0])
        dataset_test_total = np.arange(dataset_data_test.shape[0])

        dataset_data_train = (dataset_data,dataset_labels)
        dataset_data_test = (dataset_data_test, dataset_labels_test)
        self.dataset_data = {False: dataset_data_train, True:dataset_data_test}
        dataset_data_train_total = dataset_data_total
        dataset_data_test_total = dataset_test_total
        self.dataset_total = {False: dataset_data_train_total,True: dataset_data_test_total}

    def reset_dataset(self, test=False):
        """ Reset dataset iterator to include all data"""
        self.dataset_total[test] = np.arange(self.dataset_data[test][0].shape[0])

    def _parse_inputs_df(self, filename):
        """ Read annotation file """
        df = pd.read_pickle(filename)
        data = []
        for _, line in df.iterrows():
            image = [line['participant_id']+"/"+line['video_id'], line['start_frame'], line['stop_frame']]
            labels = line['verb_class']
            one_hot = np.zeros(self.num_labels)
            one_hot[labels] = 1.0
            data.append((image[0],image[1],image[2], one_hot))

        segment,start,end, softmaxlabels = list(zip(*data))

        # get labels for all
        labels = list(softmaxlabels)
        train = list(zip(segment,start,end))
        train = np.array(train)
        labels = np.array(labels)
        return train, labels

    def nextBatch(self, batch_size, test=False):
        """ Get next training samples
            loops through datasets, randomly sampling data without replacement to add to batch."""
        batch_size = int(batch_size)
        dataset_data, dataset_labels = self.dataset_data[test]
        dataset_total = self.dataset_total[test]
        file = self.filename

        # Find path to dataset
        if len(dataset_total) > batch_size:
            sample_idx_t = np.random.choice(range(dataset_total.shape[0]),size = batch_size,replace=False)
            sample_idx = dataset_total[sample_idx_t]
            self.dataset_total[test] = np.delete(dataset_total, sample_idx_t, axis=0)
        else:
            sample_idx = dataset_total
            remaining = batch_size - sample_idx.shape[0]
            dataset_total_temp = np.arange(dataset_data.shape[0])
            dataset_total_temp = np.delete(dataset_total_temp, sample_idx, axis=0)
            sample_idx_t = np.random.choice(range(dataset_total_temp.shape[0]), size=remaining, replace=False)
            sample_idx = np.concatenate((sample_idx,dataset_total_temp[sample_idx_t]))
            self.dataset_total[test] = np.delete(dataset_total_temp, sample_idx_t, axis=0)
            print("Done Epoch")

        sample = dataset_data[sample_idx]

        if self.synchronised:
            sychron = [True]*len(sample)
        else:
            sychron = [False]*len(sample)
        sample = [self.sample_segment(filen, synchronise=to_sync) for (filen, to_sync) in zip(sample, sychron)]
        sample_rgb, sample_flow = zip(*sample)
        if self.random_sync:
            half = int(len(sample)/2)
            fixed_sample_rgb = sample_rgb[:half]
            fixed_sample_flow = sample_flow[:half]
            variate_sample_rgb = sample_rgb[half:]
            variate_sample_flow = sample_flow[half:]
            variate_sample_flow = variate_sample_flow[1:] + variate_sample_flow[:1]
            sample_flow = fixed_sample_flow + variate_sample_flow
            sample_rgb = fixed_sample_rgb + variate_sample_rgb
            sychron = [True]*len(fixed_sample_rgb)+[False]*len(variate_sample_rgb)
        elif self.synchronised:
            sychron = [True] * len(sample)
        else:
            sychron = [True] * len(sample)
        sample_labels = dataset_labels[sample_idx]

        batch_labels = sample_labels
        batch_rgb = list(sample_rgb)
        batch_flow = list(sample_flow)

        # Shuffle Batch
        combined = list(zip(batch_rgb,batch_flow,batch_labels,sychron))
        shuffle(combined)
        batch_rgb, batch_flow, batch_labels,sychron = list(zip(*combined))

        batch_rgb = np.array(batch_rgb)
        batch_flow = np.array(batch_flow)
        return batch_rgb,batch_flow,batch_labels,sychron

    def nextBatchEval(self, batch_size, test=True):
        """ Get next testing samples, return 5 equidistant frames along a action segment
            loops through datasets, randomly sampling data without replacement to add to batch. """
        dataset_data, dataset_labels = self.dataset_data[test]
        dataset_total = self.dataset_total[test]
        dataset_data = dataset_data
        dataset_labels = dataset_labels
        dataset_total = dataset_total

        batch_rgb = []
        batch_flow = []
        batch_labels = np.empty(shape=[0,self.num_labels])
        done = True
        if len(dataset_total) != 0:
            done = False

            #samples segment/frame
            if len(dataset_total) > batch_size:
                sample_idx = np.random.choice(range(dataset_total.shape[0]), size=batch_size, replace=False)
            else:
                sample_idx = range(dataset_total.shape[0])
                done = True

            # read and sample frames
            sample = dataset_data[dataset_total[sample_idx]]
            # read and sample frames
            sample = [self.sample_segment_test(filen) for filen in sample]
            sample_rgb,sample_flow = zip(*sample)

            sample_labels = dataset_labels[dataset_total[sample_idx]]

            #remove frames/segments from epoch
            self.dataset_total[test] = np.delete(dataset_total,sample_idx,axis=0)

            #create batch
            batch_labels = np.concatenate((sample_labels,batch_labels))
            batch_rgb = list(sample_rgb) + batch_rgb
            batch_flow = list(sample_flow) + batch_flow
        #reset dataset if all frames/segments have been evaluatated
        if done:
            self.dataset_total[test] = np.arange(dataset_data.shape[0])

        batch_rgb = np.array(batch_rgb)
        batch_flow = np.array(batch_flow)
        return done,batch_rgb,batch_flow,batch_labels

    def sample_segment_test(self, s):
        """ Samples rgb and flow frame windows from a video segment s.
            Sampling 5 windows, equidistant along a video segment
            s = ["filename", start_frame, end_frame]"""

        def _path_to_dataset(flow):
            if flow:
                left = self.flow_data_path
            else:
                left = self.rgb_data_path
            right = "/frame_"
            numframe = 10
            return left, right, numframe

        #Optical flow stacking
        #returns u,v frames for EPIC Kitchens
        def flow_filename(frameno,num_stack=1):
            left, right,fill_frame = _path_to_dataset(True)
            left_frame = frameno - int((num_stack-1)/2)
            right_frame = frameno + int(num_stack/2)
            filename = []
            for no in range(left_frame, right_frame+1):
                filename.append(left + str(s[0]) + "/u" + right + str(no).zfill(fill_frame) + ".jpg")
                filename.append(left + str(s[0]) + "/v" + right + str(no).zfill(fill_frame) + ".jpg")
            return filename

        #return RGB frame for EPIC Kitchens
        def rgb_filename(frameno):
            left, right, fill_frame = _path_to_dataset(False)
            filename = left + str(s[0]) + right + str(frameno).zfill(fill_frame) + ".jpg"
            return filename

        def c3d_sampling():
            num_sample_frame = self.temporal_window
            half_sample_frame = int(self.temporal_window/2)
            segment_images = []
            segment_flow = []
            step = 2
            segment_start = int(s[1]) + (step*half_sample_frame)
            segment_end = int(s[2])+1 - (step*half_sample_frame)

            #if unable to keep all frames in sample inside segment, allow sampling outside of segment
            if segment_start >= segment_end:
                segment_start = int(s[1])#
                segment_end = int(s[2])
            # make sure sampling is not bellow frame 1
            if segment_start <= half_sample_frame*step+1:
                segment_start = half_sample_frame*step+2

            for center_frame in np.linspace(segment_start,segment_end,7,dtype=np.int32)[1:-1]:
                seg_f = []
                seg_i = []
                for no in range(center_frame - (step*half_sample_frame),center_frame + (step*half_sample_frame),step):
                    seg_f.append(flow_filename(int(no/2)))
                    seg_i.append(rgb_filename(no))
                segment_flow.append(seg_f)
                segment_images.append(seg_i)
            return segment_images, segment_flow

        return c3d_sampling()

    def sample_segment(self, s, synchronise=False):
        """ Samples rgb and flow frame windows from a video segment s.
            Sampling temporal windows randomly in video segment.
            s = ["filename", start_frame, end_frame]"""
        def _path_to_dataset(flow):
            if flow:
                left = self.flow_data_path
            else:
                left = self.rgb_data_path
            right = "/frame_"
            numframe = 10
            return left, right, numframe

        #Optical flow stacking
        #returns u,v frames for EPIC Kitchens
        def flow_filename(frameno,num_stack=1):
            left, right,fill_frame = _path_to_dataset(True)
            left_frame = frameno - int((num_stack-1)/2)
            right_frame = frameno + int(num_stack/2)
            filename = []
            for no in range(left_frame, right_frame + 1):
                filename.append(left + str(s[0]) + "/u" + right + str(no).zfill(fill_frame) + ".jpg")
                filename.append(left + str(s[0]) + "/v" + right + str(no).zfill(fill_frame) + ".jpg")
            return filename

        #return RGB frame for EPIC Kitchens
        def rgb_filename(frameno):
            left, right, fill_frame = _path_to_dataset(False)
            filename = left + str(s[0]) + right + str(frameno).zfill(fill_frame) + ".jpg"
            return filename

        def c3d_sampling():
            num_sample_frame = self.temporal_window
            half_sample_frame = int(self.temporal_window/2)
            segment_images = []
            segment_flow = []
            step = 2
            segment_start = int(s[1]) + (step*half_sample_frame)
            segment_end = int(s[2])+1 - (step*half_sample_frame)

            #if unable to keep all frames in sample inside segment, allow sampling outside of segment
            if segment_start >= segment_end:
                segment_start = int(s[1])#
                segment_end = int(s[2])
            # make sure sampling is not bellow frame 1
            if segment_start <= half_sample_frame*step+1:
                segment_start = half_sample_frame*step+2

            if synchronise:
                center_frame_rgb = center_frame_flow = randint(segment_start,segment_end)
            else:
                center_frame_rgb = randint(segment_start, segment_end)
                center_frame_flow = randint(segment_start, segment_end)

            for no in range(center_frame_rgb - (step*half_sample_frame),center_frame_rgb + (step*half_sample_frame),step):
                segment_images.append(rgb_filename(no))
            for no in range(center_frame_flow - (step*half_sample_frame),center_frame_flow + (step*half_sample_frame),step):
                segment_flow.append(flow_filename(int(no/2)))

            return segment_images, segment_flow

        return c3d_sampling()
