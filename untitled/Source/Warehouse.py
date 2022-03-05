import csv
import json
import cv2
import numpy as np
from joblib import dump, load
import operator

import Directory
from Video import Video
import FeaturesExtraction as FE
import time
import matplotlib.pyplot as plt

class Warehouse:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load(model_path)
        self.path_list = []
        self.report = {}
        self.count = {}
        self.is_Reported = False

    def append(self, path):
        self.path_list.append(path)

    def predict(self,labels_list, tag):
        features_vector = features_vector = FE.Extraction(tag)
        # predict_proba = np.max(self.model.predict_proba(np.array([features_vector], np.float32)))
        # if (predict_proba < 0.5): return
        predict = self.model.predict(np.array([features_vector], np.float32))[0]
        if (predict != "No Label"):
            labels_list.append(predict)

    def save_report(self):
        with open('report_clips.json', 'w') as json_file:
            json.dump(self.report, json_file)
        names = [*self.count]
        quantity = [*self.count.values()]
        with open('report_videos.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for idx, name in enumerate(names):
                csv_writer.writerow((name, quantity[idx]))

        self.is_Reported = True

    def analysis_all(self):
        for path in self.path_list:
            sources = Directory.GetDirectories(path)
            for source in sources:
                videos = Directory.GetFiles(source)
                for video in videos:
                    v = Video(video)
                    labels_list = []
                    start = time.time()
                    idx_frame = -1
                    while v.next_frame() is not None:
                        idx_frame +=1
                        if idx_frame %4 != 0: continue

                        [tag1, tag2] = v.get_tag()
                        self.predict(labels_list, tag1)
                        self.predict(labels_list, tag2)

                    v.release()
                    points = {}
                    for idx, tag in enumerate(labels_list):
                        if tag in points:
                            points[tag] += idx
                        else:
                            points[tag] = idx
                    end = time.time()
                    print(end - start)
                    print(video)
                    if (len(points) <= 0): continue
                    points = np.array(sorted(points.items(), key=operator.itemgetter(0)))
                    points = points[:, 0]
                    for name in points:
                        if name not in self.report:
                            self.report[name] = []
                            self.count[name] = 0
                        self.report[name].append(video)
                        self.count[name] += 1
        self.save_report()