import Directory
import numpy as np
import cv2
import FeaturesExtraction as FE

class Dataset:
    def __init__(self, root):
        self.root = root
        self.isLoaded = False

    def info(self):
        if (self.isLoaded == False):
            self.load()

        return 'Total: ' + str(self.total) \
               + '\nMin: ' + str(self.min)\
               + '\nAve: ' + str(self.mean)\
               + '\nMax: ' + str(self.max)

    def extract_labels(self, path = 'labels.txt'):
        f= open(path,"w+")
        self.labels = []
        players = Directory.GetDirectories(self.root)
        for idx, player in enumerate(players):
            player_name = player.split('/')[-2]
            f.write(player_name + "\n")
            self.labels.append(player_name)
        f.close()

    def load(self):
        players = Directory.GetDirectories(self.root)
        players_tag = []
        self.dataset = []
        self.target = []

        for idx, player in enumerate(players):
            tags = Directory.GetFiles(player)
            player_name = player.split('/')[-2]
            for tag in tags:
                img = cv2.imread(tag)
                features_vector = FE.Extraction(img)

                self.dataset.append(features_vector)
                self.target.append(player_name)
            players_tag.append(len(tags))

        self.dataset = np.array(self.dataset)
        self.target = np.array(self.target)

        self.total = np.sum(players_tag)
        self.min = np.min(players_tag)
        self.mean = np.mean(players_tag)
        self.max = np.max(players_tag)
        self.isLoaded = True