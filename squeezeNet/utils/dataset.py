import glob
import cv2

from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, dir, channels):
        """
        dir: directory of the dataset containing the example images.
        """
        inFiles = glob.glob(dir + "In_*.bmp")
        paFiles = glob.glob(dir + "Pa_*.bmp")
        rsFiles = glob.glob(dir + "RS_*.bmp")
        
        self.imgFiles = inFiles + paFiles + rsFiles
        self.dim = (200, 200)
        self.labelDict = {"In":0,"Pa":1,"RS":2}
        self.channels = channels
        
        self.sepClasses = [[], [], [], [], [], []]
        for f in self.imgFiles:
            defectLabel = f.split('/')[-1][:2]
            nrLabel = self.labelDict[defectLabel]
            self.sepClasses[nrLabel].append(f)         

        assert len(self.imgFiles) != 0, "No training images were found at the data folder"
        print(f"Creating Dataset with {len(self.imgFiles)} examples\n")

    def __len__(self):
        """Returns amount of examples
        """
        return len(self.imgFiles)

    def __getitem__(self, idx):
        """Returns images and respective labels in Torch Tensor format.
        idx: index of the example (from ids)
        """
        # Read image
        img1, img2, y = self.createExample(idx)
        
        return {"image1":img1,
                "image2":img2,
                "y":y}

    def createExample(self, idx):
        """Retrieves two images (same or different classes, depending on
        the idx value). y indicates if both images belong to the same 
        class.
        idx : index of the example (from ids)
        """
        classes = [0,1,2]
        
        if idx % 2 == 0:
            # same class
            y = 1
            classIdx = random.choice(classes)

            idx1 = random.choice(self.sep_classes[classIdx])
            idx2 = random.choice(self.sep_classes[classIdx])

            img1 = cv2.imread(idx1)
            img2 = cv2.imread(idx2)

            img1 = self.prepare(img1)
            img2 = self.prepare(img2)
        else:
            # different classes
            y = 0
            classIdx1 = random.choice(classes)
            classes.remove(classIdx1)
            classIdx2 = random.choice(classes)

            idx1 = random.choice(self.sep_classes[classIdx1])
            idx2 = random.choice(self.sep_classes[classIdx2])

            img1 = cv2.imread(idx1)
            img2 = cv2.imread(idx2)

            img1 = self.prepare(img1)
            img2 = self.prepare(img2)
        
        return img1, img2, y

    def prepare(self, img):
        """Input image undergoes the pre-processing steps for 
        the network. 
        img : image
        """
        img = cv2.resize(img, self.dim, cv2.INTER_AREA)
        if self.channels == 1:     
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img).type(torch.FloatTensor)

        return img