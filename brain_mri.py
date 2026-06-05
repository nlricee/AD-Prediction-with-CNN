import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time

## --- Constants --- ##

# MRI images should be classified into 4 classes:
# NonDemented: No dementia.
# Very Mild Demented: Early signs of dementia, very mild symptoms.
# Mild Demented: Clear signs of dementia, but still mild.
# Moderate Demented: More pronounced symptoms of dementia, moderate severity.
NUM_CLASSES = 4
LABEL_MAP = {
    'NonDemented':      0,
    'VeryMildDemented': 1,
    'MildDemented':     2,
    'ModerateDemented': 3
}
# Extracts class names from the keys in LABEL_MAP
# ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
CLASS_NAMES = list(LABEL_MAP.keys())


IMG_SIZE = 64           # 64x64 image (better resolution)
LIMIT_PER_CLASS = 40000  # Input images per class
EPOCHS = 30             # Number of complete pass through training dataset
                        # Greater passes = better recognition
                        # Note: risk of overfitting
DATA_DIR = 'data'       # folder name of dataset



'''
---------DATASET---------
Inherits Pytorch's Dataset class and overrides
three methods that are used when calling DataLoader
'''
class BrainMRIDataset(Dataset):
    def __init__(self, images, labels):
        """
        This constructor utilizes tensor from PyTorch
        to convert numpy array to torch tensor for GPU support
        in order to run the large dataset faster. In this context,
        numpy array images and labels are converted to torch tensor.
        Essentially, it stores all the images and labels.
        """
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """
        Override this method to return number of images
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Override this method to return one sample
        image and its corresponding label
        """
        return self.images[idx], self.labels[idx]

'''
---------LOADING DATA--------
Filters out images that are not in the right filetype 
and grayscale coloring.
Stores images and their corresponding labels into x and y
arrays respectively.
'''
def load_data(data_dir, limit_per_class, img_size):
    """
    This function ensures all data is loaded correctly
    and consistently. While the dataset should only contain
    grayscale JPEG files, the function loops through each
    subfolder to filter out wrong filetypes and image channels
    before adding the images and their corresponding labels
    to two separate arrays x, y that can be used for training
    and testing.
    Inputs: data_dir - directory name with image dataset
            limit_per_class - number of images from each subfolder
                              or class name
            img_size - size of each image
    Outputs: np.array(x) - numpy array (3D) containing
                           a number of limit_per_class images
                           that are 64x64
            np.array(y) - numpy array (1D) containing
                          the corresponding labels (integer)
                          of each image
    """

    # Initialize x, y to empty lists
    # x - image arrays
    # y - corresponding labels
    x, y = [], []

    # Loops over the four classes labeled in LABEL_MAP
    # based on the class name and its label
    # (i.e.: iteration 1 loops over the images for
    # class_name = 'NonDemented' and label = 0)
    for class_name, label in LABEL_MAP.items():

        # Builds directory path from data_dir to the class name
        # since the 'data' folder is split into subfolders
        # with names directly matching the class name
        # (i.e.: the first subfolder name is NonDemented)
        class_dir = os.path.join(data_dir, class_name)

        # List of all filenames in the folder class_dir
        # from the beginning up to the specified image limit
        # per class (where limit_per_class < 11000)
        images = os.listdir(class_dir)[:limit_per_class]

        # Loops through the list of filenames (abb. fname)
        for fname in images:

            # All files in the 'data' should be JPEG, but in case
            # the wrong filetype is downloaded, it will be filtered
            # out.
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # Opens the filename from the subfolder using Python
            # Imaging Library (PIL) and converts to grayscale ('L').
            # All files in the 'data' should also be grayscale, but
            # in case the image has more channels (i.e. 3 channels for RGB),
            # this ensures that it will be converted to grayscale.

            img = Image.open(os.path.join(class_dir, fname)).convert('L')

            # Resizes image to 64x64 (better for training)
            img = img.resize((img_size, img_size))

            # Converts PIl image to numpy array to append to x
            x.append(np.array(img))
            # Appends corresponding label to y
            y.append(label)

    return np.array(x), np.array(y)


# Implements the load_data() function above using the DATA_DIR
# 'data' folder, the LIMIT_PER_CLASS constant defined above,
# and the IMG_SIZE constant defined above
x_all, y_all = load_data(DATA_DIR, LIMIT_PER_CLASS, IMG_SIZE)

# Reshapes the numpy array (3D) of images into 4D to
# include the number of channels, which is only 1,
# because grayscale images only have one channel.
# It also converts pixel values (0-255) to float32,
# which is the expected type for PyTorch image data.
# Dividing by 255.0 normalizes the values to 0-1, which
# improves training speed and stability (large input values
# requires larger activations).
x_all = x_all.reshape(len(x_all), 1, IMG_SIZE, IMG_SIZE).astype('float32') / 255.0

## --- Splitting Training and Testing Sets --- ##

# 20% of the image set is used for testing, and 80% is for training
# Arbitrary constant 42 is used as a random seed
# The stratify parameter helps prevent bias and imbalance by ensuring that
# splitting the dataset will maintain the same class proportions
# (i.e. Instead of randomly getting 10% of ModerateDemented and 90% of
# NonDemented, the training and testing sets will have approximately
# equal proportions of the four classes).
x_train, x_test, y_train, y_test = train_test_split(
    x_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Dataloaders are created for training and testing.
# 32 images processed in one forward pass simultaneously
# for increased speed.
# Training dataset is shuffled at each epoch (successful run)
# to prevent sequence bias (model would recognize a sequential
# pattern instead of learning the features.
#
# train_loader returns DataLoader object that contains 80% of the
# total images used (total_images = LIMIT_PER_CLASS * NUM_CLASSES),
# and with 32 images per batch (batch_size=32), the total batches
# will be total_images / batch_size, which is the len(train_loader).
train_loader = DataLoader(BrainMRIDataset(x_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(BrainMRIDataset(x_test,  y_test),  batch_size=32)

'''
---------MODEL--------
Compiles all layers of the model
'''

# Model is an ordered container that sequentially holds
# the layers specified below.
model = nn.Sequential(
    # First convolutional layer
    # Uses 16 3x3 kernels to detect lower-level features (edges and gradients)
    # Input: 64x64
    nn.Conv2d(1, 16, kernel_size=3),    # 64-3+1 = 62
    nn.ReLU(),
    nn.MaxPool2d(2, 2),                      # 62/2 = 31x31

    # Second convolutional layer
    # Uses 32 3x3 kernels to detect higher-level features (shapes and textures)
    nn.Conv2d(16, 32, kernel_size=3),   # 31-3+1 = 29
    nn.ReLU(),
    nn.MaxPool2d(2, 2),                      # 29/2 = 14x14

    # Classification head
    nn.Flatten(),
    nn.Linear(32 * 14 * 14, 256), # out_channels * last layer dims
    nn.ReLU(),
    nn.Linear(256, NUM_CLASSES)
)

'''
---------TRAINING---------
Trains model using forward and back propagation,
and calculates and displays loss/error per epoch
'''

# Creates optimizer that updates the weights/parameters after each
# backward pass to minimize loss function and improve accuracy
# The Pytorch Adam optimizer updates each weight individually
# based on history of gradients, adapting the learning rate
# for each weight
optimizer = optim.Adam(model.parameters(), lr=0.001)

# The loss function is a callable object that
# measures the error in predictions.
# It is based on the computing the cross entropy
# loss, which applies the logarithm of
# Softmax (function that converts a vector of raw numerical scores
# into a probability distribution where each value is between 0 and 1
# and all sums to 1), picks log probability value at the index
# corresponding to the actual class label, and finally negates
# that value because the log of a number between 0-1 is negative.
# This positive decimal value is the loss, which will be calculated
# when calling this loss function.
loss_fn = nn.CrossEntropyLoss()

# Initializes a list of losses in an epoch
# that is used to compute a training loss curve later
epoch_losses = []

# Initialize timer
start_time = time.time()

# Loops through each epoch run
for epoch in range(EPOCHS):
    # Use Pytorch train() function
    # Default return to True to set training mode on
        # Note: Only maters for certain modules like
        # Dropout and BatchNorm, which behaviors differently
        # in training and evaluation. Good practice to include.
    model.train()

    # Initialize error to 0
    total_loss = 0

    # Recall train_loader above loads and processes 32 images at a time
    # for a total number of batches len(train_loader).
    # This loop iterates the DataLoader object returned by train_loader
    # to loop through each batch of 32 images and their corresponding
    # 32 class labels.
    for images, labels in train_loader:
        # The Pytorch zero_grad() function resets the
        # gradients of the optimizer for each batch.
        # Prevents adding new gradients to the ones from
        # the previous batch.
        optimizer.zero_grad()

        # FORWARD PROPAGATION
        #
        # Using the CNN model container created above, 32 images
        # will be forward-passed through each layer in order, and
        # the result is stored in output.
        output = model(images)

        # Using the loss function, it calculates a single error value
        # using the output computed above and 32 images and labels.
        # Value returned as Tensor object.
        loss = loss_fn(output, labels)

        # BACKWARD PROPAGATION
        #
        # Pytorch's backward() function traverses through
        # all the operations performed in model during
        # forward propagation and computes the gradient for each
        # loss with respect to the weights (kernels and biases).
        loss.backward()

        # Uses calculated gradients to update each weight
        # Basic gradient: weight = weight - (learning_rate * gradient)
        # Gradient with Adam optimizer:
        #   First Moment (momentum): calculates EMA (exponentially moving avg)
        #   of past gradients to allow algorithm to converge faster by
        #   reducing oscillations (smooth out noisy updated gradients)
        #       Case: Weight is moving in a consistent direction
        #           Ex: The sign of the gradient determines its direction.
        #               Same sign gradients = consistent direction
        #
        #
        #   Second Moment (adapative learning rate): calculates RMSprop
        #   (root-mean-square propagation) of past squared gradients to
        #   overcome problem of diminishing learning rates
        #       Case: Weight is oscillating
        #           Ex: Diff sign gradients = oscillating
        #
        #   Let w = weight, lr = learning_rate, t = time
        #   update = (lr / sqrt(second_moment)) * first_moment
        #   w(t+1) = w(t) - update
        optimizer.step()

        # Converts Tensor object to float to get
        # the loss value and adds to the total
        # error
        total_loss += loss.item()


    # Appends losses per epoch
    epoch_losses.append(total_loss / len(train_loader))
    # For each run, the average error over the total batches is displayed
    print(f"{epoch+1}/{EPOCHS}, error={total_loss/len(train_loader):.4f}")


end_time = time.time()              # End timer
total_time = end_time - start_time  # Calculate total runtime

print(f"Total training time: {total_time:.2f}s")
print(f"Average time per epoch: {total_time / EPOCHS:.2f}s")

'''
---------EVALUATION--------
Evaluates how many correct predictions were made
by the model
'''

# Sets model to evaluation mode
# Similar to train(), eval() matters to certain modules
# such as dropout and batch normalization, so for this
# particular model, it doesn't have an effect, but its good
# practice for future updates
model.eval()

# Initialize a counter of how many predictions were correct
correct = 0

# Initialize list of predictions
all_predictions = []

# Initialize list of correctly classified class labels
all_true_labels = []


with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        for pred, true in zip(preds, labels):
            if pred == true:
                correct += 1
            all_predictions.append(pred.item())
            all_true_labels.append(true.item())
            print(f"pred: {CLASS_NAMES[pred]:<20} true: {CLASS_NAMES[true]}")

print(f"\nAccuracy: {correct}/{len(x_test)} = {correct/len(x_test)*100:.1f}%")


'''
---------CONFUSION MATRIX--------
Create and displays a confusion matrix between
the predicted class label and actual class label
'''
cm = confusion_matrix(all_true_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap='Blues')
plt.title("Alzheimer's Disease Brain MRI Classification Confusion Matrix")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("ConfusionMatrix.png")
plt.show()




'''
---------TRAINING LOSS CURVE--------
Create and displays a training loss curve of how
the error decreased across each of the 30 epochs
'''
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), epoch_losses, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.xticks(range(1, EPOCHS + 1))
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()
