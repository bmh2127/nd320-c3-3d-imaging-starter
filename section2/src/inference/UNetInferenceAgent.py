"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        raise NotImplementedError

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>

        # get the number of slices
        num_slices = volume.shape[0]

        # create an empty numpy array of shape [num_slices, 1, patch_size, patch_size]
        batch = np.empty((num_slices, 1, self.patch_size, self.patch_size), dtype=np.float32)

        # loop over the slices in the volume
        for i in range(num_slices):
            # crop the patch of size [patch_size, patch_size] from the current slice
            patch = volume[i, :, :self.patch_size, :self.patch_size]

            # resize the patch to [1, patch_size, patch_size]
            patch = med_reshape(patch, [1, self.patch_size, self.patch_size])

            # convert numpy array to tensor
            patch = torch.from_numpy(patch).float()

            # push tensor to device
            patch = patch.to(self.device)

            # append tensor to batch
            batch[i] = patch

        # convert batch to tensor
        batch = torch.from_numpy(batch).float()

        # run inference on the batch
        predictions = self.model(batch)

        # convert tensor to numpy array
        predictions = predictions.cpu().detach().numpy()

        # take the argmax over the channel dimension
        predictions = np.argmax(predictions, axis=1)

        # return predictions
        return predictions
