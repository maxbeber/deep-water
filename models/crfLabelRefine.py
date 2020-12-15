import numpy as np
from pydensecrf import densecrf
from pydensecrf.utils import unary_from_labels

class CrfLabelRefine:
    """
    Represents a Conditional Random Fields model.
    Parameters
    ----------
    compat_spat : folder where the images are located
    compat_col  : size of the batch
    theta_spat  : size of each image
    theta_col   : size of each image
    num_iter    : number of iterations to run the CRF model inference
    num_classes : number of classes
    """
    def __init__(self, compat_spat=10, compat_col=30, theta_spat=20, theta_col=80, num_iter=10, num_classes=2):
        self.compat_spat = compat_spat
        self.compat_col = compat_col
        self.theta_spat = theta_spat
        self.theta_col = theta_col
        self.num_iter = num_iter
        self.num_classes = num_classes

    
    def refine(self, image, mask):
        height, width = image.shape[:2]
    
        # Create a CRF object
        d = densecrf.DenseCRF2D(width, height, 2)

        # For the predictions, densecrf needs 'unary potentials' which are labels (water or no water)
        predicted_unary = unary_from_labels(mask.astype('int') + 1, self.num_classes, gt_prob= 0.51)
    
        # set the unary potentials to CRF object
        d.setUnaryEnergy(predicted_unary)

        # to add the color-independent term, where features are the locations only:
        d.addPairwiseGaussian(\
            sxy=(self.theta_spat, self.theta_spat),
            compat=self.compat_spat,
            kernel=densecrf.DIAG_KERNEL,
            normalization=densecrf.NORMALIZE_SYMMETRIC)

        input_image_uint = (image * 255).astype(np.uint8) #enfore unsigned 8-bit
        # to add the color-dependent term, i.e. 5-dimensional features are (x,y,r,g,b) based on the input image:    
        d.addPairwiseBilateral(\
            sxy=(self.theta_col, self.theta_col),
            srgb=(5, 5, 5),
            rgbim=input_image_uint,
            compat=self.compat_col,
            kernel=densecrf.DIAG_KERNEL,
            normalization=densecrf.NORMALIZE_SYMMETRIC)

        # Finally, we run inference to obtain the refined predictions:
        refined_predictions = np.array(d.inference(self.num_iter)).reshape(self.num_classes, height, width)
    
        # since refined_predictions will be a 2 x width x height array, 
        # each slice respresenting probability of each class (water and no water)
        # therefore we return the argmax over the zeroth dimension to return a mask
        predictions = np.argmax(refined_predictions, axis=0)

        return predictions