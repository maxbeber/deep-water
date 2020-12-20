import numpy as np
from pydensecrf import densecrf
from pydensecrf.utils import unary_from_labels

class CrfLabelRefiner:
    """
    Represents a Conditional Random Fields model.
    Parameters
    ----------
    compat_spat : is a non-dimensional parameter that penalizes small pieces of segmentation
    that are spatially isolated. Larger values means larger pieces of segmentation are allowed.
    compat_col  : is a non-dimensional parameter that penalizes small peices of segmentation
    that are less uniform in color. Larger values means pieces of segmentation with less similar
    image intensity are allowed.
    theta_spat  : represents a location tolerance coeficient.
    theta_col   : representes a color intensity tolerance coeficient.
    num_iter    : number of iterations to run the CRF model inference.
    num_classes : number of classes.
    """
    def __init__(self, compat_spat=10, compat_col=30, theta_spat=20, theta_col=80, num_iter=7, num_classes=2):
        self.compat_spat = compat_spat
        self.compat_col = compat_col
        self.theta_spat = theta_spat
        self.theta_col = theta_col
        self.num_iter = num_iter
        self.num_classes = num_classes

    
    def refine(self, image, mask):
        normalization = densecrf.NORMALIZE_SYMMETRIC
        kernel = densecrf.DIAG_KERNEL
        height, width = image.shape[:2]
        image_unit = (image * 255).astype(np.uint8)
        # Create a CRF object
        d = densecrf.DenseCRF2D(width, height, 2)
        # For the predictions, densecrf needs 'unary potentials' which are labels (water or no water)
        predicted_unary = unary_from_labels(mask.astype('int') + 1, self.num_classes, gt_prob= 0.51)
        # set the unary potentials to CRF object
        d.setUnaryEnergy(predicted_unary)
        # to add the color-independent term, where features are the locations only:
        sxy = (self.theta_spat, self.theta_spat)
        d.addPairwiseGaussian(sxy=sxy, compat=self.compat_spat, kernel=kernel, normalization=normalization)
        # to add the color-dependent term, i.e. 5-dimensional features are (x,y,r,g,b) based on the input image:    
        sxy = (self.theta_col, self.theta_col)
        d.addPairwiseBilateral(sxy=sxy, srgb=(5, 5, 5), rgbim=image_unit, compat=self.compat_col, kernel=kernel, normalization=normalization)
        # Finally, we run inference to obtain the refined predictions:
        inference = d.inference(self.num_iter)
        refined_predictions = np.array(inference).reshape(self.num_classes, height, width)
        # since refined_predictions will be a 2 x width x height array, 
        # each slice respresenting probability of each class (water and no water)
        # therefore we return the argmax over the zeroth dimension to return a mask
        new_mask = np.argmax(refined_predictions, axis=0)

        return new_mask