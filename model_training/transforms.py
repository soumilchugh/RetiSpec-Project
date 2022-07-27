import yacs.config
import albumentations as A

class Transform():

    def __init__(self, config):
        self.config = config

    def create_transform(self):
        self.transform_method = A.Compose( [            
            A.ToFloat(max_value=65535.0),
                A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),        

            ],
            additional_targets={'image0': 'image', 'image1': 'image'}
        )
        return self.transform_method
