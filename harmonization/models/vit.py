"""
Module related to the Harmonized ResNet50 model
"""

import tensorflow as tf
from vit_keras import vit

HARMONIZED_VITB16_WEIGHTS = ('https://storage.googleapis.com/serrelab/prj_harmonization/'
                             'models/vit-b16_harmonized.h5')


def load_ViT_B16(include_top=True):
    """
    Loads the Harmonized ViT-B16.

    Returns
    -------
    model
        Harmonized ViT-B16 keras model.
    """
    weights_path = tf.keras.utils.get_file("vit-b16_harmonized.h5", HARMONIZED_VITB16_WEIGHTS,
                                            cache_subdir="/proj/rep-learning-robotics/users/x_nonra/Harmonization/harmonized_models")

    model = vit.vit_b16(
        image_size=224,
        activation='linear',
        pretrained=False,
        include_top=include_top,
        pretrained_top=False
    )
    model.load_weights(weights_path, skip_mismatch=True, by_name=True)

    return model

def load_ViT_B16_orig(include_top=True):
    """
    Loads the Non-Harmonized ViT-B16.

    Returns
    -------
    model
        ViT-B16 keras model.
    """
    model = vit.vit_b16(
        image_size=224,
        activation='linear',
        pretrained=True,
        include_top=include_top,
        pretrained_top=True if include_top else False
    )

    return model
