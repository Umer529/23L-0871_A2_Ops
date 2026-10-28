"""Task 2.2 / 2.3 - model definitions.

Both models take float32 images in [0, 255] and do their own scaling first,
so training, evaluation and the Flask app all feed the same input.
"""
from tensorflow import keras
from tensorflow.keras import layers

# first layer of the last block for each backbone ("unfreeze only the last block")
LAST_BLOCK_START = {
    "mobilenetv2": "block_16_expand",
    "efficientnetb0": "block7a_expand_conv",
}


def build_baseline_cnn(img_size=224, dropout=0.4):
    def conv_block(x, filters, name):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.Activation("relu", name=f"{name}_relu")(x)
        return layers.MaxPooling2D(2, name=f"{name}_pool")(x)

    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="rescale_0_1")(inputs)
    x = conv_block(x, 32, "block1")
    x = conv_block(x, 64, "block2")
    x = conv_block(x, 128, "block3")
    x = conv_block(x, 128, "block4")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer_model(backbone="mobilenetv2", img_size=224, dropout=0.3):
    shape = (img_size, img_size, 3)
    if backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape)
        # same maths as mobilenet_v2.preprocess_input ([0,255] -> [-1,1]) but as a
        # serialisable layer, so the saved model carries its own preprocessing
        preprocess = layers.Rescaling(1 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")
    elif backbone == "efficientnetb0":
        base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=shape)
        # EfficientNet normalises internally and expects raw [0, 255] pixels
        preprocess = layers.Identity(name="efficientnet_preprocess")
    else:
        raise ValueError(f"unknown backbone {backbone}")
    base.trainable = False

    inputs = keras.Input(shape, name="image")
    x = preprocess(inputs)
    # training=False keeps BatchNorm in inference mode even after unfreezing,
    # so a small dataset does not wreck the pretrained running statistics
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_transfer"), base


def unfreeze_last(base, backbone="mobilenetv2", n_layers=None):
    """Unfreeze the last block (default) or the last n_layers of the backbone.
    BatchNorm layers stay frozen either way."""
    base.trainable = True
    if n_layers is None:
        names = [l.name for l in base.layers]
        start = names.index(LAST_BLOCK_START[backbone])
    else:
        start = len(base.layers) - n_layers
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
    return [l.name for l in base.layers if l.trainable and l.weights]
