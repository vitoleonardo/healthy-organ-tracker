from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def conv_down(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    out = Activation("relu")(x)

    return out

def encoder_block(input, num_filters):
    enc = conv_down(input, num_filters)
    pool = MaxPool2D((2, 2))(enc)
    return enc, pool

def decoder_block(input, skip_features, num_filters):
    up_sample = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    skip_concat = Concatenate()([up_sample, skip_features])
    out = conv_down(skip_concat, num_filters)
    return out

def build_unet(input_shape, classes):
    inputs = Input(input_shape)

    # Part 1: Downsampling
    skip_1, enc_1 = encoder_block(inputs, 64)
    skip_2, enc_2 = encoder_block(enc_1, 128)
    skip_3, enc_3 = encoder_block(enc_2, 256)
    skip_4, enc_4 = encoder_block(enc_3, 512)

    bottleneck = conv_down(enc_4, 1024)

    # Part 2: Upsamling
    dec_1 = decoder_block(bottleneck, skip_4, 512)
    dec_2 = decoder_block(dec_1, skip_3, 256)
    dec_3 = decoder_block(dec_2, skip_2, 128)
    dec_4 = decoder_block(dec_3, skip_1, 64)

    # Last conf containing number of classes
    outputs = Conv2D(classes, (1,1), padding="same", activation="sigmoid")(dec_4)

    model = Model(inputs, outputs, name="U-Net")
    return model
