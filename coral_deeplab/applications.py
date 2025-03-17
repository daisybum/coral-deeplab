import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2DTranspose, Cropping2D, Softmax
from coral_deeplab._blocks import deeplab_aspp_module, deeplabv3_decoder, deeplabv3plus_decoder
from coral_deeplab._encoders import mobilenetv2

def CoralDeepLabV3(input_shape=(513, 513, 3), alpha=1.0, n_classes=6):
    inputs = Input(shape=input_shape)
    aspp_in = mobilenetv2(inputs, alpha)
    aspp_out = deeplab_aspp_module(aspp_in)
    logits = deeplabv3_decoder(aspp_out, n_classes)  # [33, 33, 6]
    return tf.keras.Model(inputs=inputs, outputs=logits, name="CoralDeepLabV3")

def CoralDeepLabV3Plus(input_shape=(513, 513, 3), alpha=1.0, n_classes=6):
    # CoralDeepLabV3를 인코더로 사용
    encoder = CoralDeepLabV3(input_shape, alpha)
    encoder_last = encoder.get_layer("concat_projection/relu")  # [33, 33, 256]
    encoder_skip = encoder.get_layer("expanded_conv_3/expand/relu")  # [129, 129, 24]
    
    # DeepLabV3+ 디코더 적용
    logits = deeplabv3plus_decoder(encoder_last.output, encoder_skip.output, n_classes)  # [129, 129, 6]
    
    # 전치 합성곱으로 업샘플링: 129x129 -> 516x516
    logits = Conv2DTranspose(
        filters=n_classes, 
        kernel_size=4, 
        strides=4, 
        padding='same', 
        use_bias=False
    )(logits)  # 출력: [516, 516, 6]
    
    # 크롭하여 정확히 513x513으로 맞춤
    logits = Cropping2D(cropping=((1, 2), (1, 2)))(logits)  # [513, 513, 6]
    
    # Softmax 적용으로 클래스 범위 보장
    outputs = Softmax()(logits)
    
    model = tf.keras.Model(inputs=encoder.inputs, outputs=logits, name="CoralDeepLabV3Plus")
    return model