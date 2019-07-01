from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet201, DenseNet169, DenseNet121
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetMobile, NASNetLarge, preprocess_input
from keras.optimizers import Adam, RMSprop
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.applications.mobilenet_v2 import MobileNetV2

inception_resnet_v2_path = '/root/pretrain_model/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
inception_v3_path = '/root/pretrain_model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
resnet50_path = '/root/pretrain_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg16_path = '/root/pretrain_model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
densenet121_path = '/root/pretrain_model/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
densenet201_path = '/root/pretrain_model/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'
nasnet_large_path = '/root/pretrain_model/nasnet_large_no_top.h5'
vgg19_path = '/root/pretrain_model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
densenet169_path = '/root/pretrain_model/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
mobilenet_path = '/root/pretrain_model/mobilenet_1_0_224_tf_no_top.h5'
nasnet_path = '/root/pretrain_model/nasnet_mobile_no_top.h5'
xception_path = '/root/pretrain_model/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

def nasnet_large(weights=nasnet_large_path):
    weights = weights
    pretrained_model = NASNetLarge(weights=weights, include_top=False)
    return (pretrained_model)

def nasnet_mobile(weights=nasnet_path):
    weights = weights
    pretrained_model = NASNetMobile(weights=weights, include_top=False)
    return (pretrained_model)

def densenet_169(weights = densenet169_path):
    weights = weights
    pretrained_model = DenseNet169(weights=weights, include_top=False)
    return (pretrained_model)

def mobilenet(weights = mobilenet_path):
    weights = weights
    pretrained_model = MobileNetV2(weights=weights, include_top=False)
    return (pretrained_model)

def inception(weights = inception_v3_path):
    weights = weights
    pretrained_model = InceptionV3(weights=weights, include_top=False)
    return (pretrained_model)

def resnet50(weights = resnet50_path):
    weights = weights
    pretrained_model = ResNet50(weights=weights, include_top=False)
    return (pretrained_model)

def inception_resnetv2(weights = inception_resnet_v2_path):
    weights = weights
    pretrained_model = InceptionResNetV2(weights=weights, include_top=False)
    return (pretrained_model)

def densenet_201(weights = densenet201_path):
    weights = weights
    pretrained_model = Densenet201(weights=weights, include_top=False)
    return (pretrained_model)

def densenet_121(weights = densenet121_path):
    weights = weights
    pretrained_model = DenseNet121(weights=weights, include_top=False)
    return (pretrained_model)

def vgg16(weights = vgg16_path):
    weights = weights
    pretrained_model = VGG16(weights=weights, include_top=False)
    return (pretrained_model)

def vgg19(weights = vgg19_path):
    weights = weights
    pretrained_model = VGG19(weights=weights, include_top=False)
    return (pretrained_model)