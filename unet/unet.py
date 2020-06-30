import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Dropout, Cropping2D, UpSampling2D
import keras.optimizers
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras import regularizers
import numpy as np
import os
from tqdm import tqdm
from time import time
from . import metrics
from .callbacks import CyclicLR
import pdb


class Unet2D:
    def __init__(self, snapshot=None, n_channels=1, n_classes=2, n_levels=4,
                 n_features=64, batch_norm=True, relu_alpha=0.1,decay=0.0,
                 bn_skip = 0,
                 upsample=False, k_init="he_normal", name="U-Net"):

        self.concat_blobs = []

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.n_features = n_features
        self.batch_norm = batch_norm
        self.relu_alpha = relu_alpha
        self.k_init = k_init
        self.upsample = upsample
        self.name = name
        self.decay = decay
        self.bn_skip = bn_skip
        self.metrics = [metrics.recall,
                        metrics.precision,
                        metrics.f1,
                        metrics.iou,
                        metrics.mcor,
                        #tf.keras.metrics.MeanIoU(num_classes=2, name="my_MeanIoU")
        ]

        self.opt = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=self.decay, amsgrad=False)
        self.trainModel, self.padding = self._createModel(True)
        self.testModel, _ = self._createModel(False)


        if snapshot is not None:
            self.trainModel.load_weights(snapshot)
            self.testModel.load_weights(snapshot)

    def _weighted_categorical_crossentropy(self, y_true, y_pred, weights):
        return tf.compat.v1.losses.softmax_cross_entropy(y_true, y_pred, weights=weights, reduction=tf.compat.v1.losses.Reduction.MEAN)

    def _createModel(self, training):
        
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 1
        #config.gpu_options.visible_device_list = "0"
        #K.set_session(tf.Session(config=config))

        data = keras.layers.Input(shape=(None, None, self.n_channels), name="data")

        concat_blobs = []

        if training:
            labels = keras.layers.Input(
                shape=(None, None, self.n_classes), name="labels")
            weights = keras.layers.Input(shape=(None, None), name="weights")

        # Modules of the analysis path consist of two convolutions and max pooling
        for l in range(self.n_levels):
            t = Conv2D(2**l * self.n_features, 3, padding="valid", kernel_initializer=self.k_init,
                       name="conv_d{}a-b".format(l))(data if l == 0 else t)
            if (self.batch_norm) & (l>self.bn_skip):
                t = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(t)
            t = LeakyReLU(alpha=self.relu_alpha)(t)
            t = Conv2D(2**l * self.n_features, 3, padding="valid",
                       kernel_initializer=self.k_init, name="conv_d{}b-c".format(l))(t)
            if (self.batch_norm) & (l>self.bn_skip):                    
                t = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(t)
            t = LeakyReLU(alpha=self.relu_alpha)(t)
            # if l >= 2:
            #    t = Dropout(rate=0.5)(t)
            concat_blobs.append(t)
            t = keras.layers.MaxPooling2D(pool_size=(2, 2))(concat_blobs[-1])

        # Deepest layer has two convolutions only
        t = Conv2D(2**self.n_levels * self.n_features, 3, padding="valid",
                   kernel_initializer=self.k_init, name="conv_d{}a-b".format(self.n_levels))(t)
        if self.batch_norm:
            t = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(t)
        t = LeakyReLU(alpha=self.relu_alpha)(t)
        t = Conv2D(2**self.n_levels * self.n_features, 3, padding="valid",
                   kernel_initializer=self.k_init, name="conv_d{}b-c".format(self.n_levels))(t)
        if self.batch_norm:
            t = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(t)
        t = LeakyReLU(alpha=self.relu_alpha)(t)
        pad = 8

        # Modules in the synthesis path consist of up-convolution,
        # concatenation and two convolutions
        for l in range(self.n_levels - 1, -1, -1):
            name = "upconv_{}{}{}_u{}a".format(
                *(("d", l+1, "c", l) if l == self.n_levels - 1 else ("u", l+1, "d", l)))
            if self.upsample:
                t = UpSampling2D(size=(2, 2), name=name)(t)
            else:
                t = Conv2DTranspose(2**np.max((l, 1)) * self.n_features, (2, 2), strides=2,
                                    padding='valid', kernel_initializer=self.k_init, name=name)(t)
                if (self.batch_norm) & (l>self.bn_skip):
                    t = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(t)
                t = LeakyReLU(alpha=self.relu_alpha)(t)
            t = Concatenate()(
                [Cropping2D(cropping=int(pad / 2))(concat_blobs[l]), t])

            t = Conv2D(2**np.max((l, 1)) * self.n_features, 3, padding="valid",
                       kernel_initializer=self.k_init, name="conv_u{}b-c".format(l))(t)
            if (self.batch_norm) & (l>self.bn_skip):
                t = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(t)
            t = LeakyReLU(alpha=self.relu_alpha)(t)
            t = Conv2D(2**np.max((l, 1)) * self.n_features, 3, padding="valid",
                       kernel_initializer=self.k_init, name="conv_u{}c-d".format(l))(t)
            if (self.batch_norm) & (l>self.bn_skip):
                t = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(t)
            t = LeakyReLU(alpha=self.relu_alpha)(t)
            pad = 2 * (pad + 8)

        pad /= 2

        score = Conv2D(self.n_classes, 1,
                       kernel_initializer=self.k_init, name="conv_u0d-score")(t)
        softmax_score = keras.layers.Softmax()(score)

        if training:
            model = keras.Model(
                inputs=[data, labels, weights], outputs=softmax_score)
            model.add_loss(self._weighted_categorical_crossentropy(
                labels, score, weights))
            model.compile(optimizer=self.opt, loss=None)#, metrics=[tf.keras.metrics.Recall()])
            #for m in self.metrics:
            #    model.metrics_tensors.append(m(labels, softmax_score))
            #    model.metrics_names.append(m.__name__)
            #    #model.metrics_names.append(m.name)
            
        else:
            model = keras.Model(inputs=data, outputs=softmax_score)

        return model, int(pad)


  
        
    def fit_one_cycle(self, train_generator, final_epoch, max_lr, 
                      initial_epoch=0,
                      validation_generator=None, 
                      cycle_momentum = True,
                      verbose = True, 
                      snapshot_interval=1,
                      snapshot_dir= 'checkpoints', 
                      snapshot_prefix=None,
                      log_dir = None, 
                      step_muliplier=1):
        """
        Train model using a version of Leslie Smith's 1cycle policy.
        This method can be used with any optimizer. Thus,
        cyclical momentum is not currently implemented.
        Args:
            max_lr (float): (maximum) learning rate.  
                       It is recommended that you estimate lr yourself by 
                       running lr_finder (and lr_plot) and visually inspect plot
                       for dramatic loss drop.
            epochs (int): Number of epochs.  Number of epochs
            checkpoint_folder (string): Folder path in which to save the model weights 
                                        for each epoch.
                                        File name will be of the form: 
                                        weights-{epoch:02d}-{val_loss:.2f}.hdf5
            cycle_momentum (bool):    If True and optimizer is Adam, Nadam, or Adamax, momentum of 
                                      optimzer will be cycled between 0.95 and 0.85 as described in 
                                      https://arxiv.org/abs/1803.09820.
                                      Only takes effect if Adam, Nadam, or Adamax optimizer is used.
            callbacks (list): list of Callback instances to employ during training
            verbose (bool):  verbose mode
        """
        
        callbacks = []
        if log_dir is not None:
            tb_cb = TensorBoard(log_dir= log_dir + "/{}-{}".format(self.name, time()))
            callbacks.append(tb_cb)
        if snapshot_prefix is not None:
            if not os.path.isdir(snapshot_dir):
                os.makedirs(snapshot_dir)
            c_path = os.path.join(
                snapshot_dir, (snapshot_prefix if snapshot_prefix is not None else self.name))
            callbacks.append(ModelCheckpoint(
                c_path + ".{epoch:04d}.h5", mode='auto', period=snapshot_interval))
        
        if cycle_momentum:
            max_momentum = 0.95
            min_momentum = 0.85
        else:
            max_momentum = None
            min_momentum = None
            
        num_samples = len(train_generator)*train_generator.batch_size #U.nsamples_from_data(self.train_data)
        steps_per_epoch = np.ceil(num_samples/train_generator.batch_size)*step_muliplier
        epochs = final_epoch-initial_epoch   
        clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
        clr = CyclicLR(base_lr=max_lr/10, max_lr=max_lr,
                       step_size=np.ceil((steps_per_epoch*epochs)/2), # Authors suggest setting step_size = (2-8) x (training iterations in epoch=63)
                       scale_fn=clr_fn,
                       reduce_on_plateau=0,
                       max_momentum=max_momentum,
                       min_momentum=min_momentum,
                       verbose=verbose)
        callbacks.append(clr)
        
        hist = self.trainModel.fit_generator(train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=final_epoch,
                                      initial_epoch = initial_epoch,
                                      validation_data=validation_generator,
                                      nb_val_samples = len(validation_generator) if validation_generator is not None else None,
                                      verbose=verbose,
                                      callbacks=callbacks)
        
        hist.history['lr'] = clr.history['lr']
        hist.history['iterations'] = clr.history['iterations']
        if cycle_momentum:
            hist.history['momentum'] = clr.history['momentum']
        self.history = hist
        return hist

        
    def predict(self, tile_generator):

        smscores = []
        segmentations = []

        for tileIdx in tqdm(range(tile_generator.__len__())):
            tile = tile_generator.__getitem__(tileIdx)
            outIdx = tile[0]["image_index"]
            outShape = tile[0]["image_shape"]
            outSlice = tile[0]["out_slice"]
            inSlice = tile[0]["in_slice"]
            softmax_score = self.testModel.predict(tile[0]["data"], verbose=0)
            if len(smscores) < outIdx + 1:
                smscores.append(np.empty((*outShape, self.n_classes)))
                segmentations.append(np.empty(outShape))
            smscores[outIdx][outSlice] = softmax_score[0][inSlice]
            segmentations[outIdx][outSlice] = np.argmax(
                softmax_score[0], axis=-1)[inSlice]

        return smscores, segmentations
    
    # from https://github.com/amaiya/ktrain/blob/24ec38b355f0e4decc5077194fcccfc2f1232cb2/ktrain/text/learner.py
    def set_weight_decay(self, wd=0.005):
        """
        Sets global weight decay layer-by-layer using L2 regularization.
        Args:
          wd(float): weight decay (see note above)
        Returns:
          None
              
        """

        for layer in self.trainModel.layers:
            if hasattr(layer, 'kernel_regularizer') and hasattr(layer, 'kernel'):
                layer.kernel_regularizer= regularizers.l2(wd)
                layer.add_loss(regularizers.l2(wd)(layer.kernel))

            if hasattr(layer, 'bias_regularizer') and hasattr(layer, 'bias'):
                layer.bias_regularizer= regularizers.l2(wd)
                layer.add_loss(regularizers.l2(wd)(layer.bias))
        
        self._recompile()
        return
    
    def _recompile(self):
        self.trainModel.compile(optimizer=self.opt, loss=None)
        
        labels = self.trainModel.input[1]
        softmax_score = self.trainModel.layers[-1].output
        for m in self.metrics:
            self.trainModel.metrics_tensors.append(m(labels, softmax_score))
            self.trainModel.metrics_names.append(m.__name__)
        return   
    
    def freeze(self, freeze_range=None):
        """
        If freeze_range is None, makes all layers trainable=False except last layer.
        If freeze_range is given, freezes the first <freeze_range> layers and
        unfrezes all remaining layers.
        Args:
            freeze_range(int): number of layers to freeze
        Returns:
            None
        """

        if freeze_range is None:
            # freeze everything except last layer
            for layer in self.trainModel.layers:
                if layer.name != 'conv_u0d-score': 
                    #print(layer.name)
                    layer.trainable=False
                    
        else:
            # freeze all layers up to and including layer_id
            if type(freeze_range) != type(1) or freeze_range <1: 
                raise ValueError('freeze_range must be integer > 0')
            for i, layer in enumerate(self.trainModel.layers):
                if i < freeze_range: 
                    layer.trainable=False
                else:
                    layer.trainable=True
        self._recompile()
        return    
    
    # https://github.com/amaiya/ktrain/blob/55e00d850b8dc43adcca64d0bb4f3f00c283c907/ktrain/core.py#L262 
    def unfreeze(self, exclude_range=None):
        """
        Make every layer trainable except those in exclude_range.
        unfreeze is simply a proxy method to freeze.
        """
        # make all layers trainable
        for i, layer in enumerate(self.trainModel.layers):
            layer.trainable = True
        if exclude_range:
            for i, layer in enumerate(self.trainModel.layers[:exclude_range]):
                layer.trainable = False
        self._recompile()
        return