"""
Core Module for Vanilla Gradients
"""
import tensorflow as tf

from tf_explain.utils.display import grid_display
from tf_explain.utils.image import transform_to_normalized_grayscale
from tf_explain.utils.saver import save_grayscale


class VanillaGradients:

    """
    Perform gradients backpropagation for a given input

    Paper: [Deep Inside Convolutional Networks: Visualising Image Classification
        Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
    """

    def explain(self, validation_data, model, class_index, **extra):
        """
        Perform gradients backpropagation for a given input

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class

        Returns:
            numpy.ndarray: Grid of all the gradients
        """
        images, _ = validation_data

        gradients = self.compute_gradients(images, model, class_index, **extra)

        grayscale_gradients = transform_to_normalized_grayscale(
            tf.abs(gradients)
        ).numpy()

        grid = grid_display(grayscale_gradients)

        return grid

    @staticmethod
    def compute_gradients(images, model, class_index, **extra):
        """
        Compute gradients for target class.

        Args:
            images (numpy.ndarray): 4D-Tensor of images with shape (batch_size, H, W, 3)
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class

        Returns:
            tf.Tensor: 4D-Tensor
        """

        num_classes = model.output.shape[1]

        expected_output = tf.ones([1, 14, 14, 14, 1])

        #if gt is not None:
        #    expected_output = gt
        #else:
        #    expected_output = tf.one_hot([class_index] * images.shape[0], num_classes)

        #import ipdb; ipdb.set_trace()
        inputs = tf.cast(images, tf.float32)
        if model.name == "unet":
            expected_output=extra['gt']
            with tf.GradientTape() as tape:
                inputs = tf.cast(inputs, tf.float32)
                tape.watch(inputs)
                predictions = model(inputs)
                loss = tf.keras.losses.mse(
                    expected_output, predictions
                )   
                grad = tape.gradient(loss, inputs)
                print('unet gradient')
            return grad 


        elif model.name == 'discriminator':
            expected_output = tf.ones([1, 14, 14, 14, 1])
           # inputs = [inputs, extra['pred']]
           # inputs = tf.cast(inputs, tf.float32)
            #input_0 = tf.cast(inputs[0], tf.float32)
            #input_1 = tf.cast(inputs[1], tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(inputs)
               # tape1.watch(inputs)
               # predictions = model([input_0, input_1])
                #predictions = model([extra['mri '], inputs])
                predictions = model([inputs, extra['pred']])
                loss = tf.keras.losses.mse(
                    expected_output, predictions
                )
                tape_grad = tape.gradient(loss, inputs)
                
            return tape_grad 

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Gtid of all the smoothed gradients
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_grayscale(grid, output_dir, output_name)
