import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug
from tqdm import trange

import config
import ops
import utils

from parameters import ParamsDict

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('runs', 1, 'Number of consecutive runs')

# Random, but not too much.
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

class GANModel(ParamsDict):
    """A GANModel contains all information about model itself."""

    def __init__(self, *args, **kwargs):
        super(GANModel, self).__init__(*args, **kwargs)


def input_fn(images, labels, params):
    # Dataset input
    x_placeholder = tf.placeholder(images.dtype, shape=images.shape, name='x_placeholder')
    y_placeholder = tf.placeholder(labels.dtype, shape=labels.shape, name='y_placeholder')

    x_var = tf.Variable(x_placeholder, name="x_var")
    y_var = tf.Variable(y_placeholder, name="y_var")

    input_iterator = (
        tf.data.Dataset.from_tensor_slices((x_var.initialized_value(),
                                            y_var.initialized_value()))
            .repeat()
            .apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))
            .make_initializable_iterator())

    z = tf.random_normal([params.batch_size, params.noise_size], dtype=tf.float32, name='z')
    zy = tf.random_uniform([params.batch_size], 0, params.dataset.labels_size, dtype=tf.int32, name='zy')
    zy = tf.one_hot(zy, params.dataset.labels_size)
    zy = tf.cast(zy, tf.float32)

    return (x_placeholder,
            y_placeholder,
            z,
            zy,
            input_iterator,
            input_iterator.initializer)


def model_fn(images, labels, params):
    tf.reset_default_graph()

    # Placeholders & Inputs
    training = tf.placeholder(tf.bool, name='training')
    (x_placeholder, y_placeholder,
     z, zy,
     dataset_it, dataset_init) = input_fn(images, labels, params)

    # Submodels' architectures
    def _generator(z, zy):
        with tf.variable_scope(params.gen_scope):
            imh, imw = params.dataset.image_size, params.dataset.image_size

            hidden_layers_num = 3
            imdiv = 2 ** hidden_layers_num

            h0 = tf.concat([z, zy], axis=1)

            h1 = ops.fully_connected(h0, (imh // imdiv) * (imw // imdiv) * params.gen_filters * 4, 'h1')
            if params.use_batch_norm:
                h1 = ops.batch_norm(h1, name='bn1')
            h1 = tf.reshape(h1, [-1, imh // imdiv, imw // imdiv, params.gen_filters * 4])
            h1 = ops.lrelu(h1)
            h1 = ops.dropout(h1, training=training, keep=params.gen_keep_dropout, name='dropout1')
            h1 = ops.concat(h1, zy)

            h2 = ops.deconvolution(h1, params.gen_filters_size, params.gen_filters * 2, name='h2')
            if params.use_batch_norm:
                h2 = ops.batch_norm(h2, name='bn2')
            h2 = ops.lrelu(h2)
            h2 = ops.dropout(h2, training=training, keep=params.gen_keep_dropout, name='dropout2')
            h2 = ops.concat(h2, zy)

            h3_pure = ops.deconvolution(h2, params.gen_filters_size, params.gen_filters, name='h3')
            h3 = h3_pure
            if params.use_batch_norm:
                h3 = ops.batch_norm(h3, name='bn3')
            h3 = ops.lrelu(h3)
            h3 = ops.dropout(h3, training=training, keep=params.gen_keep_dropout, name='dropout3')
            h3 = ops.concat(h3, zy)

            h4 = ops.deconvolution(h3, params.gen_filters_size, params.dataset.channels_size, name='h4')
            return tf.nn.tanh(h4), {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'h3_pure': h3_pure, 'h4': h4}

    def _discriminator(x, y, reuse_vars=False):
        with tf.variable_scope(params.dis_scope, reuse=reuse_vars):
            h0 = ops.concat(x, y)

            h1_pure = ops.convolution(h0, params.dis_filters_size, params.dis_filters, name='h1')
            h1 = h1_pure
            if params.use_batch_norm:
                h1 = ops.batch_norm(h1, name='bn1')
            h1 = ops.lrelu(h1)
            h1 = ops.concat(h1, y)

            h2 = ops.convolution(h1, params.dis_filters_size, params.dis_filters * 2, name='h2')
            if params.use_batch_norm:
                h2 = ops.batch_norm(h2, name='bn2')
            h2 = ops.lrelu(h2)
            h2 = ops.concat(h2, y)

            h3 = ops.convolution(h2, params.dis_filters_size, params.dis_filters * 4, name='h3')
            if params.use_batch_norm:
                h3 = ops.batch_norm(h3, name='bn3')
            h3 = ops.lrelu(h3)
            h3 = ops.concat(h3, y)

            h4 = tf.reshape(h3, [params.batch_size, -1])
            h4 = ops.fully_connected(h4, 1, 'h4')
            return h4, {'h0': h0, 'h1': h1, 'h1_pure': h1_pure, 'h2': h2, 'h3': h3, 'h4': h4}

    # Model layout
    x, y = dataset_it.get_next()
    gen_output, gen_layers = _generator(z, zy)
    dis_real_output, dis_real_layers = _discriminator(x, y)
    dis_fake_output, dis_fake_layers = _discriminator(gen_output, zy, reuse_vars=True)

    # Probabilities
    dis_real_output_prob = tf.nn.sigmoid(dis_real_output)
    dis_fake_output_prob = tf.nn.sigmoid(dis_fake_output)

    # Losses
    with tf.name_scope('losses'):
        gen_loss_prob = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_output,
                                                    labels=tf.fill(tf.shape(dis_fake_output), 1.0)))
        gen_feature_matching_loss = tf.reduce_mean(tf.square(gen_layers['h3_pure'] - dis_fake_layers['h1_pure']))
        gen_loss = (gen_loss_prob + gen_feature_matching_loss) / 2
        dis_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_output,
                                                    labels=tf.fill(tf.shape(dis_real_output), 0.9)))
        dis_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_output,
                                                    labels=tf.fill(tf.shape(dis_fake_output), 0.0)))
        dis_loss = (dis_real_loss + dis_fake_loss) / 2

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=params.gen_scope)
    dis_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=params.dis_scope)

    with tf.name_scope('optimizers'):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=params.gen_scope)):
            gen_train_optimizer = tf.train.AdamOptimizer(params.gen_lr, beta1=0.5)
            gen_grads = gen_train_optimizer.compute_gradients(gen_loss, var_list=gen_vars)
            gen_train_opt = gen_train_optimizer.apply_gradients(gen_grads)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=params.dis_scope)):
            dis_train_optimizer = tf.train.AdamOptimizer(params.dis_lr, beta1=0.5)
            dis_grads = dis_train_optimizer.compute_gradients(dis_loss, var_list=dis_vars)
            dis_train_opt = dis_train_optimizer.apply_gradients(dis_grads)

    return GANModel(
        training=training,
        x=x,
        y=y,
        z=z,
        zy=zy,
        x_placeholder=x_placeholder,
        y_placeholder=y_placeholder,
        dataset_it=dataset_it,
        dataset_init=dataset_init,
        gen_layers=gen_layers,
        gen_output=gen_output,
        gen_grads=gen_grads,
        gen_loss=gen_loss,
        gen_loss_prob=gen_loss_prob,
        gen_feature_matching_loss=gen_feature_matching_loss,
        gen_train_opt=gen_train_opt,
        dis_real_layers=dis_real_layers,
        dis_fake_layers=dis_fake_layers,
        dis_real_output=dis_real_output,
        dis_fake_output=dis_fake_output,
        dis_real_output_prob=dis_real_output_prob,
        dis_fake_output_prob=dis_fake_output_prob,
        dis_real_loss=dis_real_loss,
        dis_fake_loss=dis_fake_loss,
        dis_loss=dis_loss,
        dis_grads=dis_grads,
        dis_train_opt=dis_train_opt)


def summaries_fn(model, params):
    # Summaries
    tf.summary.scalar('dis_real_loss', model.dis_real_loss)
    tf.summary.scalar('dis_fake_loss', model.dis_fake_loss)
    tf.summary.scalar('gen_loss', model.gen_loss)
    tf.summary.scalar('gen_loss_prob', model.gen_loss_prob)
    tf.summary.scalar('gen_feature_matching_loss', model.gen_feature_matching_loss)
    tf.summary.histogram('gen_output_hist', tf.reshape(model.gen_output, [-1]))
    tf.summary.histogram('dis_fake_output_prob_hist', tf.reshape(model.dis_fake_output_prob, [-1]))

    for name, layer in model.gen_layers.items():
        tf.summary.histogram('gen_%s_hist' % name, tf.reshape(layer, [-1]))

    for name, layer in model.dis_real_layers.items():
        tf.summary.histogram('dis_%s_hist_real' % name, tf.reshape(layer, [-1]))

    for name, layer in model.dis_fake_layers.items():
        tf.summary.histogram('dis_%s_hist_fake' % name, tf.reshape(layer, [-1]))

    for grad, var in model.gen_grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gen-gradients', grad)

    for grad, var in model.dis_grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/dis-gradients', grad)

    tf.summary.histogram('x_hist', tf.reshape(model.x, [-1]))
    tf.summary.image('gen_output', model.gen_output, 8)
    tf.summary.image('x', model.x, 2)

    merged_summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(params.name, tf.get_default_graph(), flush_secs=15)

    return merged_summaries, writer


def select_model(sess, model, dis_loss, gen_loss, using_new_dis, using_new_gen, params):
    if using_new_dis and gen_loss >= params.loss_diff_threshold * dis_loss:
        using_new_dis = False
        print("Using old discriminator...")
        model.saver_dis.save(sess, '%s/new_dis.ckpt' % params.checkpoint_dir)

    if not using_new_dis and gen_loss < params.loss_diff_threshold_back * dis_loss:
        using_new_dis = True
        print("Using new discriminator...")
        model.saver_dis.restore(sess, '%s/new_dis.ckpt' % params.checkpoint_dir)

    if using_new_gen and dis_loss >= params.loss_diff_threshold * gen_loss:
        using_new_gen = False
        print("Using old generator...")
        model.saver_gen.save(sess, '%s/new_gen.ckpt' % params.checkpoint_dir)

    if not using_new_gen and dis_loss < params.loss_diff_threshold_back * gen_loss:
        using_new_gen = True
        print("Using new generator...")
        model.saver_gen.restore(sess, '%s/new_gen.ckpt' % params.checkpoint_dir)

    return using_new_dis, using_new_gen


def train(sess, model, summaries, writer, params):
    gen_ops = [model.gen_train_opt, model.gen_loss, model.gen_output]
    dis_ops = [model.dis_train_opt, model.dis_loss, model.dis_real_loss, model.dis_fake_loss]

    # Pre-train discriminator.
    print("Pretraining...")
    for _ in trange(params.pretrain_steps):
        sess.run(dis_ops, feed_dict={model.training: True})

    for epoch in range(params.epochs):
        # Main training.
        print("Epoch: %d" % epoch)
        dis_loss, dis_real_loss, dis_fake_loss, gen_loss = [0.5] * 4
        decayed_dis_loss = dis_loss
        decayed_gen_loss = gen_loss
        dis_steps, gen_steps = 0, 0
        using_new_gen = True
        using_new_dis = True

        for step in trange(params.steps):
            real_step = epoch * params.steps + step

            using_new_dis, using_new_gen = select_model(
                sess, model, decayed_dis_loss, decayed_gen_loss, using_new_dis, using_new_gen, params)

            # Discriminator
            if using_new_dis:
                _, dis_loss, dis_real_loss, dis_fake_loss = sess.run(
                    dis_ops, feed_dict={model.training: True})
                dis_steps += 1

            # Generator
            if using_new_gen:
                _, gen_loss, gen_output = sess.run(gen_ops, feed_dict={model.training: True})
                gen_steps += 1

            # Write summaries
            if step > 0 and step % params.summaries_steps == 0:
                summary = sess.run(summaries, feed_dict={model.training: True})
                writer.add_summary(summary, real_step)

            # Print losses
            if step % params.prints_steps == 0:
                print('dis_real: %f | dis_fake: %f | gen: %f | dis_steps: %d | gen_steps: %d | dis: %d | gen: %d' % (
                    dis_real_loss, dis_fake_loss, gen_loss, dis_steps, gen_steps,
                    using_new_dis, using_new_gen))

            if step % params.draw_steps == 0:
                gen_output, labels, dis_output = sess.run(
                    [model.gen_output, model.zy, model.dis_fake_output_prob],
                    feed_dict={model.training: True})
                labels = np.argmax(labels, axis=1)
                utils.save_images((gen_output + 1) / 2, labels, dis_output, real_step, params)

            if step in params.save_old_steps:
                print("Discriminator saved in:",
                      model.saver_dis.save(sess, '%s/old_dis.ckpt' % params.checkpoint_dir))
                print("Generator saved in:",
                      model.saver_gen.save(sess, '%s/old_gen.ckpt' % params.checkpoint_dir))

            # Checkpoint.
            if step > 0 and step in params.save_steps:
                print("Model saved in:", model.saver.save(
                        sess, '%s/step%d.ckpt' % (params.checkpoint_dir, real_step)))

            decayed_gen_loss = 0.95 * decayed_gen_loss + 0.05 * gen_loss
            decayed_dis_loss = 0.95 * decayed_dis_loss + 0.05 * dis_loss


def generate(sess, model, params):
    print("Generating images...")
    images, labels, probs = None, None, None
    threshold_probs = np.zeros(params.labels_size)
    for i in trange(params.nb_generated // params.batch_size):
        gen_outputs, dis_outputs, gen_labels = sess.run(
            [model.gen_output, model.dis_fake_output_prob, model.zy], feed_dict={model.training: False})
        images = gen_outputs if not i else np.ops.concatenate((images, gen_outputs), axis=0)
        labels = gen_labels if not i else np.ops.concatenate((labels, gen_labels), axis=0)
        probs = dis_outputs if not i else np.ops.concatenate((probs, dis_outputs), axis=0)
        for dis_output, label in zip(dis_outputs, gen_labels):
            threshold_probs[np.argmax(label)] += dis_output
    probs = np.reshape(probs, [-1])
    threshold_probs /= (1 + np.sum(labels, axis=0))
    image_thresholds = threshold_probs[np.argmax(labels, axis=1)]

    # Generate images with more than average discriminator's decision.
    images = images[probs >= image_thresholds]
    labels = labels[probs >= image_thresholds]

    np.save('%s/images.npy' % params.images_dir, images)
    np.save('%s/labels.npy' % params.images_dir, labels)
    print("Generated images saved to: %s" % params.images_dir)


def run(images, labels, params):
    params.describe()

    model = model_fn(images, labels, params)
    summaries, writer = summaries_fn(model, params)
    model.saver = tf.train.Saver()
    model.saver_gen = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=params.gen_scope))
    model.saver_dis = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=params.dis_scope))

    sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session()) if params.debug else tf.Session()
    with sess:
        # Load model from checkpoint.
        if params.model_path:
            print("Restoring model from: %s..." % params.model_path)
            model.saver.restore(sess, params.model_path)
        else:
            sess.run([tf.global_variables_initializer(), model.dataset_init], feed_dict={
                model.x_placeholder: images,
                model.y_placeholder: labels})

        if params.mode == 'train':
            train(sess, model, summaries, writer, params)
        elif params.mode == 'generate':
            generate(sess, model, params)
        else:
            raise ValueError('Unknown mode: %s' % params.mode)


def main(_):
    print('Available devices:', utils.get_available_devices())

    for _ in range(FLAGS.runs):
        # Initialize parameters.
        params = config.GANParams(config.params_defs.initialized())

        # Override parameters by flag values.
        for k, v in FLAGS.__dict__['__flags'].items():
            if v is not None:
                params[k] = v

        # Load dataset.
        images, labels = params.dataset.get(params)

        # Run training.
        run(images, labels, params)


if __name__ == '__main__':
    config.params_defs.define_flags()
    tf.app.run()
