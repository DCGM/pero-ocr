import tensorflow as tf
from functools import partial
from math import sqrt
from sys import stderr

def sequential_conv_block(net, train_phase, layer_count, filter_count, kernel_size=3, dilations=[1, 1], padding='same',
                          layer_norm=False):
    for i in range(layer_count):
        net = tf.layers.conv2d(
            net, filters=filter_count, kernel_size=kernel_size,
            dilation_rate=dilations, padding=padding, use_bias=False)
        if layer_norm:
            print('Layer norm 1', file=stderr)
            net = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)(net)
        else:
            net = tf.layers.batch_normalization(net, training=train_phase, fused=True)
        net = tf.nn.relu(net)
    return net


def sequential_conv_block_1d(net, train_phase, layer_count, filter_count, kernel_size=3, dilations=1, padding='same',
                             layer_norm=False):
    for i in range(layer_count):
        net = tf.layers.conv1d(
            net, filters=filter_count, kernel_size=kernel_size,
            dilation_rate=dilations, padding=padding)
        if layer_norm:
            print('Layer norm 2', file=stderr)
            net = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)(net)
        else:
            net = tf.layers.batch_normalization(net, training=train_phase, fused=True)
        net = tf.nn.relu(net)
    return net


def get_aggregation(net, count, train_phase):
    height = int(net.shape[1])
    print('height', height, file=stderr)
    net = sequential_conv_block(net, train_phase, 1, count, kernel_size=(height, 1), padding="valid")
    net = net[:, 0, :, :]
    return net


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def build_simple_net(class_count, input_tensor, train_phase, keep_prob, seq_len,
                     num_hidden=96, block_layer_count=2, base_filter_count=12,
                     block_count=3, output_subsampling=4):
    class_count = class_count + 1
    net = input_tensor
    print('INPUT', net.shape, file=stderr)

    for i in range(block_count):
        net = sequential_conv_block(net, train_phase, block_layer_count, base_filter_count * (2**i))
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.nn.dropout(net, keep_prob)
        print('CONV BLOCK', i, net.shape, file=stderr)

    net = get_aggregation(net, num_hidden, train_phase)
    net = tf.nn.dropout(net, keep_prob**2)
    print("1D", net.shape, file=stderr)

    net = sequential_conv_block_1d(net, train_phase, block_layer_count, num_hidden)
    net = tf.nn.dropout(net, keep_prob**2)
    print("1D", net.shape, file=stderr)

    for i in range(block_count - int(sqrt(output_subsampling))):
        net = tf.keras.layers.UpSampling1D()(net)
        net = sequential_conv_block_1d(net, train_phase, block_layer_count, num_hidden)
        print("1D upsampling", net.shape, file=stderr)

    logits = tf.layers.conv1d(net, class_count, 1)
    print("LOGIT SHAPE", logits.shape, file=stderr)

    logits_t = tf.transpose(logits, (1, 0, 2))
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_t, seq_len, merge_repeated=True)

    return decoded, logits, logits_t, log_prob


def build_deep_net(class_count, input_tensor, train_phase, keep_prob, seq_len,
                   num_hidden=96, num_recurrent_layers=1, block_layer_count=2, base_filter_count=12, block_count=3,
                   output_subsampling=4, layer_norm=False):
    class_count = class_count + 1
    net = input_tensor
    print('INPUT', net.shape, file=stderr)

    bypass = []
    for i in range(block_count):
        net = sequential_conv_block(net, train_phase, block_layer_count, base_filter_count * (2**i),
                                    layer_norm=layer_norm)
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.nn.dropout(net, keep_prob)
        bp = get_aggregation(net, base_filter_count * 2, train_phase)
        if i != block_count - 1:
            bp = tf.layers.max_pooling1d(bp, 2**(block_count-i-1), 2**(block_count-i-1))
        bypass.append(bp)
        print('CONV BLOCK', i, net.shape, file=stderr)

    net = sequential_conv_block(net, train_phase, 1, num_hidden, padding="same", layer_norm=layer_norm)

    net = get_aggregation(net, num_hidden, train_phase)
    net = tf.nn.dropout(net, keep_prob**2)
    bypass.append(net)

    print("1D", net.shape, file=stderr)

    net = tf.concat(bypass + [net], axis=2)
    lstm = tf.contrib.cudnn_rnn.CudnnGRU(num_recurrent_layers, num_hidden,
                                         direction='bidirectional', name="awesome_lstm")
    net = tf.transpose(net, [1, 0, 2])
    net, x = lstm(net)
    net = tf.transpose(net, [1, 0, 2])
    if layer_norm:
        print('Layer norm 3', file=stderr)
        net = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)(net)
    else:
        net = tf.layers.batch_normalization(net, training=train_phase)

    net = tf.nn.relu(net)
    net = tf.nn.dropout(net, keep_prob ** 2)

    net = tf.concat(bypass + [net], axis=2)

    for i in range(block_count - int(sqrt(output_subsampling))):
        net = tf.keras.layers.UpSampling1D()(net)
        net = sequential_conv_block_1d(net, train_phase, block_layer_count, num_hidden, layer_norm=layer_norm)
        print("1D upsampling", net.shape, file=stderr)

    logits = tf.layers.conv1d(net, class_count, 1)
    print("LOGIT SHAPE", logits.shape, file=stderr)

    logits_t = tf.transpose(logits, (1, 0, 2))
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_t, seq_len, merge_repeated=True)

    return decoded, logits, logits_t, log_prob

def build_u_net(class_count, input_tensor, train_phase, keep_prob, seq_len,
                   num_hidden=96, num_recurrent_layers=1, block_layer_count=2, base_filter_count=12, block_count_2d=3,
                   block_count_1d=2, output_subsampling=4, layer_norm=False):

    class_count = class_count + 1
    net = input_tensor
    print('INPUT', net.shape, file=stderr)

    scales = []
    for i in range(block_count_2d):
        net = sequential_conv_block(net, train_phase, block_layer_count, base_filter_count * (2**i),
                                    layer_norm=layer_norm)
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.nn.dropout(net, keep_prob)
        scales.append(get_aggregation(net, base_filter_count * 2, train_phase))
        print('CONV BLOCK 2D', i, net.shape, file=stderr)

    net = get_aggregation(net, num_hidden, train_phase)

    for i in range(block_count_1d):
        net = sequential_conv_block_1d(net, train_phase, block_layer_count, base_filter_count * (2**(i+block_count_2d)),
                                                                                           layer_norm=layer_norm)
        net = tf.layers.max_pooling1d(net, 2, 2)
        net = tf.nn.dropout(net, keep_prob)
        if i != block_count_1d - 1:
            scales.append(net)
        print('CONV BLOCK 1D', i, net.shape, file=stderr)

    print("1D", net.shape, file=stderr)

    while True:
        bypass = []
        for scale in scales:
            if net.shape[1] == scale.shape[1]:
                bypass.append(scale)
        if bypass:
            net = tf.concat(bypass + [net], axis=2)

        scale_id = int(input_tensor.shape[2]) // int(net.shape[1])
        filter_count = base_filter_count * scale_id
        print('Upsampling input', input_tensor.shape, net.shape, scale_id, filter_count, file=stderr)

        lstm = tf.contrib.cudnn_rnn.CudnnGRU(num_recurrent_layers, filter_count // 2,
                                             direction='bidirectional')
        lstm_net = tf.transpose(net, [1, 0, 2])
        lstm_net, x = lstm(lstm_net)
        lstm_net = tf.transpose(lstm_net, [1, 0, 2])
        if layer_norm:
            lstm_net = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)(net)
        else:
            lstm_net = tf.layers.batch_normalization(lstm_net, training=train_phase)

        net = sequential_conv_block_1d(net, train_phase, 1, filter_count, kernel_size=1, dilations=1,
                                 padding='same',
                                 layer_norm=layer_norm)
        net = net + lstm_net
        net = tf.nn.dropout(net, keep_prob)
        if net.shape[1] < int(input_tensor.shape[2]) / output_subsampling:
            net = tf.keras.layers.UpSampling1D()(net)
        else:
            break
        print('Upsampling output', net.shape, file=stderr)

    logits = tf.layers.conv1d(net, class_count, 1)
    print("LOGIT SHAPE", logits.shape, file=stderr)

    logits_t = tf.transpose(logits, (1, 0, 2))
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_t, seq_len, merge_repeated=True)

    return decoded, logits, logits_t, log_prob




def build_deep_net_with_multiple_last_layer(class_count, input_tensor, train_phase,
                                            keep_prob, seq_len_all, number_of_last_layers, num_hidden=96,
                                            num_recurrent_layers=1, block_layer_count=2,
                                            base_filter_count=12, block_count=3, output_subsampling=4):
    class_count = class_count + 1
    net = input_tensor
    print('INPUT', net.shape, file=stderr)

    bypass = []
    for i in range(block_count):
        net = sequential_conv_block(net, train_phase, block_layer_count, base_filter_count * (2**i))
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.nn.dropout(net, keep_prob)
        bp = get_aggregation(net, base_filter_count * 2, train_phase)

        if i != block_count - 1:
            bp = tf.layers.max_pooling1d(bp, 2**(block_count-i-1), 2**(block_count-i-1))
        bypass.append(bp)
        print('CONV BLOCK', i, net.shape, file=stderr)

    net = sequential_conv_block(net, train_phase, 1, num_hidden, padding="same")

    net = get_aggregation(net, num_hidden, train_phase)
    net = tf.nn.dropout(net, keep_prob**2)
    bypass.append(net)

    print("1D", net.shape, file=stderr)

    net = tf.concat(bypass + [net], axis=2)
    lstm = tf.contrib.cudnn_rnn.CudnnGRU(num_recurrent_layers, num_hidden,
                                         direction='bidirectional', name="awesome_lstm")
    net = tf.transpose(net, [1, 0, 2])
    net, x = lstm(net)
    net = tf.transpose(net, [1, 0, 2])
    net = tf.layers.batch_normalization(net, training=train_phase)
    net = tf.nn.relu(net)
    net = tf.nn.dropout(net, keep_prob ** 2)

    net = tf.concat(bypass + [net], axis=2)

    for i in range(block_count - int(sqrt(output_subsampling))):
        net = tf.keras.layers.UpSampling1D()(net)
        net = sequential_conv_block_1d(net, train_phase, block_layer_count, num_hidden)
        print("1D upsampling", net.shape, file=stderr)

    mask_all = []
    for last_layer_index in range(number_of_last_layers):
        mask_all.append(tf.placeholder(tf.bool, shape=[None], name='MASK_{}'.format(last_layer_index)))

    logits_all = []
    for i, mask in zip(range(number_of_last_layers), mask_all):
        logits_all.append(tf.layers.conv1d(tf.boolean_mask(net, mask, axis=0), class_count, 1))
        print("LOGIT SHAPE", logits_all[i].shape, file=stderr)

    logits_t_all = []
    decoded_all = []
    log_prob_all = []
    for logits, seq_len in zip(logits_all, seq_len_all):
        logits_t_all.append(tf.transpose(logits, (1, 0, 2)))
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_t_all[-1], seq_len, merge_repeated=True)
        decoded_all.append(decoded)
        log_prob_all.append(log_prob)

    return decoded_all, logits_all, logits_t_all, log_prob_all, mask_all


def build_deep_net_with_multiple_LSTM(class_count, input_tensor, train_phase,
                                      keep_prob, seq_len_all, number_of_last_layers, num_hidden=96,
                                      num_recurrent_layers=1, block_layer_count=2,
                                      base_filter_count=12, block_count=3, output_subsampling=4):
    class_count = class_count + 1
    net = input_tensor
    print('INPUT', net.shape, file=stderr)

    bypass = []
    for i in range(block_count):
        net = sequential_conv_block(net, train_phase, block_layer_count, base_filter_count * (2**i))
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.nn.dropout(net, keep_prob)
        bp = get_aggregation(net, base_filter_count * 2, train_phase)

        if i != block_count - 1:
            bp = tf.layers.max_pooling1d(bp, 2**(block_count-i-1), 2**(block_count-i-1))
        bypass.append(bp)
        print('CONV BLOCK', i, net.shape, file=stderr)

    net = sequential_conv_block(net, train_phase, 1, num_hidden, padding="same")

    net = get_aggregation(net, num_hidden, train_phase)
    net = tf.nn.dropout(net, keep_prob**2)
    bypass.append(net)

    print("1D", net.shape, file=stderr)

    net = tf.concat(bypass + [net], axis=2)

    mask_all = []
    for last_layer_index in range(number_of_last_layers):
        mask_all.append(tf.placeholder(tf.bool, shape=[None], name='MASK_{}'.format(last_layer_index)))

    lstm_all = []
    for i, mask in zip(range(number_of_last_layers), mask_all):
        branch = tf.transpose(tf.boolean_mask(net, mask, axis=0), [1, 0, 2])
        print("awesome_lstm_{}".format(i), file=stderr)
        if i == 0:
            lstm = tf.contrib.cudnn_rnn.CudnnGRU(num_recurrent_layers, num_hidden, direction='bidirectional',
                                                 name="awesome_lstm")
        else:
            lstm = tf.contrib.cudnn_rnn.CudnnGRU(num_recurrent_layers, num_hidden, direction='bidirectional',
                                                 name="awesome_lstm_{}".format(i))
        branch, x = lstm(branch)
        branch = tf.transpose(branch, [1, 0, 2])
        branch = tf.layers.batch_normalization(branch, training=train_phase)
        branch = tf.nn.relu(branch)
        branch = tf.nn.dropout(branch, keep_prob ** 2)
        masked_bypass = tf.concat(bypass, axis=2)
        masked_bypass = tf.boolean_mask(masked_bypass, mask, axis=0)
        branch = tf.concat([masked_bypass, branch], axis=2)
        lstm_all.append(branch)

    logits_all = []
    for i in range(number_of_last_layers):
        logits_all.append(tf.layers.conv1d(lstm_all[i], class_count, 1))
        print("LOGIT SHAPE", logits_all[i].shape, file=stderr)

    logits_t_all = []
    decoded_all = []
    log_prob_all = []
    for logits, seq_len in zip(logits_all, seq_len_all):
        logits_t_all.append(tf.transpose(logits, (1, 0, 2)))
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_t_all[-1], seq_len, merge_repeated=True)
        decoded_all.append(decoded)
        log_prob_all.append(log_prob)

    return decoded_all, logits_all, logits_t_all, log_prob_all, mask_all


def build_CRNN(class_count, input_tensor, train_phase, keep_prob, seq_len):
    class_count = class_count + 1
    net = input_tensor
    print('INPUT', net.shape, file=stderr)

    net = sequential_conv_block(net, train_phase, 1, 64)
    net = tf.layers.max_pooling2d(net, 2, 2)

    net = sequential_conv_block(net, train_phase, 1, 128)
    net = tf.layers.max_pooling2d(net, 2, 2)

    net = sequential_conv_block(net, train_phase, 2, 256)
    net = tf.layers.max_pooling2d(net, (2, 1), (2, 1))

    net = sequential_conv_block(net, train_phase, 2, 512)
    net = tf.layers.max_pooling2d(net, (2, 1), (2, 1))

    net = tf.layers.conv2d(
        net, filters=512, kernel_size=(2, 1),
        padding='valid', use_bias=False)
    net = tf.nn.relu(net)
    print("CONV SHAPE", net.shape, file=stderr)

    net = net[:, 0, :, :]


    lstm = tf.contrib.cudnn_rnn.CudnnGRU(2, 256, direction='bidirectional', name="awesome_lstm")
    net = tf.transpose(net, [1, 0, 2])
    net, x = lstm(net)
    net = tf.transpose(net, [1, 0, 2])

    logits = tf.layers.conv1d(net, class_count, 1)
    print("LOGIT SHAPE", logits.shape, file=stderr)

    logits_t = tf.transpose(logits, (1, 0, 2))
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_t, seq_len, merge_repeated=True)

    return decoded, logits, logits_t, log_prob


line_nets = {
    # NET_001
    'CRNN': partial(build_CRNN),
    'LSTM_BC_3_BLC_2_BFC_12': partial(
        build_deep_net, num_hidden=96, num_recurrent_layers=1,
        block_layer_count=2, base_filter_count=12, block_count=3, output_subsampling=4),
    'LSTM_BC_2_BLC_3_BFC_12': partial(
        build_deep_net, num_hidden=96, num_recurrent_layers=1,
        block_layer_count=3, base_filter_count=12, block_count=2, output_subsampling=4),
    'LSTM_BC_4_BLC_2_BFC_12': partial(
        build_deep_net, num_hidden=96, num_recurrent_layers=1,
        block_layer_count=2, base_filter_count=12, block_count=4, output_subsampling=4),
    # NET_002
    'LSTM_BC_3_BLC_2_BFC_24': partial(
        build_deep_net, num_hidden=256, num_recurrent_layers=1,
        block_layer_count=2, base_filter_count=24, block_count=3, output_subsampling=4),
    'LSTM_BC_3_BLC_2_BFC_24_LAST_N': partial(
        build_deep_net_with_multiple_last_layer, num_hidden=256, num_recurrent_layers=1,
        block_layer_count=2, base_filter_count=24, block_count=3, output_subsampling=4),
    'LSTM_BC_2_BLC_3_BFC_24': partial(
        build_deep_net, num_hidden=256, num_recurrent_layers=1,
        block_layer_count=3, base_filter_count=24, block_count=2, output_subsampling=4),
    'LSTM_BC_2_BLC_3_BFC_24_LN': partial(
        build_deep_net, num_hidden=256, num_recurrent_layers=1,
        block_layer_count=3, base_filter_count=24, block_count=2, output_subsampling=4, layer_norm=True),
    'LSTM_BC_2_BLC_3_BFC_24_LAST_N': partial(
        build_deep_net_with_multiple_last_layer, num_hidden=256, num_recurrent_layers=1,
        block_layer_count=3, base_filter_count=24, block_count=2, output_subsampling=4),
    'LSTM_BC_2_BLC_3_BFC_24_LSTM_N': partial(
        build_deep_net_with_multiple_LSTM, num_hidden=256, num_recurrent_layers=1,
        block_layer_count=3, base_filter_count=24, block_count=2, output_subsampling=4),
    'LSTM_BC_4_BLC_2_BFC_24': partial(
        build_deep_net, num_hidden=256, num_recurrent_layers=1,
        block_layer_count=2, base_filter_count=24, block_count=4, output_subsampling=4),
    'LSTM_BC_3_BLC_2_BFC_36': partial(
        build_deep_net, num_hidden=256, num_recurrent_layers=1,
        block_layer_count=2, base_filter_count=36, block_count=3, output_subsampling=4),
    'LSTM_BC_2_BLC_3_BFC_36': partial(
        build_deep_net, num_hidden=256, num_recurrent_layers=1,
        block_layer_count=3, base_filter_count=36, block_count=2, output_subsampling=4),
    'LSTM_BC_4_BLC_4_BFC_64': partial(
        build_deep_net, num_hidden=1024, num_recurrent_layers=1,
        block_layer_count=4, base_filter_count=64, block_count=4, output_subsampling=4),
    'LSTM_BC_4_BLC_2_BFC_36': partial(
        build_deep_net, num_hidden=256, num_recurrent_layers=1,
        block_layer_count=2, base_filter_count=36, block_count=4, output_subsampling=4),
    'NET_SIMPLE_BC_2_BLC_1_BFC_6': partial(
        build_simple_net, num_hidden=96, block_layer_count=1, base_filter_count=6,
        block_count=2, output_subsampling=4),
    'NET_SIMPLE_BC_3_BLC_1_BFC_12': partial(
        build_simple_net, num_hidden=96, block_layer_count=1, base_filter_count=12,
        block_count=3, output_subsampling=4),
    'NET_SIMPLE_BC_3_BLC_1_BFC_24': partial(
        build_simple_net, num_hidden=96, block_layer_count=1, base_filter_count=24,
        block_count=3, output_subsampling=4),
    'NET_SIMPLE_BC_2_BLC_2_BFC_24': partial(
        build_simple_net, num_hidden=96, block_layer_count=2, base_filter_count=24,
        block_count=2, output_subsampling=4),
    'NET_SIMPLE_BC_3_BLC_1_BFC_48': partial(
        build_simple_net, num_hidden=96, block_layer_count=1, base_filter_count=48,
        block_count=3, output_subsampling=4),
    'NET_SIMPLE_BC_3_BLC_2_BFC_12': partial(
        build_simple_net, num_hidden=96, block_layer_count=2, base_filter_count=12,
        block_count=3, output_subsampling=4),
    'NET_SIMPLE_BC_2_BLC_2_BFC_48': partial(
        build_simple_net, num_hidden=96, block_layer_count=2, base_filter_count=48,
        block_count=2, output_subsampling=4),
    'NET_U_2DB_3_1DB_2_BFC_12': partial(build_u_net, num_recurrent_layers=2, block_layer_count=2, base_filter_count=12,
                                        block_count_2d=3, block_count_1d=2, output_subsampling=4, layer_norm=False),
    'NET_U_2DB_3_1DB_2_BFC_16': partial(build_u_net, num_recurrent_layers=2, block_layer_count=2, base_filter_count=16,
                                        block_count_2d=3, block_count_1d=2, output_subsampling=4, layer_norm=False)
}


def build_train_net(data_shape, class_count, net_builder, loss="ctc", manipulator=None, logits_regularization=0):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    variable_summaries(learning_rate)
    train_phase = tf.placeholder(tf.bool, name='train_phase')
    input_data = tf.placeholder(
        tf.uint8, shape=data_shape, name='input_data')
    if loss == "sce":
        targets = tf.placeholder(tf.int32, name='TARGETS')
        str_targets = tf.sparse_placeholder(tf.int32, name='str_targets')
    elif loss == "ctc":
        targets = tf.sparse_placeholder(tf.int32, name='TARGETS')

    seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
    if manipulator is None:
        net = tf.cast(input_data, tf.float32)
    else:
        net = manipulator().transform(input_data, train_phase=train_phase)

    transformed_data = net
    decoded, logits, logits_t, log_prob = net_builder(class_count, net, train_phase, keep_prob, seq_len)

    # Inaccuracy: label error rate
    if loss == "sce":
        char_preds = tf.argmax(logits, axis=2, output_type=tf.int32)
        ler = tf.reduce_mean(tf.cast(tf.not_equal(char_preds, targets), tf.float32), name='label_error_rate')
        ler_chars = tf.reduce_mean(
            tf.edit_distance(tf.cast(decoded[0], tf.int32), str_targets),
            name='str_label_error_rate'
        )

        weights = tf.cast(tf.not_equal(targets, class_count-1), tf.float32)

        trn_loss = tf.losses.sparse_softmax_cross_entropy(labels=targets, logits=logits, weights=weights)
        # trn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits, name="sparse_softmax_cross_entropy")
    elif loss == "ctc":
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets), name='label_error_rate')
        trn_loss = tf.nn.ctc_loss(targets, logits_t, seq_len, ctc_merge_repeated=True)
    logits_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(logits, 2), 1))
    trn_loss = tf.reduce_sum([tf.reduce_mean(trn_loss, name='trn_loss'), tf.multiply(logits_loss, logits_regularization)])

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(trn_loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=0.33)

    if loss == "sce":
        return (init, saver, keep_prob, learning_rate, train_phase, input_data, transformed_data, targets, seq_len,
                logits, decoded, log_prob, trn_loss, ler, optimizer, str_targets, ler_chars)
    elif loss == "ctc":
        return (init, saver, keep_prob, learning_rate, train_phase, input_data, transformed_data, targets, seq_len,
                logits, decoded, log_prob, trn_loss, ler, optimizer)


def build_train_net_with_multiple_last_layer(data_shape, class_count, net_builder, number_of_last_layers,
                                             manipulator=None, train_only_last_layers=False, train_last_n=None, train_first_n=None):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    variable_summaries(learning_rate)
    train_phase = tf.placeholder(tf.bool, name='train_phase')
    input_data = tf.placeholder(tf.uint8, shape=data_shape, name='input_data')

    targets_all = []
    for last_layer_index in range(number_of_last_layers):
        targets_all.append(tf.sparse_placeholder(tf.int32, name='TARGETS_{}'.format(last_layer_index)))

    seq_len_all = []
    for last_layer_index in range(number_of_last_layers):
        seq_len_all.append(tf.placeholder(tf.int32, [None], name='seq_len_{}'.format(last_layer_index)))

    if manipulator is None:
        net = tf.cast(input_data, tf.float32)
    else:
        net = manipulator().transform(input_data, train_phase=train_phase)

    transformed_data = net
    decoded_all, logits_all, logits_t_all, log_prob_all, mask_all = net_builder(class_count, net, train_phase, keep_prob, seq_len_all, number_of_last_layers)

    ler_all = []
    for decoded, targets in zip(decoded_all, targets_all):
        ler_all.append(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    trn_loss_all = []
    trn_loss_reduce_all = []
    print(len(targets_all), len(logits_t_all), len(seq_len_all))
    for i, (targets, logits_t, seq_len) in enumerate(zip(targets_all, logits_t_all, seq_len_all)):
        trn_loss_all.append(tf.nn.ctc_loss(targets, logits_t, seq_len, ctc_merge_repeated=True))
        trn_loss_reduce_all.append(tf.reduce_mean(trn_loss_all[-1], name='trn_loss'))

    trn_loss = tf.add_n(trn_loss_reduce_all)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        layers_to_train = []
        if train_only_last_layers or train_last_n is not None or train_first_n is not None:
            layer_names = []
            for v in tf.trainable_variables():
                layer_name = v.name.split('/')[0]
                if layer_name not in layer_names and "batch" not in layer_name:
                    layer_names.append(layer_name)
            print("LAYER NAMES: {}".format(layer_names))
        if train_last_n is not None:
            layers_to_train += layer_names[-train_last_n-number_of_last_layers:-number_of_last_layers]
        if train_first_n is not None:
            layers_to_train += layer_names[:train_first_n]
        if train_only_last_layers or train_last_n is not None or train_first_n is not None:
            layers_to_train += layer_names[-number_of_last_layers:]
            variables_to_train = []
            for layer in layers_to_train:
                variables_to_train += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer)
            print("LAYERS TO TRAIN: {}".format(layers_to_train))
            print("VARIABLES TO TRAIN: {}".format(variables_to_train))
            grads_and_vars = tf.train.AdamOptimizer(learning_rate).compute_gradients(trn_loss, var_list=variables_to_train)
            optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads_and_vars)
        else:
            grads_and_vars = tf.train.AdamOptimizer(learning_rate).compute_gradients(trn_loss)
            optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads_and_vars)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=None)

    return (init, saver, keep_prob, learning_rate, train_phase, input_data,
            transformed_data, targets_all, seq_len_all, logits_all, logits_t_all,
            decoded_all, log_prob_all, trn_loss_all, trn_loss, ler_all, mask_all,
            optimizer, grads_and_vars)


def build_train_net_with_multiple_LSTM(data_shape, class_count, net_builder, number_of_last_layers,
                                       manipulator=None, train_only_last_layers=False, train_last_n=None, train_first_n=None):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    variable_summaries(learning_rate)
    train_phase = tf.placeholder(tf.bool, name='train_phase')
    input_data = tf.placeholder(tf.uint8, shape=data_shape, name='input_data')

    targets_all = []
    for last_layer_index in range(number_of_last_layers):
        targets_all.append(tf.sparse_placeholder(tf.int32, name='TARGETS_{}'.format(last_layer_index)))

    seq_len_all = []
    for last_layer_index in range(number_of_last_layers):
        seq_len_all.append(tf.placeholder(tf.int32, [None], name='seq_len_{}'.format(last_layer_index)))

    if manipulator is None:
        net = tf.cast(input_data, tf.float32)
    else:
        net = manipulator().transform(input_data, train_phase=train_phase)

    transformed_data = net
    decoded_all, logits_all, logits_t_all, log_prob_all, mask_all = net_builder(class_count, net, train_phase, keep_prob, seq_len_all, number_of_last_layers)

    ler_all = []
    for decoded, targets in zip(decoded_all, targets_all):
        ler_all.append(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    trn_loss_all = []
    trn_loss_reduce_all = []
    print(len(targets_all), len(logits_t_all), len(seq_len_all))
    for i, (targets, logits_t, seq_len) in enumerate(zip(targets_all, logits_t_all, seq_len_all)):
        trn_loss_all.append(tf.nn.ctc_loss(targets, logits_t, seq_len, ctc_merge_repeated=True))
        trn_loss_reduce_all.append(tf.reduce_mean(trn_loss_all[-1], name='trn_loss'))

    trn_loss = tf.add_n(trn_loss_reduce_all)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        layers_to_train = []
        if train_only_last_layers or train_last_n is not None or train_first_n is not None:
            layer_names = []
            for v in tf.trainable_variables():
                layer_name = v.name.split('/')[0]
                if layer_name not in layer_names and "batch" not in layer_name:
                    print(layer_name)
                    layer_names.append(layer_name)
                if layer_name not in layer_names and "batch" in layer_name:
                    print(layer_name)
            print("LAYER NAMES: {}".format(layer_names))
        if train_last_n is not None:
            layers_to_train += layer_names[-train_last_n-number_of_last_layers * 2:-number_of_last_layers * 2]
        if train_first_n is not None:
            layers_to_train += layer_names[:train_first_n]
        if train_only_last_layers or train_last_n is not None or train_first_n is not None:
            layers_to_train += layer_names[-number_of_last_layers * 2:]
            variables_to_train = []
            for layer in layers_to_train:
                variables_to_train += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer)
            print("LAYERS TO TRAIN: {}".format(layers_to_train))
            print("VARIABLES TO TRAIN: {}".format(variables_to_train))
            grads_and_vars = tf.train.AdamOptimizer(learning_rate).compute_gradients(trn_loss, var_list=variables_to_train)
            optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads_and_vars)
        else:
            grads_and_vars = tf.train.AdamOptimizer(learning_rate).compute_gradients(trn_loss)
            optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads_and_vars)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=None)

    return (init, saver, keep_prob, learning_rate, train_phase, input_data,
            transformed_data, targets_all, seq_len_all, logits_all, logits_t_all,
            decoded_all, log_prob_all, trn_loss_all, trn_loss, ler_all, mask_all,
            optimizer, grads_and_vars)


def build_eval_net(data_shape, class_count, net_builder, manipulator=None):
    keep_prob = 1
    train_phase = False
    input_data = tf.placeholder(
        tf.uint8, shape=data_shape, name='input_data')
    seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

    if manipulator is None:
        net = tf.cast(input_data, tf.float32) / 255
    else:
        net = manipulator().transform(input_data, tf.constant(False, dtype=tf.bool))

    transformed_data = net

    decoded, logits, logits_t, log_prob = net_builder(class_count, net, train_phase, keep_prob, seq_len)

    saver = tf.train.Saver()

    return (saver, input_data, transformed_data, seq_len, logits, logits_t, decoded, log_prob)
