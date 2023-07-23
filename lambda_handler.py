import logging
import boto3
import json
import numpy as np
import tempfile

logger = logging.getLogger()
logger.setLevel(logging.INFO)
region = 'us-east-1'
relevant_timestamps = []

import mxnet as mx

from mxnet import gluon

def read_s3_file(bucketname, filename):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucketname)
    obj = bucket.Object(filename)
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'wb') as f:
        obj.download_fileobj(f)
    return tmp.name


def load_model(s_fname, p_fname):
    """
    Load model checkpoint from file.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    symbol = mx.symbol.load(s_fname)
    save_dict = mx.nd.load(p_fname)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return symbol, arg_params, aux_params


# sym, arg_params, aux_params = load_model('model2.json', 'model2.params')
sym_file = read_s3_file('reeltimes3', 'model/Inception-7-symbol.json')
params_file = read_s3_file('reeltimes3', 'model/Inception-7-0001.params')
sym, arg_params, aux_params = gluon.nn.SymbolBlock.imports(sym_file, ['data'], params_file)


# load json and params into model
# mod = None

# We bind the module with the input shape and specify that it is only for predicting.
# The number 1 added before the image shape (3x224x224) means that we will only predict one image at a time.

# FULL MODEL
# mod = mx.mod.Module(symbol=sym, label_names=None)
# mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
# mod.set_params(arg_params, aux_params, allow_missing=True)


from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


def lambda_handler(event, context):
    # PARTIAL MODEL
    mod2 = None
    all_layers = sym.get_internals()
    print(all_layers.list_outputs()[-10:])
    sym2 = all_layers['global_pool_output']
    mod2 = mx.mod.Module(symbol=sym2, label_names=None)
    # mod2.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod2._label_shapes)
    mod2.bind(for_training=False, data_shapes=[('data', (1,3,299,299))])
    mod2.set_params(arg_params, aux_params)

    # Get image(s) from s3
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(event['reeltimes3'])
    object = bucket.Object(event['filename'])

    # img = mx.image.imread('image.jpg')

    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'wb') as f:
        object.download_fileobj(f)
        img=mx.image.imread(tmp.name)
        # convert into format (batch, RGB, width, height)
        img = mx.image.imresize(img, 299, 299) # resize
        img = img.transpose((2, 0, 1)) # Channel first
        img = img.expand_dims(axis=0) # batchify

        mod2.forward(Batch([img]))
    out = np.squeeze(mod2.get_outputs()[0].asnumpy())

    kinesis_client = boto3.client('kinesis')
    put_response = kinesis_client.put_record(StreamName='bottleneck_stream',
                                            Data=json.dumps({'filename': event['filename'], 'features': out.tolist()}),
                                            PartitionKey="partitionkey")
    return 'Wrote features to kinesis stream'
