class TransformerBaseParams(object):
    batch_size = 4096  # Maximum number of tokens per batch of examples.
    max_length = 256  # Maximum number of tokens per example.

    # Model params
    initializer_gain = 1.0  # Used in trainable variable initialization.
    vocab_size = 50000  # Number of tokens defined in the vocabulary file.
    hidden_size = 512  # Model dimension in the hidden layers.
    embedding_size = 512
    num_hidden_layers = 6  # Number of layers in the encoder and decoder stacks.
    num_heads = 8  # Number of heads to use in multi-headed attention.
    filter_size = 2048  # 2048  # Inner layer dimensionality in the feedforward network.
    source_vocab_size = vocab_size
    target_vocab_size = vocab_size
    shared_three_embedding = True
    shared_embedding_softmax_weights = False
    share_encoder_decoder = False
    share_encoder_decoder_softmax = False
    add_tag = False
    distill_from_teacher = False
    layer_postprocess_dropout = 0.1
    attention_dropout = 0.1
    relu_dropout = 0.1

    # Training params
    label_smoothing = 0.1
    learning_rate = 2.0
    learning_rate_decay_rate = 1.0
    learning_rate_warmup_steps = 4000

    # Optimizer params
    optimizer_adam_beta1 = 0.9
    optimizer_adam_beta2 = 0.98
    optimizer_adam_epsilon = 1e-09

    # Default prediction params
    extra_decode_length = 50
    beam_size = 4
    alpha = 0.6  # used to calculate length normalization in beam search
    return_beams = False    # return all beam outputs



class TransformerBigParams(TransformerBaseParams):
    """Parameters for the big Transformer model."""
    batch_size = 4096
    hidden_size = 1024
    filter_size = 4096
    embedding_size = 1024
    num_heads = 16
    num_hidden_layers = 6


