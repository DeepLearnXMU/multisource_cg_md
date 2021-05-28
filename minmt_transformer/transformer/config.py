from __future__ import unicode_literals
from transformer.model import model_params

PAD = "<pad>"
PAD_ID = 0
EOS = "<EOS>"
EOS_ID = 1
UNK= "UNK"
UNK_ID = 2 # will be the last token.
RESERVED_TOKENS = {
    PAD: PAD_ID,
    EOS: EOS_ID,
    UNK: UNK_ID
}

# shards "tag:total_shards"
SHARDS = {
    "train": 100,
    "dev": 1
}

PARAMS_MAP = {
    "base": model_params.TransformerBaseParams,
    "big": model_params.TransformerBigParams,
}
