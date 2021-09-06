#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib

"""
 'Router'-like set of methods for component initialization with lazy imports 
"""

def init_hf_lxmert_bert_biencoder(args, **kwargs):
#     if importlib.util.find_spec("transformers") is None:
#         raise RuntimeError('Please install transformers lib')
    from .hf_models import get_lxmert_bert_biencoder_components
    
    return get_lxmert_bert_biencoder_components(args, **kwargs)

BIENCODER_INITIALIZERS = {
    'hf_lxmert_bert': init_hf_lxmert_bert_biencoder
}


TENSORIZER_INITIALIZERS = {
    'hf_lxmert_bert': init_hf_lxmert_bert_biencoder,
}

# READER_INITIALIZERS = {
#     'hf_bert': init_hf_bert_reader,
# }

def init_comp(initializers_dict, type, args, **kwargs):
    if type in initializers_dict:
        return initializers_dict[type](args, **kwargs)
    else:
        raise RuntimeError('unsupported model type: {}'.format(type))


def init_biencoder_components(encoder_type: str, args, **kwargs):
    
    return init_comp(BIENCODER_INITIALIZERS, encoder_type, args, **kwargs)


# def init_reader_components(encoder_type: str, args, **kwargs):
#     return init_comp(READER_INITIALIZERS, encoder_type, args, **kwargs)


def init_tenzorizer(encoder_type: str, args, **kwargs):
    return init_comp(TENSORIZER_INITIALIZERS, encoder_type, args, **kwargs)
