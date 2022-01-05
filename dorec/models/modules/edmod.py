#!/usr/bin/env python

import os.path as osp

from dorec.core.utils import save_model

from ..base import ModuleBase
from ..heads import HeadBase


class EDModule(ModuleBase):
    """Encoder and Decoder module

    Args:
        encoder (dorec.models.ModuleBase)
        decoder (dorec.models.HeadBase)
    """

    def __init__(self, encoder, decoder):
        super(EDModule, self).__init__()
        assert isinstance(encoder, ModuleBase)
        assert isinstance(decoder, HeadBase)

        self.encoder = encoder
        self.decoder = decoder

        self.enc_name = encoder.name
        self.dec_name = decoder.name

        self.deep_sup = decoder.deep_sup
        self._name = self.enc_name + "+" + self.dec_name

    @property
    def name(self):
        return self._name

    def save_models_separately(self, save_dir, prefix=None):
        """Save models' weight separately
        Args:
            save_dir (str)
            prefix (str, optional)
        """
        enc_filename = self.enc_name
        dec_filename = self.dec_name

        if prefix is not None:
            enc_filename += ("_" + str(prefix))
            dec_filename += ("_" + str(prefix))

        enc_filename = osp.join(save_dir, enc_filename + ".pth")
        dec_filename = osp.join(save_dir, dec_filename + ".pth")

        save_model(self.encoder, enc_filename)
        save_model(self.decoder, dec_filename)

    def forward(self, inputs):
        segSize = inputs.shape[2:]

        x = self.encoder(inputs)
        x = self.decoder(x, segSize)

        return x
