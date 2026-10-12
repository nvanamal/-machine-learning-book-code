# ENCODER-DECODER / SEQ2SEQ | EXAMPLE C
# Teacher forcing shifts targets by one position
# Dr. Nagendrababu Vanamala

import numpy as np
BOS, EOS = 1, 2
target = np.array([7, 8, EOS])
decoder_input = np.r_[BOS, target[:-1]]
print("Decoder input:", decoder_input)
print("Expected next tokens:", target)
# During inference, previous predictions replace true prior tokens.
