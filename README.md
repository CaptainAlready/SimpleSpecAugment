# SimpleSpecAugment

Short and simple implementation of SpecAugment using numpy and functional programming.

**Original SpecAugment Paper** : https://doi.org/10.48550/arXiv.1904.08779

**Example of usage**

Librosa is required to run this example :
`pip install Librosa`

```python
import numpy as np
import librosa
from SimpleSpecAugment.augment import spec_augment

signal, sr = librosa.load(librosa.example('pistachio'), sr=16000)
spectrogram = librosa.feature.melspectrogram(y = signal, sr=sr, n_mels=128, fmax=8000)

#augment is a list containing the augmented spectrograms
augment = spec_augment(spectrogram, augment_times = 3, masks = 1) 
```


![augmented](https://github.com/CaptainAlready/SimpleSpecAugment/assets/58816142/5b7d5e81-ee6f-41e9-84a0-7d257f7235ba)
