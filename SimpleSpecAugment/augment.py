import numpy as np
import random

def spec_augment(spectrogram, augment_times = 1, masks=2, freq_masking=0.15, time_masking=0.15):
        """
        Implementation of SpecAugment using numpy.

        Args:
            spectrogram (numpy.ndarray): Input 2D spectrogram with shape (freq, time).
            augment_times (int): Number of times to augment original spectrogram.
            masks (int): Number of masks for frequency and time masking.
            freq_masking (float, optional): Maximum frequency masking length. Defaults to 0.15.
            time_masking (float, optional): Maximum time masking length. Defaults to 0.15.
            
        Returns:
            augmented_spectrograms (numpy.ndarray): List of augmented spectrograms.
        """
    
        augmented_spectrograms = []
        original_spectrogram = spectrogram.copy()
        masks = max(1, masks) #mask cannot be 0 or negative
        
        for i in range(augment_times):
            augmented = original_spectrogram.copy()
            
            for i in range(masks):
                
                 #frequency masking
                 freqs, time_frames  = augmented.shape
                 freq_mask_percentage = random.uniform(0.0, freq_masking)
                 masked_freqs = int(freq_mask_percentage * freqs)
                 f0 = int(np.random.uniform(low=0.0, high=freqs - masked_freqs))
                 augmented[f0:f0 + masked_freqs, :] = spectrogram.min()
                 
                 #time masking
                 time_frames_mask_percentage = random.uniform(0.0, time_masking)
                 masked_time_frames = int(time_frames_mask_percentage * time_frames)
                 t0 = int(np.random.uniform(low=0.0, high=time_frames - masked_time_frames))
                 augmented[:, t0:t0 + masked_time_frames] = spectrogram.min()
                 
            augmented_spectrograms.append(augmented)

        return augmented_spectrograms