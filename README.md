# Voice-differentiation
This AI tries to differentiate between different voices

This is one of my first attempts of writing a neural network.
I collected 1 minute audio samples of a few of my friends and converted them into sprectrogram images.
I split these images into smaller ones so the CNN (Convolutional Neural Network) can analyse small snippets of the images. I thought this might be better, because the images should still contain the characteristics of the spectrogram of each person.
After training it on the images for a while, I tested it on new audio samples.

It worked, but not as good as I hoped for.
It was right about who spoke in the sample about 40-50% of the time.
Which satisfied me enough considering it's better than 25% (25% because I trained it with samples of 4 friends: 100%/4 = 25%).
