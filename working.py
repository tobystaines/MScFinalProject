import Audio_functions as af

path = 'C:/Users/Toby/MSc_Project/Test_Audio/GANdatasetsMini/train_sup/Mixed/F01_22GC010A_BUS.CH1.wav'

test_audio = af.read_audio(path, 44100)
spec = af.compute_spectrogram(test_audio, 1024, 256)
