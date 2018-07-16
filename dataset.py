import os, re, subprocess
from soundfile import SoundFile
import glob

_find_sampling_rate = re.compile('.* ([0-9:]+) Hz,', re.MULTILINE)
_find_channels = re.compile(".*Hz,( .*?),", re.MULTILINE)
_find_duration = re.compile('.*Duration: ([0-9:]+)', re.MULTILINE)


def timestamp_to_seconds( ms ):
    """Convert a hours:minutes:seconds string representation to the appropriate time in seconds."""
    a = ms.split(':')
    assert 3 == len(a)
    return float(a[0]) * 3600 + float(a[1]) * 60 + float(a[2])


def seconds_to_min_sec( secs ):
    """Return a minutes:seconds string representation of the given number of seconds."""
    mins = int(secs) / 60
    secs = int(secs - (mins * 60))
    return "%d:%02d" % (mins, secs)


def get_mp3_metadata(audio_path):
    """Determine length of tracks listed in the given input files (e.g. playlists)."""
    ffmpeg = subprocess.check_output(
      'ffmpeg -i "%s"; exit 0' % audio_path,
      shell = True,
      stderr = subprocess.STDOUT )

    # Get sampling rate
    match = _find_sampling_rate.search( ffmpeg )
    assert(match)
    sampling_rate = int(match.group( 1 ))

    # Get channels
    match = _find_channels.search( ffmpeg )
    assert(match)
    channels = match.group( 1 )
    channels = (2 if channels.__contains__("stereo") else 1)

    # Get duration
    match = _find_duration.search( ffmpeg )
    assert(match)
    duration = match.group( 1 )
    duration = timestamp_to_seconds(duration)

    return sampling_rate, channels, duration


def get_audio_metadata(audioPath, sphereType=False):
    """
    Returns sampling rate, number of channels and duration of an audio file
    :param audioPath:
    :param sphereType:
    :return:
    """

    snd_file = SoundFile(audioPath, mode='r')
    inf = snd_file._info
    sr = inf.samplerate
    channels = inf.channels
    duration = float(inf.frames) / float(inf.samplerate)
    return int(sr), int(channels), float(duration)


class Sample(object):
    '''
    Represents a particular audio track - maintains metadata about the audio file for faster audio handling during training
    '''
    def __init__(self, path, sample_rate, channels, duration):
        self.path = path
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = duration

    @classmethod
    def from_path(cls, path):
        '''
        Create new sample object from audio file path by retrieving metadata.
        :param path:
        :return:
        '''

        sr, channels, duration = get_audio_metadata(path)
        return cls(path, sr, channels, duration)


def getCHiME3(dataset):

    root = 'C:/Users/Toby/Jupyter Notebooks/My Work/MSc Project/Test Audio/GANdatasetsMini/'
    if dataset == 'train_unsup':
        mix_list = glob.glob(root+dataset+'/*.wav')
        voice_list = list()
    else:
        mix_list = glob.glob(root+dataset+'/Mixed/*.wav')
        voice_list = glob.glob(root+dataset+'/Voice/*.wav')
    mix = list()
    voice = list()
    for item in mix_list:
        mix.append(Sample.from_path(item))
    for item in voice_list:
        voice.append(Sample.from_path(item))
    return mix, voice
