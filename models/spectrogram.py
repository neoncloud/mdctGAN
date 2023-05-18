"""
Module to transform signals
from https://github.com/nils-werner/stft/

"""
from __future__ import division, absolute_import
import itertools
import torch
import torch.fft
import torch.nn.functional
import numpy as np

def _pad(data, frame_length):
    return torch.nn.functional.pad(
        data,
        pad=(
            0,
            int(
                np.ceil(
                    len(data) / frame_length
                ) * frame_length - len(data)
            )
        ),
        mode='constant',
        value=0
    )

def unpad(data, outlength):
    slicetuple = [slice(None)] * data.ndim
    slicetuple[0] = slice(None, outlength)
    return data[tuple(slicetuple)]


def center_pad(data, frame_length):
    #padtuple = [(0, 0)] * data.ndim
    padtuple = (frame_length // 2, frame_length // 2)
    #print(padtuple)
    return torch.nn.functional.pad(
        data,
        pad = padtuple,
        mode = 'constant',
        value = 0
    )

def center_unpad(data, frame_length):
    slicetuple = [slice(None)] * data.ndim
    slicetuple[1] = slice(frame_length // 2, -frame_length // 2)
    return data[tuple(slicetuple)]

def process(
    data,
    window_function,
    halved,
    transform,
    padding=0,
    n_fft=1024
):
    """Calculate a windowed transform of a signal

    Parameters
    ----------
    data : array_like
        The signal to be calculated. Must be a 1D array.
    window : array_like
        Tapering window
    halved : boolean
        Switch for turning on signal truncation. For real signals, the fourier
        transform of real signals returns a symmetrically mirrored spectrum.
        This additional data is not needed and can be removed.
    transform : callable
        The transform to be used.
    padding : int
        Zero-pad signal with x times the number of samples.

    Returns
    -------
    data : array_like
        The spectrum

    """

    data = data * window_function

    if padding > 0:
        data = torch.nn.functional.pad(
            data,
            pad=(
                0,
                len(data) * padding
            ),
            mode='constant',
            value=0
        )

    result = transform(data,n_fft)

    if(halved):
        result = result[0:result.size // 2 + 1]

    return result


def iprocess(
    data,
    window_function,
    halved,
    transform,
    padding=0,
    n_fft=1024
):
    """Calculate the inverse short time fourier transform of a spectrum

    Parameters
    ----------
    data : array_like
        The spectrum to be calculated. Must be a 1D array.
    window : array_like
        Tapering window
    halved : boolean
        Switch for turning on signal truncation. For real output signals, the
        inverse fourier transform consumes a symmetrically mirrored spectrum.
        This additional data is not needed and can be removed. Setting this
        value to :code:`True` will automatically create a mirrored spectrum.
    transform : callable
        The transform to be used.
    padding : int
        Signal before FFT transform was padded with x zeros.


    Returns
    -------
    data : array_like
        The signal

    """
    if halved:
        data = torch.nn.functional.pad(data, (0, data.shape[0] - 2), 'reflect')
        start = data.shape[0] // 2 + 1
        data[start:] = data[start:].conjugate()

    output = transform(data,n_fft)
    if torch.is_complex(output):
        output = torch.real(output)

    if padding > 0:
        output = output[0:-(len(data) * padding // (padding + 1))]

    return output * window_function


def spectrogram(
    data,
    frame_length=1024,
    step_length=None,
    overlap=None,
    centered=True,
    n_fft=1024,
    window_function=None,
    halved=True,
    transform=None,
    padding=0
):
    """Calculate the spectrogram of a signal

    Parameters
    ----------
    data : array_like
        The signal to be transformed. May be a 1D vector for single channel or
        a 2D matrix for multi channel data. In case of a mono signal, the data
        is must be a 1D vector of length :code:`samples`. In case of a multi
        channel signal, the data must be in the shape of :code:`samples x
        channels`.
    frame_length : int
        The signal frame length. Defaults to :code:`1024`.
    step_length : int
        The signal frame step_length. Defaults to :code:`None`. Setting this
        value will override :code:`overlap`.
    overlap : int
        The signal frame overlap coefficient. Value :code:`x` means
        :code:`1/x` overlap. Defaults to :code:`2`.
    centered : boolean
        Pad input signal so that the first and last window are centered around
        the beginning of the signal. Defaults to true.
    window : callable, array_like
        Window to be used for deringing. Can be :code:`False` to disable
        windowing. Defaults to :code:`scipy.signal.cosine`.
    halved : boolean
        Switch for turning on signal truncation. For real signals, the fourier
        transform of real signals returns a symmetrically mirrored spectrum.
        This additional data is not needed and can be removed. Defaults to
        :code:`True`.
    transform : callable
        The transform to be used. Defaults to :code:`scipy.fft.fft`.
    padding : int
        Zero-pad signal with x times the number of samples.
    save_settings : boolean
        Save settings used here in attribute :code:`out.stft_settings` so that
        :func:`ispectrogram` can infer these settings without the developer
        having to pass them again.

    Returns
    -------
    data : array_like
        The spectrogram (or tensor of spectograms) In case of a mono signal,
        the data is formatted as :code:`bins x frames`. In case of a multi
        channel signal, the data is formatted as :code:`bins x frames x
        channels`.

    Notes
    -----
    The data will be padded to be a multiple of the desired FFT length.

    See Also
    --------
    stft.stft.process : The function used to transform the data

    """

    if overlap is None:
        overlap = 2

    if step_length is None:
        step_length = frame_length // overlap

    if halved and torch.any(torch.iscomplex(data)):
        raise ValueError("You cannot treat a complex input signal as real "
                         "valued. Please set keyword argument halved=False.")

    data = torch.squeeze(data)

    if transform is None:
        transform = torch.fft.fft

    if not isinstance(transform, (list, tuple)):
        transform = [transform]

    transforms = itertools.cycle(transform)

    if centered:
        #print("center pad")
        data = center_pad(data, frame_length)

    if window_function is None:
        window_array = torch.ones(frame_length)

    if callable(window_function):
        window_array = window_function(frame_length)
    else:
        window_array = window_function
        frame_length = len(window_array)

    def traf(data):
        # Pad input signal so it fits into frame_length spec
        #print(data.size())
        #data = _pad(data, frame_length)
        #print(data.size())

        values = list(enumerate(
            range(0, len(data) - frame_length + step_length, step_length)
        ))
        #print(values)
        for j, i in values:
            sig = process(
                data[i:i + frame_length],
                window_function=window_array,
                halved=halved,
                transform=next(transforms),
                padding=padding,
                n_fft = n_fft
            ) / (frame_length // step_length // 2)
            if(i == 0):
                output_ = torch.zeros(
                    (sig.shape[0], len(values)), dtype=sig.dtype
                )

            output_[:, j] = sig

        return output_

    if data.ndim > 2:
        raise ValueError("spectrogram: Only 1D or 2D input data allowed")
    if data.ndim == 1:
        out = traf(data)
    elif data.ndim == 2:
        #print(data.size())
        for i in range(data.shape[0]):
            tmp = traf(data[i,:])
            #print(tmp.size())
            if i == 0:
                out = torch.empty(
                    ((data.shape[0],)+tmp.shape), dtype=tmp.dtype
                )
            out[i, :, :] = tmp
    return out


def ispectrogram(
    data,
    frame_length=1024,
    step_length=None,
    overlap=None,
    centered=True,
    n_fft=1024,
    window_function=None,
    halved=True,
    transform=None,
    padding=0,
    out_length=None
):
    """Calculate the inverse spectrogram of a signal

    Parameters
    ----------
    data : array_like
        The spectrogram to be inverted. May be a 2D matrix for single channel
        or a 3D tensor for multi channel data. In case of a mono signal, the
        data must be in the shape of :code:`bins x frames`. In case of a multi
        channel signal, the data must be in the shape of :code:`bins x frames x
        channels`.
    frame_length : int
        The signal frame length. Defaults to infer from data.
    step_length : int
        The signal frame step_length. Defaults to infer from data. Setting this
        value will override :code:`overlap`.
    overlap : int
        The signal frame overlap coefficient. Value :code:`x` means
        :code:`1/x` overlap. Defaults to infer from data.
    centered : boolean
        Pad input signal so that the first and last window are centered around
        the beginning of the signal. Defaults to to infer from data.
    window : callable, array_like
        Window to be used for deringing. Can be :code:`False` to disable
        windowing. Defaults to to infer from data.
    halved : boolean
        Switch to reconstruct the other halve of the spectrum if the forward
        transform has been truncated. Defaults to to infer from data.
    transform : callable
        The transform to be used. Defaults to infer from data.
    padding : int
        Zero-pad signal with x times the number of samples. Defaults to infer
        from data.
    outlength : int
        Crop output signal to length. Useful when input length of spectrogram
        did not fit into frame_length and input data had to be padded. Not
        setting this value will disable cropping, the output data may be
        longer than expected.

    Returns
    -------
    data : array_like
        The signal (or matrix of signals). In case of a mono output signal, the
        data is formatted as a 1D vector of length :code:`samples`. In case of
        a multi channel output signal, the data is formatted as :code:`samples
        x channels`.

    Notes
    -----
    By default :func:`spectrogram` saves its transformation parameters in
    the output array. This data is used to infer the transform parameters
    here. Any aspect of the settings can be overridden by passing the according
    parameter to this function.

    During transform the data will be padded to be a multiple of the desired
    FFT length. Hence, the result of the inverse transform might be longer
    than the input signal. However it is safe to remove the additional data,
    e.g. by using

    .. code:: python

        output.resize(input.shape)

    where :code:`input` is the input of :code:`stft.spectrogram()` and
    :code:`output` is the output of :code:`stft.ispectrogram()`

    See Also
    --------
    stft.stft.iprocess : The function used to transform the data

    """

    if overlap is None:
        overlap = 2

    if step_length is None:
        step_length = frame_length // overlap

    if window_function is None:
        window_array = torch.ones(frame_length)

    if callable(window_function):
        window_array = window_function(frame_length)
    else:
        window_array = window_function

    if transform is None:
        transform = torch.fft.ifft

    if not isinstance(transform, (list, tuple)):
        transform = [transform]
    
    transforms = itertools.cycle(transform)

    def traf(data):
        i = 0
        values = range(0, data.shape[1])
        for j in values:
            sig = iprocess(
                data[:, j],
                window_function=window_array,
                halved=halved,
                transform=next(transforms),
                padding=padding,
                n_fft=n_fft
            )

            if(i == 0):
                output = torch.zeros(
                    frame_length + (len(values) - 1) * step_length,
                    dtype=sig.dtype
                ).cuda()

            output[i:i + frame_length] += sig

            i += step_length

        return output

    if data.ndim == 2:
        out = traf(data)
    elif data.ndim == 3:
        for i in range(data.shape[0]):
            tmp = traf(data[i ,:, :])

            if i == 0:
                out = torch.empty(
                    ((data.shape[0],)+tmp.shape), dtype=tmp.dtype
                )
            out[i,:] = tmp
    else:
        raise ValueError("ispectrogram: Only 2D or 3D input data allowed")

    if centered:
        print(out.size())
        out = center_unpad(out, frame_length)

    return unpad(out, out_length)