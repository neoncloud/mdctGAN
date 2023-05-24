import numpy as np
import torch
import torch.nn
import torch.fft
import torch.nn.functional
from . import spectrogram

import functools
#import debugpy
#debugpy.listen(("localhost", 5678))
class MDCT(torch.nn.Module):
    """
    Serial version of the MDCT.
    Adapted from https://github.com/nils-werner/mdct/
    I modified the original code from scipy+numpy to torch, and I made it into torch layers with GPU support.
    """
    def __init__(self, window_function, step_length=None, n_fft=2048, center=True, device='cpu') -> None:
        super().__init__()
        self.window_function = window_function
        self.step_length = step_length
        self.n_fft = n_fft
        self.center = center
        self.device = device

    def mdct(
        self,
        x,
        odd=True,
        center = True,
        **kwargs
    ):
        """ Calculate lapped MDCT of input signal
        Parameters
        ----------
        x : array_like
            The signal to be transformed. May be a 1D vector for single channel or
            a 2D matrix for multi channel data. In case of a mono signal, the data
            is must be a 1D vector of length :code:`samples`. In case of a multi
            channel signal, the data must be in the shape of :code:`samples x
            channels`.
        odd : boolean, optional
            Switch to oddly stacked transform. Defaults to :code:`True`.
        framelength : int
            The signal frame length. Defaults to :code:`2048`.
        hopsize : int
            The signal frame hopsize. Defaults to :code:`None`. Setting this
            value will override :code:`overlap`.
        overlap : int
            The signal frame overlap coefficient. Value :code:`x` means
            :code:`1/x` overlap. Defaults to :code:`2`. Note that anything but
            :code:`2` will result in a filterbank without perfect reconstruction.
        centered : boolean
            Pad input signal so that the first and last window are centered around
            the beginning of the signal. Defaults to :code:`True`.
            Disabling this will result in aliasing
            in the first and last half-frame.
        window : callable, array_like
            Window to be used for deringing. Can be :code:`False` to disable
            windowing. Defaults to :code:`scipy.signal.cosine`.
        transforms : module, optional
            Module reference to core transforms. Mostly used to replace
            fast with slow core transforms, for testing. Defaults to
            :mod:`mdct.fast`
        padding : int
            Zero-pad signal with x times the number of samples.
            Defaults to :code:`0`.
        save_settings : boolean
            Save settings used here in attribute :code:`out.stft_settings` so that
            :func:`ispectrogram` can infer these settings without the developer
            having to pass them again.
        Returns
        -------
        out : array_like
            The signal (or matrix of signals). In case of a mono output signal, the
            data is formatted as a 1D vector of length :code:`samples`. In case of
            a multi channel output signal, the data is formatted as :code:`samples
            x channels`.
        See Also
        --------
        mdct.fast.transforms.mdct : MDCT
        """
        def cmdct(x, odd=True):
            """ Calculate complex MDCT/MCLT of input signal
            Parameters
            ----------
            x : array_like
                The input signal
            odd : boolean, optional
                Switch to oddly stacked transform. Defaults to :code:`True`.
            Returns
            -------
            out : array_like
                The output signal
            """
            N = len(x) // 2
            n0 = (N + 1) / 2
            if odd:
                outlen = N
                pre_twiddle = torch.exp(-1j * np.pi * torch.arange(N * 2) / (N * 2)).to(self.device)
                offset = 0.5
            else:
                outlen = N + 1
                pre_twiddle = 1.0
                offset = 0.0

            post_twiddle = torch.exp(
                -1j * np.pi * n0 * (torch.arange(outlen) + offset) / N
            ).to(self.device)

            X = torch.fft.fft(x * pre_twiddle)[:outlen]

            if not odd:
                X[0] *= np.sqrt(0.5)
                X[-1] *= np.sqrt(0.5)

            return X * post_twiddle * np.sqrt(1 / N)

        def _mdct(x, odd=True):
            """ Calculate modified discrete cosine transform of input signal
            Parameters
            ----------
            X : array_like
                The input signal
            odd : boolean, optional
                Switch to oddly stacked transform. Defaults to :code:`True`.
            Returns
            -------
            out : array_like
                The output signal
            """
            return torch.real(cmdct(x, odd=odd)) * np.sqrt(2)

        def _mdst(x, odd=True):
            """ Calculate modified discrete sine transform of input signal
            Parameters
            ----------
            X : array_like
                The input signal
            odd : boolean, optional
                Switch to oddly stacked transform. Defaults to :code:`True`.
            Returns
            -------
            out : array_like
                The output signal
            """
            return -1 * torch.imag(cmdct(x, odd=odd)) * np.sqrt(2)

        frame_length = len(self.window_function)

        if not odd:
            return spectrogram.spectrogram(
                x,
                transform=[
                    functools.partial(_mdct, odd=False),
                    functools.partial(_mdst, odd=False),
                ],
                halved=False,
                frame_length = frame_length,
                **kwargs
            )
        else:
            return spectrogram.spectrogram(
                x,
                transform=_mdct,
                halved=False,
                frame_length = frame_length,
                **kwargs
            )

    def forward(self, x):
        x = self.mdct(x=x, window_function=self.window_function, step_length=self.step_length, n_fft=self.n_fft, center=self.center,padding = 0)
        return x

class IMDCT(torch.nn.Module):
    def __init__(self, window_function, step_length=None, device='cuda',n_fft=2048, out_length = 48000, center=True):
        super().__init__()
        self.window_function = window_function
        self.step_length = step_length
        self.n_fft = n_fft
        self.out_length = out_length
        self.device = device

    def imdct(
        self,
        X,
        odd=True,
        **kwargs
    ):
        """ Calculate lapped inverse MDCT of input signal
        Parameters
        ----------
        x : array_like
            The spectrogram to be inverted. May be a 2D matrix for single channel
            or a 3D tensor for multi channel data. In case of a mono signal, the
            data must be in the shape of :code:`bins x frames`. In case of a multi
            channel signal, the data must be in the shape of :code:`bins x frames x
            channels`.
        odd : boolean, optional
            Switch to oddly stacked transform. Defaults to :code:`True`.
        framelength : int
            The signal frame length. Defaults to infer from data.
        hopsize : int
            The signal frame hopsize. Defaults to infer from data. Setting this
            value will override :code:`overlap`.
        overlap : int
            The signal frame overlap coefficient. Value :code:`x` means
            :code:`1/x` overlap. Defaults to infer from data. Note that anything
            but :code:`2` will result in a filterbank without perfect
            reconstruction.
        centered : boolean
            Pad input signal so that the first and last window are centered around
            the beginning of the signal. Defaults to to infer from data.
            The first and last half-frame will have aliasing, so using
            centering during forward MDCT is recommended.
        window : callable, array_like
            Window to be used for deringing. Can be :code:`False` to disable
            windowing. Defaults to to infer from data.
        halved : boolean
            Switch to reconstruct the other halve of the spectrum if the forward
            transform has been truncated. Defaults to to infer from data.
        transforms : module, optional
            Module reference to core transforms. Mostly used to replace
            fast with slow core transforms, for testing. Defaults to
            :mod:`mdct.fast`
        padding : int
            Zero-pad signal with x times the number of samples. Defaults to infer
            from data.
        outlength : int
            Crop output signal to length. Useful when input length of spectrogram
            did not fit into framelength and input data had to be padded. Not
            setting this value will disable cropping, the output data may be
            longer than expected.
        Returns
        -------
        out : array_like
            The output signal
        See Also
        --------
        mdct.fast.transforms.imdct : inverse MDCT
        """
        def icmdct(X, odd=True):
            """ Calculate inverse complex MDCT/MCLT of input signal
            Parameters
            ----------
            X : array_like
                The input signal
            odd : boolean, optional
                Switch to oddly stacked transform. Defaults to :code:`True`.
            Returns
            -------
            out : array_like
                The output signal
            """
            if not odd and len(X) % 2 == 0:
                raise ValueError(
                    "Even inverse CMDCT requires an odd number "
                    "of coefficients"
                )

            if odd:
                N = len(X)
                n0 = (N + 1) / 2

                post_twiddle = torch.exp(
                    1j * np.pi * (torch.arange(N * 2) + n0) / (N * 2)
                ).to(self.device)

                Y = torch.zeros(N * 2, dtype=X.dtype)
                Y[:N] = X
                #Y[N:] = -1 * torch.conj(X[::-1])
                Y[N:] = -1 * torch.conj(X.flip(dims=(0,)))
            else:
                N = len(X) - 1
                n0 = (N + 1) / 2

                post_twiddle = 1.0

                X[0] *= torch.sqrt(2)
                X[-1] *= torch.sqrt(2)

                Y = torch.zeros(N * 2, dtype=X.dtype)
                Y[:N+1] = X
                #Y[N+1:] = -1 * torch.conj(X[-2:0:-1])
                Y[N+1:] = -1 * torch.conj(X[:-2].flip(dims=(0,)))

            pre_twiddle = (torch.exp(1j * np.pi * n0 * torch.arange(N * 2) / N)).to(self.device)

            y = torch.fft.ifft(Y.to(self.device) * pre_twiddle)

            return torch.real(y * post_twiddle) * np.sqrt(N)

        def _imdct(X, odd=True):
            """ Calculate inverse modified discrete cosine transform of input signal
            Parameters
            ----------
            X : array_like
                The input signal
            odd : boolean, optional
                Switch to oddly stacked transform. Defaults to :code:`True`.
            Returns
            -------
            out : array_like
                The output signal
            """
            return icmdct(X, odd=odd) * np.sqrt(2)

        def _imdst(X, odd=True):
            """ Calculate inverse modified discrete sine transform of input signal
            Parameters
            ----------
            X : array_like
                The input signal
            odd : boolean, optional
                Switch to oddly stacked transform. Defaults to :code:`True`.
            Returns
            -------
            out : array_like
                The output signal
            """
            return -1 * icmdct(X * 1j, odd=odd) * np.sqrt(2)

        frame_length = len(self.window_function)

        if not odd:
            return spectrogram.ispectrogram(
                X,
                transform=[
                    functools.partial(_imdct, odd=False),
                    functools.partial(_imdst, odd=False),
                ],
                halved=False,
                **kwargs
            )
        else:
            return spectrogram.ispectrogram(
                X,
                transform=_imdct,
                halved=False,
                frame_length = frame_length,
                **kwargs
            )
    def forward(self, X):
        X = self.imdct(X=X, window_function=self.window_function, step_length=self.step_length, n_fft=self.n_fft, padding = 0, out_length = self.out_length)
        return X


class MDCT4(torch.nn.Module):
    """
    The exact version of the MDCT, using modified DCT-IV.
    Borrowed from MATLAB implementation.
    """
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, window=None, center=True, pad_mode='constant', device='cuda') -> None:
        super().__init__()
        self.n_fft = n_fft
        self.pad_mode = pad_mode
        self.device = device
        self.hop_length = hop_length
        self.center = center

        # making window
        if window is None:
            window = torch.ones
        if callable(window):
            self.win_length = int(win_length)
            self.window = window(self.win_length).to(self.device)
        else:
            self.window = window.to(self.device)
            self.win_length = len(window)

        assert self.win_length <= self.n_fft, 'Window lenth %d should be no more than fft length %d'%(self.win_length, self.n_fft)
        assert self.hop_length <= self.win_length, 'You hopped more than one frame'

        self.exp1 = torch.exp(-1j*torch.pi/self.n_fft*torch.arange(start=0, end=self.n_fft,step=1, dtype=torch.float64, device=self.device))
        self.exp2 = torch.exp(-1j*(torch.pi/(2*self.n_fft)+torch.pi/4)*torch.arange(start=1, end=self.n_fft, step=2, dtype=torch.float64, device=self.device))

    def forward(self, signal, return_frames:bool=False):
        # Pad the signal to a proper length
        signal_len = int(len(signal))
        start_pad = 0
        # Pad the signal so that the t-th frame is centered at time t * hop_length. Otherwise, the t-th frame begins at time t * hop_length.
        if self.center:
            start_pad = self.hop_length
        additional_len = signal_len%self.hop_length
        end_pad = start_pad
        if additional_len:
            end_pad = start_pad + self.hop_length - additional_len
        signal = pad(signal, (start_pad,end_pad), mode=self.pad_mode)

        # Slice the signal with overlapping
        signal = signal.unfold(dimension=-1, size=self.win_length, step=self.hop_length)

        # Apply windows to each pieces
        signal = torch.mul(signal, self.window)
        if return_frames:
            frames = signal.clone()
        else:
            frames = torch.empty(1)

        # Pad zeros for DCT
        if self.n_fft > self.win_length:
            signal = pad(signal, (0, self.n_fft-self.win_length), mode='constant')

        signal = signal*self.exp1
        signal = torch.fft.fft(signal)[...,:self.n_fft//2]
        signal = torch.real(self.exp2*signal)

        return signal, frames


class IMDCT4(torch.nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, window=None, center=True, pad_mode='constant', out_length=None,device='cuda') -> None:
        super().__init__()
        self.n_fft = n_fft
        self.pad_mode = pad_mode
        self.device = device
        self.hop_length = hop_length
        self.center = center
        self.out_length = out_length

        # making window
        if window is None:
            window = torch.ones
        if callable(window):
            self.win_length = int(win_length)
            self.window = window(self.win_length).to(self.device)
        else:
            self.window = window.to(self.device)
            self.win_length = len(window)

        assert self.win_length <= self.n_fft, 'Window lenth %d should be no more than fft length %d'%(self.win_length, self.n_fft)
        assert self.hop_length <= self.win_length, 'You hopped more than one frame'

        self.exp1 = torch.exp(-1j*(torch.pi/(2*self.n_fft)+torch.pi/4)*torch.arange(start=1, end=self.n_fft, step=2, dtype=torch.float64, device=self.device))
        self.exp2 = torch.exp(-1j*torch.pi/(2*self.n_fft)*torch.arange(start=0, end=2*self.n_fft,step=2, dtype=torch.float64, device=self.device))

    def forward(self, signal, return_frames:bool=False):
        assert signal.dim() == 3, 'Only tensors shaped in BHW are supported, got tensor of shape %s'%(str(signal.size()))
        assert signal.size()[-1] == self.n_fft//2, 'The last dim of input tensor should match the n_fft. Expected %d ,got %d'%(self.n_fft, signal.size()[-1])

        # Inverse transform at the last dim
        signal = self.exp1*signal
        signal = torch.fft.fft(signal,n=self.n_fft)
        signal = torch.real(signal*self.exp2)

        # Remove padded zeros when doing dct
        if self.n_fft > self.win_length:
            signal = signal[...,:self.win_length]

        # Apply windows to each pieces
        signal = torch.mul(signal, self.window)
        if return_frames:
            frames = signal.clone()
        else:
            frames = torch.zeros(1)

        # Overlapping adding by fold()
        out_len = (signal.size()[-2]-1) * self.hop_length + self.win_length
        signal = 4/self.n_fft*fold(signal.transpose_(-1,-2), kernel_size=(1,self.win_length), stride=(1,self.hop_length), output_size=(1,out_len))

        if self.center:
            # extract the middle part
            signal = signal[..., self.win_length//2:-self.win_length//2]
        signal = signal if self.out_length is None else signal[...,:self.out_length]
        return signal, frames

from typing import Optional, Union, Callable
from einops import rearrange
from torch_scatter import scatter
class FastMDCT4(torch.nn.Module):
    def __init__(self, n_fft: Optional[int] = 2048, hop_length: Optional[int] = None, win_length: Optional[int] = None, window: Union[torch.Tensor, np.ndarray, list, Callable, None] = None, center: bool = True, pad_mode: str = 'constant', device: str = 'cuda') -> None:
        super().__init__()
        self.n_fft = n_fft
        self.pad_mode = pad_mode
        self.device = device
        self.hop_length = hop_length
        self.center = center

        if callable(window):
            self.win_length = int(win_length)
            self.window = window(self.win_length).to(
                device=self.device, dtype=torch.float64)
        elif isinstance(window, torch.Tensor):
            self.window = window.to(device=self.device, dtype=torch.float64)
            self.win_length = len(window)
        elif isinstance(window, np.ndarray) or isinstance(window, list):
            self.window = torch.tensor(
                window, device=self.device, dtype=torch.float64)
            self.win_length = len(window)
        elif window is None:
            if win_length is not None:
                self.win_length = win_length
            elif n_fft is not None:
                self.win_length = n_fft
            else:
                assert False, 'You should specify window length or n_fft'
            self.window = torch.ones(
                (self.win_length,), device=self.device, dtype=torch.float64)
        else:
            raise NotImplementedError

        assert self.win_length <= self.n_fft, f'Window lenth {self.win_length} should be no more than fft length {self.n_fft}'
        assert self.hop_length <= self.win_length, 'You hopped more than one frame'

        self.idx = torch.stack((
            torch.arange(
                start=0, end=n_fft//2, step=2,
                dtype=torch.long, device=self.device),
            torch.arange(
                start=n_fft-1, end=n_fft//2, step=-2,
                dtype=torch.long, device=self.device),
            torch.arange(
                start=n_fft//2, end=n_fft, step=2,
                dtype=torch.long, device=self.device),
            torch.arange(
                start=n_fft//2-1, end=0, step=-2,
                dtype=torch.long, device=self.device)
        ), dim=0)

        # self.sqrtN = torch.sqrt(torch.tensor(
        #     [self.n_fft], device=self.device, dtype=torch.float64))
        self.post_exp = torch.exp(
            -2j*torch.pi/self.n_fft*(
                torch.arange(
                    start=0,
                    end=self.n_fft//4,
                    step=1,
                    dtype=torch.float64,
                    device=self.device
                )+1/8
            )
        ).to(torch.complex64)

        self.pre_exp = (self.make_pre_exp()*self.window).to(torch.complex64)
        self.pre_idx = self.make_pre_idx()
        self.post_idx = self.make_post_idx()
        # self.idx = self.idx.clone().mT.roll(self.n_fft//8,0).contiguous()

    def make_pre_exp(self):
        sgn = torch.ones(1, self.n_fft, dtype=torch.complex128,
                         device=self.device)
        # Shift for Time-Domain Aliasing Cancellation (TDAC)
        sgn[..., -self.n_fft//4:] *= -1
        sgn = sgn.roll(self.n_fft//4, dims=-1)
        sgn[..., self.idx[0]] *= self.post_exp
        sgn[..., self.idx[1]] *= -self.post_exp
        sgn[..., self.idx[2]] *= -1j*self.post_exp
        sgn[..., self.idx[3]] *= 1j*self.post_exp
        return sgn.roll(-self.n_fft//4, dims=-1).contiguous()

    def make_pre_idx(self):
        i = torch.arange(start=0, end=self.n_fft, step=1, dtype=torch.long, device=self.device)
        i = i.roll(self.n_fft//4,dims=-1)
        idx_ = torch.stack([i[self.idx[0]], i[self.idx[1]], i[self.idx[2]], i[self.idx[3]]],dim=1)
        index = torch.zeros(1,self.n_fft,device=self.device, dtype=torch.long)
        for i in torch.arange(0,self.n_fft//4,dtype=torch.long):
          index[...,idx_[i]]=i
        return index.squeeze().contiguous()

    def make_post_idx(self):
        idx = torch.arange(self.n_fft//2, dtype=torch.long,
                           device=self.device).reshape(-1, 2)
        idx[:, 1] = idx[:, 1].flip(-1)
        return idx.flatten().contiguous()

    def forward(self, signal: torch.tensor, return_frames: bool = False):
        if signal.dim() == 2: # B T (mono)
            signal = signal[:, None, :]
        elif signal.dim() == 3: # B C T (stereo)
            pass
        else:
            raise NotImplementedError

        # Pad the signal to a proper length
        B, C, T = signal.shape
        start_pad = 0
        # Pad the signal so that the t-th frame is centered at time t * hop_length. Otherwise, the t-th frame begins at time t * hop_length.
        if self.center:
            start_pad = self.hop_length
        additional_len = T % self.hop_length
        end_pad = start_pad
        if additional_len:
            end_pad = start_pad + self.hop_length - additional_len
        signal = pad(signal, (start_pad, end_pad), mode=self.pad_mode)

        # Slice the signal with overlapping
        signal = signal.unfold(dimension=-1, size=self.win_length, step=self.hop_length)
        signal = signal*self.pre_exp
        signal = scatter(signal, self.pre_idx, dim=-1, reduce='sum')
        signal = torch.fft.fft(signal,dim=-1)
        # post-twiddle
        signal = torch.conj_physical(signal*self.post_exp)
        # rearranging
        signal = torch.view_as_real(signal)
        signal = signal.flatten(-2)[..., self.post_idx]

        return signal, None


class FastIMDCT4(torch.nn.Module):
    def __init__(self, n_fft: Optional[int] = 2048, hop_length: Optional[int] = None, win_length: Optional[int] = None, window: Union[torch.Tensor, np.ndarray, list, Callable, None] = None, center: bool = True, pad_mode: str = 'constant', out_length: Optional[int] = None, device: str = 'cuda') -> None:
        super().__init__()
        self.n_fft = n_fft
        self.pad_mode = pad_mode
        self.device = device
        self.hop_length = hop_length
        self.center = center
        self.out_length = out_length

        if callable(window):
            self.win_length = int(win_length)
            self.window = window(self.win_length).to(
                device=self.device, dtype=torch.float64)
        elif isinstance(window, torch.Tensor):
            self.window = window.to(device=self.device, dtype=torch.float64)
            self.win_length = len(window)
        elif isinstance(window, np.ndarray) or isinstance(window, list):
            self.window = torch.tensor(
                window, device=self.device, dtype=torch.float64)
            self.win_length = len(window)
        elif isinstance(window, None):
            if win_length is not None:
                self.win_length = win_length
            elif n_fft is not None:
                self.win_length = n_fft
            else:
                assert False, 'You should specify window length or n_fft'
            self.window = torch.ones(
                (self.win_length,), device=self.device, dtype=torch.float64)
        else:
            raise NotImplementedError

        assert self.win_length <= self.n_fft, f'Window lenth {self.win_length} should be no more than fft length {self.n_fft}'
        assert self.hop_length <= self.win_length, 'You hopped more than one frame'

        self.exp = torch.exp(
            -2j*torch.pi/self.n_fft*(
                torch.arange(
                    start=0,
                    end=self.n_fft//4,
                    step=1,
                    dtype=torch.float32,
                    device=self.device
                )+1/8
            )
        ).contiguous()
        self.pre_idx = self.make_pre_idx()
        self.post_idx = self.make_post_index()
        self.window = (4.0*self.make_sign()*self.window/self.n_fft).to(torch.float32).contiguous()

    def make_pre_idx(self):
        a = torch.arange(self.n_fft//2, dtype=torch.long,
                         device=self.device).unfold(-1, 2, 2)
        return torch.stack((a[..., 0], a[..., 1].flip(-1)), dim=-1).contiguous()

    def make_post_index(self):
        a = torch.arange(0, self.n_fft//2, 2,
                         dtype=torch.long, device=self.device)
        b = torch.arange(self.n_fft//2-1, 0, -2,
                         dtype=torch.long, device=self.device)
        idx = torch.empty((self.n_fft,), dtype=torch.long, device=self.device)
        idx[0:self.n_fft//2:2] = a
        idx[1:self.n_fft//2:2] = b
        idx[self.n_fft//2:] = idx[:self.n_fft//2].flip(0)
        return idx.roll(-self.n_fft//4).contiguous()

    def make_sign(self):
        sign = torch.ones((self.n_fft,), device=self.device,
                          dtype=torch.float64)
        sign[1::2] *= -1
        sign[..., 0:self.n_fft//4] *= -1
        return sign.roll(-self.n_fft//4).contiguous()

    def forward(self, signal: torch.Tensor, return_frames: bool = False):
        assert signal.dim(
        ) <= 4, f'Only tensors shaped in BHW or BCHW are supported, got tensor of shape {signal.shape}'
        assert signal.shape[
            -1] == self.n_fft//2, f'The last dim of input tensor should match the n_fft. Expected {self.n_fft}, got {signal.shape[-1]}'

        if signal.dim() == 4:
            C = signal.shape[1]
            signal = rearrange(signal, 'B C T N -> (B C) T N')
        else:
            C = 1

        signal = signal.to(self.device)
        # # Inverse transform at the last dim
        signal = torch.view_as_complex(signal[..., self.pre_idx])

        signal = self.exp*signal
        signal = torch.fft.fft(signal)
        signal = self.exp*signal

        # [0+4j, 1+5j, 2+6j, 3+7j] -> [2,-5, 3, -4, 4, -3, 5, -2, 6, -1, 7, 0, 0, 7, -1, 6]
        signal = torch.view_as_real(signal).flatten(-2)[..., self.post_idx]

        # Apply windows to each pieces
        signal = self.window*signal
        if return_frames:
            frames = signal.clone()
        else:
            frames = torch.empty(1)

        # Overlapping adding by fold()
        out_len = (signal.shape[-2]-1) * self.hop_length + self.win_length
        signal = fold(signal.mT, kernel_size=(1, self.win_length),
                      stride=(1, self.hop_length), output_size=(1, out_len))

        if self.center:  # extract the middle part
            signal = signal[..., self.win_length//2:-self.win_length//2]
        if self.out_length is not None:
            signal = signal[..., :self.out_length]
        if C != 1:
            signal = rearrange(signal, '(B C) T N-> B C T N')
        return signal, frames