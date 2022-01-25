from typing import Callable
import numpy as np
import pygame, pygame.sndarray
from math import sin, cos
from time import sleep

sampling = 44100
pygame.mixer.init(frequency=sampling, size=-16, channels=1)


def play_for(wave: np.ndarray, ms: int) -> None:
    """Play the given NumPy array, as a sound, for ms milliseconds."""
    size = wave.size
    wave = np.repeat(wave.reshape(size, 1), 2, axis=1).astype(np.int16)
    sound = pygame.sndarray.make_sound(wave)
    sound.play(-1)
    pygame.time.delay(ms)
    sound.stop()


def summation(func: Callable[[int, float], float], x:float, *, a: int = 1, b: int) -> float:
    """
    Returns a summation of func(n) as n goes from a to b, inclusive. It's just a sigma function.

    :param func: the function to be summed
    :param x: the x value of the function
    :param a: lower bound of n, inclusive
    :param b: upper bound of n, inclusive
    :return: a single floating-point number
    """
    return sum([func(n, x) for n in range(a, b + 1)])


def sine_wave(hz, peak, *, n_samples=sampling):
    """
    Compute N samples of a sine wave with given frequency and peak amplitude. Defaults to one second.
    """
    length = sampling / float(hz)
    omega = np.pi * 2 / length
    x_values = np.arange(int(length)) * omega
    onecycle = peak * np.sin(x_values)
    return np.resize(onecycle, (n_samples,)).astype(np.int16)


def sigma_wave(*, func: Callable[[int, float], float], peak: int = 2048, a: int = 1, b: int = 25) -> np.ndarray:
    x_values = np.arange(sampling * np.pi) / 16
    print(x_values)
    wave = np.array([sum([func(n, x) for n in range(a, b + 1)]) for x in x_values])
    print(wave)
    print(np.max(wave))
    print(wave / np.max(wave) * peak)
    print(np.max(wave / np.max(wave) * peak))
    return (wave / np.max(wave) * peak).astype(np.int16)


print(sine_wave(440, 2048))
print(np.max(sine_wave(440, 2048)))
# play_for(sine_wave(440, 2048), 500)
sin_wave = lambda n, x: sin(x)
square_wave = lambda n, x: (1 / (2*n-1)) * sin((2*n-1)*x)
saw_wave = lambda n, x: (1 / n) * sin(n * x)
print(summation(lambda n, x: (1 / (2*n-1)) * sin((2*n-1)*x), 2, b=25))
play_for(sigma_wave(func=sin_wave, b=25, peak=4096), 1000)
