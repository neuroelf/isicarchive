"""
isicarchive.lut

This module generates color look-up tables (LUTs) for coloring
statistical results.
"""


from typing import Callable, Iterator, Union


# some simple variables and definitions
LUT_RED = 0
LUT_GREEN = 1
LUT_BLUE = 2
LUT_MSIZE = 39.001

def p255(v:float, d:float=50.0) -> float:
    return 255.0 - d * (4.0 * (0.25 - (v - 0.5) ** 2))
def nsq255(v:float) -> float:
    return 255.0 * (1.0 - (1.0 - v) ** 2)
def t255(v:float) -> float:
    return 255.0 * v
rgb255 = {
    'redt': t255,
    'greent': t255,
    'bluet': t255,
}


# local implementation of arange(...)
def lut_range(rfrom:float, rto:float, rstep:float = 1.0) -> Iterator[float]:
    if rstep == 0.0:
        raise ValueError('rstep must be != 0.0')
    val = rfrom
    if rstep > 0.0:
        while val < rto:
            yield val
            val += rstep
    else:
        while val > rto:
            yield val
            val += rstep

# compile a LUT from single values, lists, or range definitions
def lut_compile(
    red:Union[int, list],
    green:Union[int, list],
    blue:Union[int, list],
    redt:Callable[[float], float] = None,
    greent:Callable[[float], float] = None,
    bluet:Callable[[float], float] = None,
    ) -> list:
    if isinstance(red, list) and len(red) == 3:
        red = [r for r in lut_range(red[0], red[1], red[2])]
    if isinstance(green, list) and len(green) == 3:
        green = [g for g in lut_range(green[0], green[1], green[2])]
    if isinstance(blue, list) and len(blue) == 3:
        blue = [b for b in lut_range(blue[0], blue[1], blue[2])]
    if not isinstance(red, list):
        if not isinstance(green, list):
            if not isinstance(blue, list):
                raise ValueError('at least one color must be a list.')
            else:
                red = [red for r in range(len(blue))]
        else:
            red = [red for r in range(len(green))]
    if not isinstance(green, list):
        green = [green for g in range(len(red))]
    if not isinstance(blue, list):
        blue = [blue for b in range(len(red))]
    if len(red) != len(green) or len(green) != len(blue):
        raise ValueError('resulting lists must be of the same length')
    l = [None] * len(red)
    idx = 0
    for (r, g, b) in zip(red, green, blue):
        if not redt is None:
            r = redt(r)
        if not greent is None:
            g = greent(g)
        if not bluet is None:
            b = bluet(b)
        if r > 255.0:
            r = 255.0
        elif r < 0.0:
            r = 0.0
        if g > 255.0:
            g = 255.0
        elif g < 0.0:
            g = 0.0
        if b > 255.0:
            b = 255.0
        elif b < 0.0:
            b = 0.0
        l[idx] = [int(r+0.5), int(g+0.5), int(b+0.5)]
        idx += 1
    return l


# actual definitions
LUT_COOL = [
    lut_compile([ 0.0, 1.0,  1.0/LUT_MSIZE], [ 1.0, 0.0  , -1.0/LUT_MSIZE],              1.0           , **rgb255)]
LUT_FULL = [
    lut_compile(             1.0,            [ 1.0, 0.125, -1.0/44.6     ], [ 1.0, 0.0, -1.0/LUT_MSIZE], t255, nsq255, t255),
    lut_compile([ 1.0, 0.0, -1.0/LUT_MSIZE], [ 1.0, 0.125, -1.0/44.6     ], [ 1.0, 0.0, -1.0/LUT_MSIZE], t255, nsq255, p255),
]
LUT_GRAY = [
    lut_compile([ 0.5, 1.0,  0.5/LUT_MSIZE], [ 0.5, 1.0  ,  0.5/LUT_MSIZE], [ 0.5, 1.0,  0.5/LUT_MSIZE], **rgb255),
    lut_compile([ 0.5, 0.0, -0.5/LUT_MSIZE], [ 0.5, 0.0  , -0.5/LUT_MSIZE], [ 0.5, 0.0, -0.5/LUT_MSIZE], **rgb255),
]
LUT_HOT = [
    lut_compile([ 0.0, 2.5,  2.5/LUT_MSIZE], [-1.0, 1.5  ,  2.5/LUT_MSIZE], [-3.0, 1.0,  4.0/LUT_MSIZE], **rgb255),
]
LUT_REVERSED = [
    lut_compile(             0.0,            [0.25, 1.0  , 0.75/LUT_MSIZE], [ 1.0,0.25,-0.75/LUT_MSIZE], **rgb255),
    lut_compile(             1.0,            [0.25, 1.0  , 0.75/LUT_MSIZE],              0.0           , **rgb255),
]
LUT_STANDARD = [
    lut_compile(             1.0,            [0.25, 1.0  , 0.75/LUT_MSIZE],              0.0           , **rgb255),
    lut_compile(             0.0,            [0.25, 1.0  , 0.75/LUT_MSIZE], [ 1.0,0.25,-0.75/LUT_MSIZE], **rgb255),
]
LUT_WINTER = [
    lut_compile(             0.0,            [ 0.0, 1.0  ,  1.0/LUT_MSIZE], [ 1.0, 0.5, -0.5/LUT_MSIZE], **rgb255),
]

LUTs = {
    'cool': LUT_COOL,
    'full': LUT_FULL,
    'gray': LUT_GRAY,
    'hot': LUT_HOT,
    'reversed': LUT_REVERSED,
    'standard': LUT_STANDARD,
    'winter': LUT_WINTER,
}
