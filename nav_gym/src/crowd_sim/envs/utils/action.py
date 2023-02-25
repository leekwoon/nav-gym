from collections import namedtuple

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRot = namedtuple('ActionRot', ['v', 'r'])
ActionXYRot = namedtuple('ActionXYRot', ['vx', 'vy', 'r'])
