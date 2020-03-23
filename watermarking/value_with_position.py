"""This script is to encapsulate wavelet diff and its position as an object."""

class ValueWithPosition:
    """Only contains attributes"""
    def __init__(self, value, x, y):
        self.value = value
        self.x = x
        self.y = y
        