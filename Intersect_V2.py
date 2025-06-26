#This file is more of a test file that might slowly and eventually be made into Intersect V2
from Intersect_Engine_V2 import IntersectEngine, start, Element
from random import random

global runtime_active
runtime_active = True

class Interest_v2(IntersectEngine):
    title = "Intersect V2"
    window_size = (700, 700)
    resizable = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        runtime_active = True
        self.test = Element(iee_filepath="assets/template.iee")


if __name__ != '__main__':
    try:
        start(Interest_v2)
    except Exception as e:
        print(f"ERR: {e}")
    finally:
        runtime_active = False

start(Interest_v2)