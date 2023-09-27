import random
import string


class Run:

    def __init__(self):
        self.id = ''.join(random.choices(string.ascii_lowercase +
                                              string.digits, k=7))