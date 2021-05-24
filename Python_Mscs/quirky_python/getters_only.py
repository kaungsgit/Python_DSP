"""
Prime example of "protected"/"read only" attribute that can only be accessed but not set (only getter, no setter)
It'll be write only if you implement only the setters.
"""


class Student:
    def __init__(self, first_name, last_name):
        self._first_name = first_name
        self._last_name = last_name

    @property
    def first_name(self):
        return self._first_name

    @property
    def last_name(self):
        return self._last_name

    # @last_name.setter
    # def last_name(self, last_name):
    #     self._last_name = last_name

    @property
    def name(self):
        return f"{self.first_name} {self.last_name}"


student1 = Student('John', 'Doe')
pass
