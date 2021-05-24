""" Abstract methods are like you, the user, filling out a form, created by the developer
Abstract methods require you, the user, to implement these specific methods, which the developer
had left as blank. The developer's code then uses this method implementation to complete more tasks (such as
logging your data into the database or plotting your data).
For more info, read up Interfaces and Abstract Classes
"""
from abc import ABC, abstractmethod


class Polygon(ABC):

    @abstractmethod
    def no_of_sides(self):
        pass


class Triangle(Polygon):

    # overriding abstract method
    def no_of_sides(self):
        print("I have 3 sides")


# class Whiteboard:
#
#     # overriding abstract method
#     def write(self):
#         print("Writing...")

# the moment you inherit an abstract class, you need to provide the abstract method implementation
class Whiteboard(Polygon):

    def no_of_sides(self):
        print('I am a whiteobard with 4 sides')

    # overriding abstract method
    def write(self):
        print("Writing...")


shape1 = Triangle()
shape1.no_of_sides()

wb1 = Whiteboard()
wb1.write()
wb1.no_of_sides()

pass
