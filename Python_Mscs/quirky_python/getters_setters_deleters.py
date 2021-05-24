"""https://www.freecodecamp.org/news/python-property-decorator/#:~:text=1%EF%B8%8F%E2%83%A3%20Advantages%20of
%20Properties%20in%20Python&text=Properties%20can%20be%20considered%20the,
is%20very%20concise%20and%20readable.&text=By%20using%20%40property%2C%20you%20can,
getters%2C%20setters%2C%20and%20deleters. """

"""Most useful when
1. You want to have some encapsulation (getters, _price is private attribute)
2. You want to perform argument validation (setters)
3. By defining properties, you can change the internal implementation of a class without 
affecting the program, so you can add getters, setters, and deleters that act as intermediaries 
"behind the scenes" to avoid accessing or modifying the data directly.

"""


class House:

    def __init__(self, price):
        self._price = price

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, new_price):
        if new_price > 0 and isinstance(new_price, float):
            self._price = new_price
        else:
            print("Please enter a valid price")

    @price.deleter
    def price(self):
        del self._price


# A weirder way of implementing getters and setters
# still, you can call FinalClass.a to get and .a=something to set
class FinalClass:

    def __init__(self, var):
        # calling the set_a() method to set the value 'a' by checking certain conditions
        self._set_a(var)

    # getter method to get the properties using an object
    def _get_a(self):
        return self._a

    # setter method to change the value 'a' using an object
    def _set_a(self, var):

        # condition to check whether var is suitable or not
        if var > 0 and var % 2 == 0:
            self._a = var
        else:
            self._a = 2

    # this defines the .x syntax for getter and setter
    attri = property(_get_a, _set_a)


obj1 = FinalClass(100)
print(obj1.attri)
obj1.attri = 134
print(obj1.attri)

pass
