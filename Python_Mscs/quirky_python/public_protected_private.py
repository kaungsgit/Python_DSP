class House:

    def __init__(self, price, color, age):
        self.color = color
        self._price = price  # private attribute
        self.__age = age

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

# h1 = House(10e3, 'blue', 85)

# pass
