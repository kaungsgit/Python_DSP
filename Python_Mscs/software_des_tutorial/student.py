from person import Person
# from enroll import Enroll
import enroll


class Student(Person):
    def __init__(self, first, last, dob, phone, address, international=False):
        super().__init__(first, last, dob, phone, address)
        self.international = international
        self.enrolled = []

    def add_enrollment(self, enroll_):
        if not isinstance(enroll_, enroll.Enroll):
            raise RuntimeError("Invalid Enroll...")

        self.enrolled.append(enroll_)

    def is_on_probation(self):
        return False

    def is_part_time(self):
        return len(self.enrolled) <= 3
