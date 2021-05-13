from professor import Professor
# from enroll import Enroll
import enroll


class Course:
    def __init__(self, name, code, max_, min_, professor):
        self.name = name
        self.code = code
        self.max = max_
        self.min = min_
        self.professors = []
        self.enrollments = []

        if isinstance(professor, Professor):
            self.professors.append(professor)
        elif isinstance(professor, list):
            for entry in professor:
                if not isinstance(entry, Professor):
                    raise RuntimeError("Invalid professor...")

            self.professors = professor
        else:
            raise RuntimeError("Invalid professor...")

    def add_professor(self, professor):
        if not isinstance(professor, Professor):
            raise RuntimeError("Invalid professor...")

        self.professors.append(professor)

    def add_enrollment(self, enroll_):
        if not isinstance(enroll_, enroll.Enroll):
            raise RuntimeError("Invalid Enroll")

        if len(self.enrollments) == self.max:
            raise RuntimeError("Cannot enroll, course if full...")

        self.enrollments.append(enroll_)

    def is_cancelled(self):
        return len(self.enrollments) < self.min
