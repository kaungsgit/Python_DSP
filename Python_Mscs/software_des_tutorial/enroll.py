# from course import Course
# from student import Student
from datetime import datetime
import student
import course


class Enroll:
    def __init__(self, student_, course_):
        if not isinstance(student_, student.Student):
            raise RuntimeError("Invalid student...")

        if not isinstance(course_, course.Course):
            raise RuntimeError("Invalid course...")

        self.student = student_
        self.course = course_
        self.grade = None
        self.date = datetime.now()

    def set_grade(self, grade):
        self.grade = grade
