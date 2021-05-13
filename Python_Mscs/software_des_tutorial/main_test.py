from course import Course
from person import Person
from address import Address
from student import Student
from professor import Professor
from enroll import Enroll


def enroll_course(student, course):
    enroll = Enroll(student, course)
    student.add_enrollment(enroll)
    course.add_enrollment(enroll)


address1 = Address('US', 'MA', 'Boston', '94 Lake Street', '01876')
student1 = Student('Kaung1', 'Oo', '5_10_1993', '1234', address1, international=True)

student2 = Student('Kaung2', 'Oo', '5_10_1993', '1234', address1, international=True)

student3 = Student('Kaung3', 'Oo', '5_10_1993', '1234', address1, international=True)

student4 = Student('Kaung4', 'Oo', '5_10_1993', '1234', address1, international=True)

prof1_addr = Address('US', 'FL', 'Orlando', '99 Lake Street', '043432')
prof1 = Professor('Kong', 'King', '5_10_1887', '4321', prof1_addr, 80e3)

course1 = Course('Intro ECE', 'ECE101', 6, 2, prof1)
prof1.add_course(course1)

enroll_course(student1, course1)

enroll_course(student2, course1)

enroll_course(student3, course1)

enroll_course(student4, course1)

enroll_course(student4, course1)

enroll_course(student4, course1)

pass
