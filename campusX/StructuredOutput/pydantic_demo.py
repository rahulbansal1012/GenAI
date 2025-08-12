from pydantic import BaseModel

class student(BaseModel):
    name : str
    age : int

new_student  = {'name' : 'Rahul' , 'age' : 25}

student = student(**new_student)
print(student)