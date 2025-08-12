from typing import TypedDict

class person(TypedDict):
    name : str
    age : int

new_person : person = {'name' : 'Rahul', 'age':23}
print(new_person)