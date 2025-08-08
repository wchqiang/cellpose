class Animal:
    __run = "great"
    def __init__(self, name, age):
        self.name = name 
        self.age  = age 
        self._speak = "haha"

    def speak(self):
        print(f"{self.name} can speak {self._speak}")
        return """f'"great speak"+
        "greak question"'"""

    @staticmethod
    def add(a:int, b:int) -> int:
        return a + b
    
    @classmethod
    def from_birth_year(cls, name: str, birth_year: int):
        age = 2023 - birth_year
        return cls(name, age)
    
dog = Animal("wang",12)
print(dog.speak())
print("--"*10)
print(dog._Animal__run)
print(dog.add(5, 10))