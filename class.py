#-------------------------------------self--------------------------------------------------
class Person:
    def __init__(self, name, job):
        self.name = name
        self.job = job
        
    def introduce(self):
        return f"내 이름은 {self.name}, {self.job}이죠"

p = Person("코난", "탐정")
print(p.introduce())

# self = 클래스의 메소드를 실행시키는데 그 첫 인자로 인스턴스를 주는것
#------------------------------__init__,__call__---------------------------------------------
class A:
    def __init__(self):
        print('init')
    def __call__(self):
        print('call')
    def myfunc(self):
        print('my')

a = A() # init : init은 객체 생성될 때 불러와짐
a() # call : call은 인스턴스 생성될 때 불러와짐
a.myfunc() # my


# ------------------------------------상속-----------------------------------------------------
class Person:# 부모 클래스
    def __init__(self, name, age, gender):
        self.Name = name 
        self.Age = age 
        self.Gender = gender 
    def aboutMe(self):
        print("저의 이름은 " + self.Name + "이구요, 제 나이는 " + self.Age + "살 입니다.")
class Employee(Person): #자식 클래스
    def __init__(self, name, age, gender, salary, hiredate):
        Person.__init__(self, name, age, gender) # 부모로 부터 상속 받은것을 사용함 이름, 나이, 성별
        self.Salary = salary 
        self.Hiredate = hiredate 
    def doWork(self):
        print("열심히 일을 합니다.") 
    def aboutMe(self):
        Person.aboutMe(self) 
        print("제 급여는 " + self.Salary + "원 이구요, 제 입사일은 " + self.Hiredate + " 입니다.") 

objectEmployee = Employee("김철수", "18", "남", "5000000", "2013년 10월 28일")
objectEmployee.doWork() # 열심히 일을 합니다.
objectEmployee.aboutMe() # 자식클래스에 있는 aboutMe매서드를 출력했는데 부모 클래스에 있는 이름, 나이 성별이 출력됨을 확인할수 있음
'''
저의 이름은 김철수이구요, 제 나이는 18살 입니다.
제 급여는 5000000원 이구요, 제 입사일은 2013년 10월 28일 입니다.
'''
