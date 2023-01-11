import os

os.chdir("A1")
print(os.getcwd())
from A1.A1 import a1
a1()

os.chdir("../A2")
print(os.getcwd())
from A2.A2 import a2
a2()

os.chdir("../B1")
print(os.getcwd())
from B1.B1 import b1
b1()

os.chdir("../B2")
print(os.getcwd())
from B2.B2 import b2
b2()