# -*- coding: utf-8 -*
'''
Author: coffeecat
Date: 2025-04-03 23:40:18
LastEditors: Do not edit
LastEditTime: 2025-04-04 09:19:24
'''
# # -*- coding: utf-8 -*
# print(globals())
# # from .test123.test123 import *
# # import test.test123.test123
# print(123)
# print(globals())
# a = 12
# if __name__ =='__main__':
#     print("hello main_")

# from . import test2
# 全局变量
global_variable_1 = "Global variable in module1"

# 定义一个函数
def function_in_module1():
    # 局部变量
    local_variable_1 = "Local variable in function_in_module1"
    print("Function function_in_module1 locals:", locals().keys())
    print("Function function_in_module1 globals:", globals().keys())
    return local_variable_1

# 定义一个类
class ClassInModule1:
    class_variable_1 = "Class variable in ClassInModule1"

    def __init__(self):
        self.instance_variable_1 = "Instance variable in ClassInModule1"
        print("__init__  locals:", locals().keys())
        print("__init__  locals:", globals().keys())

    def method_in_class1(self):
        local_variable_in_method = "Local variable in method_in_class1"
        print("Method method_in_class1 locals:", locals().keys())
        print("Method method_in_class1 locals:", globals().keys())
        return local_variable_in_method


print("Module1 globals:", globals().keys())
# python -m test.test

function_in_module1()

x = ClassInModule1()
x.method_in_class1()
print(id(ClassInModule1))
print(id(x.method_in_class1))
print(id(x))