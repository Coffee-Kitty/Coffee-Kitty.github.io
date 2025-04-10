# -*- coding: utf-8 -*
'''
Author: coffeecat
Date: 2025-04-04 09:01:51
LastEditors: Do not edit
LastEditTime: 2025-04-04 09:02:03
'''
# 全局变量
global_variable_2 = "Global variable in module2"

# 定义一个函数
def function_in_module2():
    # 局部变量
    local_variable_2 = "Local variable in function_in_module2"
    print("Function function_in_module2 locals:", locals())
    return local_variable_2

# 定义一个类
class ClassInModule2:
    class_variable_2 = "Class variable in ClassInModule2"

    def __init__(self):
        self.instance_variable_2 = "Instance variable in ClassInModule2"

    def method_in_class2(self):
        local_variable_in_method = "Local variable in method_in_class2"
        print("Method method_in_class2 locals:", locals())
        return local_variable_in_method


print("Module2 globals:", globals())
