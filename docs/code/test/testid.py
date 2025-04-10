'''
Author: coffeecat
Date: 2025-04-04 09:22:10
LastEditors: Do not edit
LastEditTime: 2025-04-04 09:29:18
'''
class MyClass:
    # 类变量，可模拟静态变量的部分特性
    class_variable = "This is a class variable (simulating static variable)"

    def __init__(self, value):
        # 实例变量
        self.instance_variable = value

    @classmethod
    def class_method(cls):
        return f"Class method accessed class variable: {cls.class_variable}"

    def instance_method(self):
        return f"Instance method accessed instance variable: {self.instance_variable}"

    @staticmethod
    def static_method():
        return "This is a static method"


# 打印类创建时的内存地址
print(f"类 MyClass 的内存地址: {id(MyClass)}")
print(f"类变量 class_variable 的内存地址: {id(MyClass.class_variable)}")
print(f"类方法 class_method 的内存地址: {id(MyClass.class_method)}")
print(f"静态方法 static_method 的内存地址: {id(MyClass.static_method)}")

# 创建两个实例
instance1 = MyClass(10)
instance2 = MyClass(20)

# 打印实例的内存地址
print(f"\n实例 instance1 的内存地址: {id(instance1)}")
print(f"实例 instance1 的实例变量 instance_variable 的内存地址: {id(instance1.instance_variable)}")
print(f"实例 instance1 调用实例方法的内存地址: {id(instance1.instance_method)}")
print(f"\n实例 instance2 的内存地址: {id(instance2)}")
print(f"实例 instance2 的实例变量 instance_variable 的内存地址: {id(instance2.instance_variable)}")
print(f"实例 instance2 调用实例方法的内存地址: {id(instance2.instance_method)}")

# 调用类方法
print("\n调用类方法:")
print(MyClass.class_method())

# 调用实例方法
print("\n调用实例 1 的实例方法:")
print(instance1.instance_method())
print("调用实例 2 的实例方法:")
print(instance2.instance_method())

# 调用静态方法
print("\n调用静态方法:")
print(MyClass.static_method())



"""
 实例方法是绑定方法
在 Python 中，实例方法是绑定方法（bound method）。当你通过实例去调用实例方法时，Python 会把实例和方法绑定在一起，生成一个新的绑定方法对象。对于不同的实例，即使调用的是同一个类的相同实例方法，它们所生成的绑定方法对象也是不同的。
"""
# https://pythontutor.com/python-compiler.html#mode=edit