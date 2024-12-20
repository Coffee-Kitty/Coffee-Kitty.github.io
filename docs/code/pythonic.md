# Pythonic





## 装饰器

普通装饰器

```python
def decorator(func):
    def wrapper(*args,**kwargs):
        print(f"{func.__name__} execute start")

        result = func(*args,**kwargs)

        print(f"{func.__name__} execute end")

        return result

    return wrapper

```

共有两种调用方法：

1.

```python
def compute():
    print("1+2==:",3)
decorator_compute = decorator(compute)

decorator_compute()
"""
compute execute start
1+2==: 3
compute execute end
"""
```

2.

```python
@decorator
def compute():
    print("1+2==:",3)
compute()
# """
# compute execute start
# 1+2==: 3
# compute execute end
# """
```



