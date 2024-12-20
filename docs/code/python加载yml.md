[YAML 语言教程](https://www.ruanyifeng.com/blog/2016/07/yaml.html) 

[YAML 入门教程](https://www.runoob.com/w3cnote/yaml-intro.html)

[python加载yaml](https://blog.csdn.net/John_xyz/article/details/80183244)

## yaml

YAML 语言（发音 /ˈjæməl/ ）的设计目标，就是方便人类读写。在开发的这种语言时，YAML 的意思其实是："Yet Another Markup Language"（仍是一种标记语言)

<font color='red'>实质上是一种通用的数据串行化格式</font>。

这里简单介绍下YAML的语法。



### 简介

基本语法规则：

* <font color='red'>大小写敏感</font>
* 使用<font color='red'>缩进表示层级</font>关系
* 缩进时不允许使用Tab键，<font color='green'>只允许使用空格</font>。
* 缩进的<font color='red'>空格数目不重要</font>，只要相同层级的元素左侧对齐即可
* `#` 表示注释，从这个字符一直到行尾，都会被解析器忽略。

支持三种数据结构：

- 对象：<font color='red'>键值对的集合</font>，又称为映射（mapping）/ 哈希（hashes） / 字典（dictionary）
- 数组：<font color='green'>一组按次序排列的值</font>，又称为序列（sequence） / 列表（list）
- 纯量（scalars）：单个的、不可再分的值



### 对象

对象键值对使用冒号结构表示 **key: value**，冒号后面要加一个空格。

可以使用 **key:{key1: value1, key2: value2, ...}**。

还可以使用缩进表示层级关系；

key: 
    child-key: value
    child-key2: value2

---

下面是简单例子：

对象的一组键值对，使用冒号结构表示

```yaml
animal: pets
```

对应于{ animal: 'pets' }

Yaml 也允许另一种写法，将所有键值对写成一个行内对象。

```yaml
hash: { name: Steve, foo: bar }
```

对应于{ hash: { name: 'Steve', foo: 'bar' } }

### 数组

以 **-**（连词线） 开头的行表示构成一个数组：

```yaml
- Cat
- Dog
- Goldfish
- lion
```

对应于[ 'Cat', 'Dog', 'Goldfish' ]

数据结构的子成员是一个数组，则可以在该项下面缩进一个空格。

```yaml
-
 - Cat
 - Dog
 - Goldfish
```

对应于[ [ 'Cat', 'Dog', 'Goldfish' ] ]

YAML 支持多维数组，可以使用行内表示：

```yaml
animal: [cat, dog]
```



一个相对复杂的例子：

```yaml
companies:
    -
        id: 1
        name: company1
        price: 200W
    -
        id: 2
        name: company2
        price: 500W
```

意思是 companies 属性是一个数组，每一个数组元素又是由 id、name、price 三个属性构成。







复合使用，如

```yaml

languages:
 - Ruby
 - Perl
 - Python 
websites:
 YAML: yaml.org 
 Ruby: ruby-lang.org 
 Python: python.org 
 Perl: use.perl.org 
```

对应着

{ languages: [ 'Ruby', 'Perl', 'Python' ],
  websites: 
   { YAML: 'yaml.org',
     Ruby: 'ruby-lang.org',
     Python: 'python.org',
     Perl: 'use.perl.org' } }



### 纯量

纯量是最基本的，不可再分的值，包括：

- 字符串
- 布尔值
- 整数
- 浮点数
- Null
- 时间
- 日期



使用一个例子来快速了解纯量的基本使用：

```yaml
boolean: 
    - TRUE  #true,True都可以
    - FALSE  #false，False都可以
float:
    - 3.14
    - 6.8523015e+5  #可以使用科学计数法
int:
    - 123
    - 0b1010_0111_0100_1010_1110    #二进制表示
null:
    nodeName: 'node'
    parent: ~  #使用~表示null
string:
    - 哈哈
    - 'Hello world'  #可以使用双引号或者单引号包裹特殊字符
    - newline
      newline2    #字符串可以拆成多行，每一行会被转化成一个空格
date:
    - 2018-02-17    #日期必须使用ISO 8601格式，即yyyy-MM-dd
datetime: 
    -  2018-02-17T15:02:31+08:00    #时间使用ISO 8601格式，时间和日期之间使用T连接，最后使用+代表时区
```



### 引用

**&** 锚点和 ***** 别名，可以用来引用:

```yaml
defaults: &defaults
  adapter:  postgres
  host:     localhost

development:
  database: myapp_development
  <<: *defaults

test:
  database: myapp_test
  <<: *defaults
```



```yaml
defaults:
  adapter:  postgres
  host:     localhost

development:
  database: myapp_development
  adapter:  postgres
  host:     localhost

test:
  database: myapp_test
  adapter:  postgres
  host:     localhost
```



<font color='red'>**&** 用来建立锚点（defaults），**<<** 表示合并到当前数据，***** 用来引用锚点。</font>

下面是另一个例子:

```yaml
- &showell Steve 
- Clark 
- Brian 
- Oren 
- *showell 
```

转为 JavaScript 代码如下:

```javascript
[ 'Steve', 'Clark', 'Brian', 'Oren', 'Steve' ]
```



## python解析yaml

下面来看两个例子看python如何load和dump yaml文件。

### load yaml文件

假设我们有如下yaml文件

```yaml
# test.yaml
age: 37
spouse:
  name: Jane Smith
  age: 25
children:
  - name: Jimmy Smith
    age: 15
  - name1: Jenny Smith
    age1: 12
```

可以使用yaml.load将文件解析成一个字典

```python
import yaml

with open('test.yaml','r') as f:
    data = yaml.load(f)

print(type(data))
print(data)

'''
输出结果
<type 'dict'>
{'age': 37, 'spouse': {'age': 25, 'name': 'Jane Smith'}, 'children': [{'age': 15, 'name': 'Jimmy Smith'}, {'age1': 12, 'name1': 'Jenny Smith'}]}

'''
```

### dump yaml文件

```python
import yaml

data = {'name':'johnson', 'age':23,
        'spouse':{'name':'Hallie', 'age':23},
        'children':[{'name':'Jack', 'age':2}, {'name':'Linda', 'age':2}]}

with open('test2.yaml','w') as f:
    f.write(yaml.dump(data))
    print yaml.dump(data)
'''
输出结果：
age: 23
children:
- {age: 2, name: Jack}
- {age: 2, name: Linda}
name: johnson
spouse: {age: 23, name: Hallie}
'''
```

得到的yaml文件如下所示:

```yaml
# test2.yaml
age: 23
children:
- {age: 2, name: Jack}
- {age: 2, name: Linda}
name: johnson
spouse: {age: 23, name: Hallie}
```



## 







