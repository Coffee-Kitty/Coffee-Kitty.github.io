
from typing import Generator,Set
import tree_sitter_c,tree_sitter_cpp,tree_sitter_java, tree_sitter_python
from tree_sitter import Language, Parser, Node, Tree

python_language = Language(tree_sitter_python.language())
c_language = Language(tree_sitter_c.language())
java_language = Language(tree_sitter_java.language())
cpp_language = Language(tree_sitter_cpp.language())


python_parser = Parser(python_language)
python_tree = python_parser.parse(
    bytes(
        """
\"\"\"
这是一个模块
\"\"\"
class A:
    pass
class B:
    \"\"\"
    这是一个类
    \"\"\"
    def __init__(self):
        pass
# 这是#注释
def foo():
    \"\"\"
    这是一个Python函数
    \"\"\"
    if bar:
        baz()
def sub():
    return a+b
""",
        "utf8"
    )
)

def traverse_tree(tree: Tree) -> Generator[Node, None, None]:
    """
    深度优先遍历树， 但是是生成器
    """
    cursor = tree.walk()

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent():
            break


# def breadth_first_traverse(tree: Tree) -> Generator[Node, None, None]:
#     """
#     层序遍历（广度优先遍历）树，使用队列实现
#     """
#     from collections import deque
#     queue = deque([tree.root_node])
    
#     while queue:
#         node = queue.popleft()
#         yield node
#         # 将所有子节点加入队列
#         for child in node.children:
#             queue.append(child)
print("\n深度优先遍历结果:")
node_names = map(lambda node: node.type, traverse_tree(python_tree))
for name in node_names:
    print(name)

# print("\n广度优先遍历结果:")
# bfs_node_names = map(lambda node: node.type, breadth_first_traverse(python_tree))
# for name in bfs_node_names:
#     print(name)


def extract_skeleton_python(code: str, reserved_lines: Set[int] = None) -> str:
    """
    将Python代码中所有函数的函数体仅保留指定行和函数级注释（文档字符串）
    
    Args:
        code: Python源代码
        reserved_lines: 需要额外保留的行号集合（相对于函数体内部的行号），默认为None
    
    Returns:
        处理后的代码
    """
    if reserved_lines is None:
        reserved_lines = set()
    
    # 解析代码
    parser = Parser(python_language)
    tree = parser.parse(bytes(code, "utf8"))
    
    # 将代码转换为可编辑的列表
    lines = code.split('\n')
    
    # 记录需要删除的行号
    lines_to_delete = []
    
    # 遍历语法树，找到所有函数定义
    for node in traverse_tree(tree):
        if node.type == 'function_definition':
            # 找到函数体
            block_node = None
            for child in node.children:
                if child.type == 'block':
                    block_node = child
                    break
            
            if block_node:
                # 获取函数体的起始和结束位置 point是一个二维坐标 (行，列)
                block_start = block_node.start_point[0]
                block_end = block_node.end_point[0]
                
                # 找到函数体内的文档字符串
                docstring_start = None
                docstring_end = None
                
                # 检查函数体内的第一个语句是否为文档字符串
                for child in block_node.children:
                    if child.type == 'expression_statement':
                        for grandchild in child.children:
                            if grandchild.type in ['string', 'identifier']:
                                docstring_start = child.start_point[0]
                                docstring_end = child.end_point[0]
                                break
                    if docstring_start is not None:
                        break
                
                # 标记需要删除的行（除了保留行、return语句和文档字符串）
                for i in range(block_start, block_end+1):
                    # 如果在文档字符串范围内，跳过
                    if docstring_start is not None and docstring_start <= i <= docstring_end:
                        continue
                    
                    # 如果在保留行中，跳过
                    if i in reserved_lines:  # 转换为函数体内的相对行号
                        continue
                    
                    lines_to_delete.append(i)
    
    # 设置为空行
    for line_num in sorted(lines_to_delete, reverse=True):
        del lines[line_num]
    
    # 重新组合代码
    return '\n'.join(lines)


def extract_skeleton_cpp(code: str, reserved_lines: Set[int] = None) -> str:
    """
    将C++代码中所有函数的函数体仅保留指定行和函数级注释（文档字符串）
    
    Args:
        code: C++源代码
        reserved_lines: 需要额外保留的行号集合（相对于函数体内部的行号），默认为None
    
    Returns:
        处理后的代码
    """
    if reserved_lines is None:
        reserved_lines = set()
    
    # 解析代码
    parser = Parser(cpp_language)
    tree = parser.parse(bytes(code, "utf8"))
    
    # 将代码转换为可编辑的列表
    lines = code.split('\n')
    
    # 记录需要删除的行号
    lines_to_delete = []
    
    # 遍历语法树，找到所有函数定义
    for node in traverse_tree(tree):
        if node.type == 'function_definition':
            # 找到函数体
            block_node = None
            for child in node.children:
                if child.type == 'compound_statement':
                    block_node = child
                    break
            
            if block_node:
                # 获取函数体的起始和结束位置 point是一个二维坐标 (行，列)
                block_start = block_node.start_point[0]
                block_end = block_node.end_point[0]
                
                # 找到函数体内的文档字符串
                docstring_start = None
                docstring_end = None
                
                # 检查函数体内的第一个语句是否为文档字符串
                # 在C++中，文档字符串通常是注释形式
                for child in block_node.children:
                    if child.type == 'comment':
                        # 检查是否为文档字符串注释
                        docstring_start = child.start_point[0]
                        docstring_end = child.end_point[0]
                        break
                
                # 标记需要删除的行（除了保留行、return语句和文档字符串）
                for i in range(block_start+1 , block_end):  # 不包括大括号行
                    # 如果在文档字符串范围内，跳过
                    if docstring_start is not None and docstring_start <= i <= docstring_end:
                        continue
                    
                    # 如果在保留行中，跳过
                    if i in reserved_lines:  # 转换为函数体内的相对行号
                        continue
                    
                    lines_to_delete.append(i)
    
    # 设置为空行
    for line_num in sorted(lines_to_delete, reverse=True):
        del lines[line_num]
    
    # 重新组合代码
    return '\n'.join(lines)


def extract_skeleton_java(code: str, reserved_lines: Set[int] = None) -> str:
    """
    将Java代码中所有函数的函数体仅保留指定行和函数级注释（文档字符串）
    
    Args:
        code: Java源代码
        reserved_lines: 需要额外保留的行号集合（相对于函数体内部的行号），默认为None
    
    Returns:
        处理后的代码
    """
    if reserved_lines is None:
        reserved_lines = set()
    
    # 解析代码
    parser = Parser(java_language)
    tree = parser.parse(bytes(code, "utf8"))
    # node_names = map(lambda node: node.type, traverse_tree( parser.parse(bytes(code, "utf8"))))
    # for name in node_names:
    #     print(name)

    # 将代码转换为可编辑的列表
    lines = code.split('\n')
    
    # 记录需要删除的行号
    lines_to_delete = []
    
    # 遍历语法树，找到所有函数定义
    for node in traverse_tree(tree):
        if node.type == 'method_declaration' or node.type == 'constructor_declaration':
            # 找到函数体
            block_node = None
            for child in node.children:
                if child.type == 'block' or child.type == 'constructor_body':
                    block_node = child
                    break
            
            if block_node:
                # 获取函数体的起始和结束位置 point是一个二维坐标 (行，列)
                block_start = block_node.start_point[0]
                block_end = block_node.end_point[0]
                
                # 找到函数体内的文档字符串
                docstring_start = None
                docstring_end = None
                
                # 检查函数体内的第一个语句是否为文档字符串
                # 在Java中，文档字符串通常是注释形式
                for child in block_node.children:
                    if child.type == 'line_comment' or child.type == 'block_comment':
                        # 检查是否为文档字符串注释
                        docstring_start = child.start_point[0]
                        docstring_end = child.end_point[0]
                        break
                
                # 标记需要删除的行（除了保留行、return语句和文档字符串）
                for i in range(block_start+1 , block_end):  # 不包括大括号行
                    # 如果在文档字符串范围内，跳过
                    if docstring_start is not None and docstring_start <= i <= docstring_end:
                        continue
                    
                    # 如果在保留行中，跳过
                    if i in reserved_lines:  # 转换为函数体内的相对行号
                        continue
                    
                    lines_to_delete.append(i)
    
    # 设置为空行
    for line_num in sorted(lines_to_delete, reverse=True):
        del lines[line_num]
    
    # 重新组合代码
    return '\n'.join(lines)


# 示例代码
        
sample_code = """
\"\"\"
模块级注释保留
\"\"\"
class A:
    pass
class B:
    \"\"\"
    类注释保留
    \"\"\"
    def __init__(self):
        \"\"\"
        魔术方法保留
        \"\"\"
        self.a=a
    # 这是#注释
    def foo(self):
        \"\"\"
        这是一个Python函数
        \"\"\"
        if bar:
            baz()
def windows(a,b):
    \"\"\"
    测试保留行
    \"\"\"
    return a+b
def sub(a,b):
    \"\"\"
    将仅仅留下注释
    \"\"\"
    return a-b
"""

print("原始代码:")
print(sample_code)

print("\n提取骨架后的代码(仅保留return语句和函数级注释):")
result1 = extract_skeleton_python(sample_code)
print(result1)

print("\n提取骨架后的代码(保留指定行和函数级注释):")
result2 = extract_skeleton_python(sample_code, {26}) # 行号从0开始
print(result2)

# 示例C++代码
sample_cpp_code = """#include <iostream>

class MyClass {
public:
    MyClass() {
        /*
        这是一个示例函数
        */
        value = 0;
    }
    
    int my_function(int x, int y) {
        // 这是一个示例函数
        int result = x + y;
        std::cout << result << std::endl;
        return result;
    }
    
    void another_function() {
        // 这是另一个函数
        return;
    }
};

int standalone_function() {
    // 独立函数
    std::cout << "Hello, World!" << std::endl;
    return 42;
}
"""

print("\n原始C++代码:")
print(sample_cpp_code)

print("\n提取骨架后的C++代码(仅保留return语句和函数级注释):")

result3 = extract_skeleton_cpp(sample_cpp_code)
print(result3)

result4 = extract_skeleton_cpp(sample_cpp_code, {8}) # 行号从0开始
print(result4)

# 示例Java代码
sample_java_code = """
class MyClass {
    public MyClass() {
        /*
        初始化函数·
        */
        int value = 0;
    }
    
    public int myFunction(int x, int y) {
        // 这是一个示例函数
        int result = x + y;
        System.out.println(result);
        return result;
    }
    
    public void anotherFunction() {
        // 这是另一个函数
        return;
    }
}

public int standaloneFunction() {
    // 独立函数
    System.out.println("Hello, World!");
    return 42;
}
"""

print("\n原始Java代码:")
print(sample_java_code)

print("\n提取骨架后的Java代码(仅保留函数级注释):")

result5 = extract_skeleton_java(sample_java_code)
print(result5)


print("\n提取骨架后的Java代码(保留指定行和函数级注释):")

result6 = extract_skeleton_java(sample_java_code, {11,12,13}) # 行号从0开始
print(result6)








