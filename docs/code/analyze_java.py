from tree_sitter import Language, Parser, Node
import tree_sitter_java

# 初始化Java语言
java_language = Language(tree_sitter_java.language())

# 示例Java代码
java_code = """
class MyClass {
    public MyClass() {
        // 初始化函数
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

# 解析代码
parser = Parser(java_language)
tree = parser.parse(bytes(java_code, "utf8"))

# 打印语法树结构
def print_tree(node: Node, indent=0):
    print("  " * indent + f"{node.type} [{node.start_point}-{node.end_point}]")
    for child in node.children:
        print_tree(child, indent + 1)

print("Java代码的语法树结构:")
print_tree(tree.root_node)

# 查找函数定义节点
print("\n\n函数定义节点:")
def find_function_nodes(node: Node):
    if node.type == 'method_declaration':
        print(f"方法定义: {node.type} [{node.start_point}-{node.end_point}]")
        # 查找方法体
        for child in node.children:
            if child.type == 'block':
                print(f"  方法体: {child.type} [{child.start_point}-{child.end_point}]")
                # 打印方法体内的内容
                for grandchild in child.children:
                    print(f"    {grandchild.type} [{grandchild.start_point}-{grandchild.end_point}]")
    for child in node.children:
        find_function_nodes(child)

find_function_nodes(tree.root_node)