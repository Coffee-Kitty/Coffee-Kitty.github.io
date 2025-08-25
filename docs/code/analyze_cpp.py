import tree_sitter_cpp, tree_sitter
from tree_sitter import Language, Parser

cpp_language = Language(tree_sitter_cpp.language())
cpp_parser = Parser(cpp_language)

code = '''
class MyClass {
public:
    void myFunction() {
        // This is a comment
        int x = 10;
        return x;
    }
};

int main() {
    MyClass obj;
    obj.myFunction();
    return 0;
}
'''

tree = cpp_parser.parse(bytes(code, "utf8"))
root = tree.root_node

print("Root node type:", root.type)
print("Root node children:")
for child in root.children:
    print(f"  - {child.type}")

print("\nTraversing tree:")
def traverse(node, depth=0):
    print("  " * depth + f"{node.type} ({node.start_point} - {node.end_point})")
    for child in node.children:
        traverse(child, depth + 1)

traverse(root)

# 查找函数定义节点
print("\n\nFinding function definitions:")
def find_function_definitions(node, depth=0):
    if node.type == "function_definition":
        print("  " * depth + f"Found function definition: {node.type}")
        print("  " * depth + f"  Start point: {node.start_point}")
        print("  " * depth + f"  End point: {node.end_point}")
        # 查找函数体
        for child in node.children:
            if child.type == "compound_statement":  # C++ 中的函数体通常是一个复合语句
                print("  " * depth + f"  Function body: {child.type}")
                print("  " * depth + f"    Body start point: {child.start_point}")
                print("  " * depth + f"    Body end point: {child.end_point}")
    for child in node.children:
        find_function_definitions(child, depth)

find_function_definitions(root)