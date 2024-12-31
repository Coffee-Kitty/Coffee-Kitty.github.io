# Tire Tree

## [LCR 062. 实现 Trie (前缀树)](https://leetcode.cn/problems/QC3q1f/)

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = [None] * 26
        self.is_end = False


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self 
        for w in word:
            t = ord(w)-ord('a')
            if node.children[t] is None:
                node.children[t]=Trie()
            node=node.children[t]
        node.is_end = True
    
    def searchPrefix(self, prefix:str):
        node = self 
        for p in prefix:
            t = ord(p) - ord('a')
            if node.children[t] is None:
                return None 
            node = node.children[t]
        return node


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.searchPrefix(word) 
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.searchPrefix(prefix) 
        return node is not None


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

