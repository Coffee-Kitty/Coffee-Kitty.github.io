# lchot100

途中发现 [B站@灵茶山艾府](https://leetcode.cn/link/?target=https%3A%2F%2Fspace.bilibili.com%2F206214) 的题解一般比较优雅。

2025.1.8

## [1. 两数之和](https://leetcode.cn/problems/two-sum/)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
            遍历到i位置，查看target-nums[i]之前是否存在
        """
        hashmap = {}
        for i,num in enumerate(nums):
            if target - num in hashmap:
                return [hashmap[target-num],i]
            hashmap[num] = i 
        return [-1,-1]
```



## [49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/)

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        排序后结果作为关键字
        """
        hashmap = {}

        for s in strs:
            key = ''.join(sorted(s))
           
            if key in hashmap:
                hashmap[key].append(s)
            else:
                hashmap[key]=[s]

        return list(hashmap.values())
```





## ** [128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/)

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        """
        首先不能排序， 排序复杂度为O(nlogn),不符合题意得O(n)

        核心思路：
            1).将nums中的数都放到哈希表中,这样可以O(1)判断数字是否在nums中
            2).对于一块连续得数字，我们只从最小得数字开始统计计数，如下：
              1，3，2，4
              我们将从1开始，向右一直数到4，而不会从2，3，4重复数数

              > 这个操作得具体实现方法是： 对于数字x， 判断x-1是否在哈希表中，只有x-1不在得情况下才去计数
        """
        nums = set(nums)
        res = 1
        for x in nums:
            if x-1 in nums:
                continue
            
            tmp = 0 #本次统计长度
            while x in nums:
                tmp += 1
                x += 1
            res =max(res, tmp) #更新值
        return res
```





## [283. 移动零](https://leetcode.cn/problems/move-zeroes/)



```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        
        > [0,1,0,3,12]
        将非0元素移动到开头，使用i记录长度， 最后其余位置填0
        """
        n = len(nums)

        i = 0
        for j in range(n):
            if nums[j] != 0:
                nums[i] = nums[j]
                i += 1
        
        for k in range(i,n):
            nums[k] = 0
```



## [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        """
            对于 i，k 两根柱子(i从0开始递增， k从n-1开始递减)，
            假设height[i] < height[k]
                由于较短的柱子决定了容器容纳水的高度，

                此时，只有 i+1处柱子 高于 i才有可能获得更多的水量

                应该移动i到i+1

                即两段谁短挪动谁
            假设height[i] == height[k]
                不管挪动谁， 宽度减小， 而容器高度上限不可能变大
                容器容纳水的量一定不会更多，
                即相同长度挪动谁都行
            基于这样的思路遍历一遍 
        """
        n = len(height)
        i,k = 0,n-1
        res = 0
        while i<k:
            tmp = (k-i) * min(height[i], height[k])
            res = max(res,tmp)
            if height[i]<=height[k]:
                i+=1
            else:
                k-=1
        return res
```





## ** [15. 三数之和](https://leetcode.cn/problems/3sum/) 



```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
            固定i，问题变为是否存在 两数之和为-nums[i]

            ---------------------------------------------

            理想情况下，如上述所言

            然而， 本题要求返回的是 元素而 **不是索引下标**，
                  要求返回元素，不能包含重复三元组

                  这里去重很复杂，所以考虑使用 排序 + 双指针
        """
        # 排序
        nums = sorted(nums)

        n = len(nums)
        res = []
        for i in range(n):
            # 相同的元素 不能再次作为 i 元素
            if i>=1 and nums[i]==nums[i-1]:
                continue

            j,k = i+1, n-1
            while j<k:

                if nums[j]+nums[k] == -nums[i]:
                    res.append([nums[i],nums[j],nums[k]])
                   
                    # 找到答案后再去重
                    while j<k and nums[j]==nums[j+1]:
                        j+=1
                    while k>j and nums[k]==nums[k-1]:
                        k-=1
                    # 双指针收缩
                    j,k = j+1, k-1
                elif nums[j]+nums[k] < -nums[i]:
                    j+=1
                else:
                    k-=1

        return res
```



## [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        """
            雨水量 一种是固定宽度，一种是固定高度

            这里选用固定宽度为1， 算每一个位置他能聚拢水的最大高度
            对于位置i而言，其最大聚拢水的高度为，左侧柱子最大高度和右侧柱子最大高度的较小值 - 当前柱子高度
            因此，适宜采用前后缀分解的方法来处理
        """
        res = 0 

        n = len(height)
        # prefix = [0] *n # prefix可以边用变算
        suffix = [height[n-1]] * n # sufix[i]表示 从i到n最高的柱子高度
        for i in range(n-2,-1,-1):
            suffix[i] = max(suffix[i+1], height[i])
        
        prefix = height[0]
        for i in range(1,n):#边缘柱子肯定不能抬水
            prefix = max(prefix, height[i])

            tmp = min(prefix, suffix[i]) - height[i]

            res += tmp
        
        return res
```





## [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
            子串是连续的， 非负的
            可以尝试使用滑动窗口
        """

        l,r = -1,-1
        hashmap = {}
        n = len(s)
        res = 0
        for r in range(n):
            # 入窗口
            c=s[r]
            hashmap[c]= hashmap.get(c,0) +1

            # 收缩左侧直至满足条件
            # 注意为 左开右闭区间 (l,r]
            while hashmap[c] > 1:
                l+=1 
                hashmap[s[l]]-=1

            # 更新答案
            res = max(res, r-l)
        return res

```



## **[438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        """
            emm, 定长滑动窗口
        """
        ans = []
        cnt = Counter(p)
        cnt_s = Counter()
        left =  -1 # 左开右闭区间
        for right, c in enumerate(s):
            # 右端点字母进入窗口
            cnt_s[c] += 1
            if right-left < len(p):
                continue
            
            # 更新答案
            if cnt == cnt_s:
                ans.append(left+1)

            # 左端点字母离开窗口
            left += 1
            cnt_s[s[left]] -= 1
        return ans
```



## *[560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/)

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        """
            前缀和+哈希

            s[i]为nums的前缀和

            到位置j时，需要知道是否存在i，使得s[j]-s[i] == k
            使用哈希表后，也就是说 s[j]-k 是否存在哈希表中

            如果说s[j] == k 呢， 所以说要把 0:1存储一下
        """

        n=len(nums)
        ans = 0
        pre = 0
        hashamp = {0:1}
        for num in nums:
            pre+=num # 前缀和 

            if pre - k in hashamp:
                ans += hashamp[pre-k]

            hashamp[pre]= hashamp.get(pre,0)+1
        return ans

```



## *[239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

```python
class Solution:

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
            [1  3  -1] -3  5  3  6  7       3
            1 [3  -1  -3] 5  3  6  7       3
            1  3 [-1  -3  5] 3  6  7       5
            1  3  -1 [-3  5  3] 6  7       5
            1  3  -1  -3 [5  3  6] 7       6
            1  3  -1  -3  5 [3  6  7]      7

            定长窗口中 ， 观察上图中 第3行
            当5出现后， 窗口中所有比5小的值，可以被5消灭掉， 因为5更大，并且更靠后
            由此使用 单调递减序列  序列头为最大值
        """
        q = deque([])
        def push(q, r):
            while q and nums[q[-1]] <= nums[r]:
                q.pop()
            q.append(r)

        l,r = -1,-1
        ans =[]
        for r,num in enumerate(nums):
            push(q, r)
            # 预启动
            if r<k-1:
                continue

            ans.append(nums[q[0]])

            # 后处理窗口值
            l+=1
            if q[0] == l:
                q.popleft()
        
        return ans


```



## **[76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)



可以详细比对本题 和 [438. 找到字符串中所有字母异位词]的差异

体会定长滑动窗口 和 不定长滑动窗口的细微差别。

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        """
            不定长滑动窗口
            r向右拓展，直至覆盖子串t，
            此时收缩l，同时更新答案字符串，直到(l,r]不再覆盖
        """

        l,r = -1,-1
        cnt_t = Counter(t)
        cnt_s = Counter()
        ans = ""
        for r,ch in enumerate(s):
            # r向右拓展
            if ch in cnt_t:
                cnt_s[ch] +=1 

            while cnt_s >= cnt_t:
                tmp = s[l+1:r+1]
                
                ans = tmp if len(tmp) < len(ans) or ans=="" else ans

                l+=1
                if s[l] in cnt_t:
                    cnt_s[s[l]] -= 1
        return ans
```



## [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)



```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        """
            存在负数，不好考虑滑动窗口

            考虑前缀和 O(n*2)

            考虑dp
                dp[i]表示 以nums[i]结尾的最大和连续子数组
                dp[i+1] == max {dp[i]+nums[i+1] , nums[i+1]}

        """
        n = len(nums)
        dp = 0
        ans = -math.inf
        for num in nums:
            dp = max(dp+num, num)

            ans = max(ans, dp)

        return ans
```



## **[56. 合并区间](https://leetcode.cn/problems/merge-intervals/)

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
            按照左端点排序，从而去拓展右端点
        """
        intervals = sorted(intervals, key = lambda x: x[0])

        ans = []
        l,r=intervals[0]

        for i, interval in enumerate(intervals[1:]):
            if r>= interval[0]:
                r = max(r,interval[1])
                continue
            
            ans.append([l,r])
            l,r = interval
        ans.append([l,r])
        return ans
```



## **[189. 轮转数组](https://leetcode.cn/problems/rotate-array/)

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        >>> 神奇解法
        >>> 12345  k=2
        反转整个数组为 54321
        反转前k个元素 45321， 再反转前n-k个元素 45123

        还应注意：轮转k次等于轮转k%n次
        """
        if k==0:
            return 

        def reverse(i,j):
            while i<j:
                nums[i], nums[j] = nums[j],nums[i]
                i,j = i+1,j-1
        
        n=len(nums)
        k %= n # 轮转k次等于轮转k%n次
        reverse(0,n-1)
        reverse(0,k-1)
        reverse(k,n-1)
```



## [238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/)

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """
            根据题目提示，使用前后缀做
            pre[i]表示0-i的所有乘积，类推suffix[i]
        """
        pre = 1
        n= len(nums)
        suffix = [1]*(n+1)
        suffix[n-1] = nums[n-1]
        for i in range(n-2,-1,-1):
            suffix[i]=suffix[i+1]*nums[i]

        answer = []
        for i in range(n):
            answer.append(pre * suffix[i+1])

            pre = pre*nums[i]
        return answer
```



## **[41. 缺失的第一个正数](https://leetcode.cn/problems/first-missing-positive/)

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        """
        核心点： 长度为N的数组，没有出现过的最小正整数 一定在[1,N+1]中

        恰好将1，N这个数组存储状态来用

        首先将所有负数包括0变为N+1
        然后
        如果x为正数，将 x-1下标即 数组第x位置的数置为负数, 即打上标记
        如果x是负数， 将 |x|-1下标打上标记
        注意： 标记不要重复打
        最后从0到i如果都被打上标记，即数组元素为负数，则答案为N+1，否则为第一个没打上标记位置t+1

        """

        # 1.所有负数变为N+1
        n = len(nums)
        for i,num in enumerate(nums):
            if num <=0:
                nums[i]=n+1

        #2.遍历数组并且打标记
        for num in nums:
           num = abs(num)
           if num<=n:
            nums[num-1]= -abs(nums[num-1])
            
        # 最后查找答案
        for i,num in enumerate(nums):
            if num > 0:
                return i+1
        return n+1


```



## **[73. 矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/)

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.

        '''
        利用第0行和第0列作为标志
        1.首先考虑第一行第一列对自身的影响，
            考虑第一行第一列是否最后都会变为0
        
        2. 将除了第一行第一列的元素 的影响 转移到第一行第一列
            遍历除了第一行和第一列的元素，如果该元素等于0
            那么对应的第0行和第0列的元素置为0
        3. 再根据第一行第一列，对除了第一行第一列的元素进行影响
            置0，遍历除了第一行和第一列的元素，如果对应的
            第0行和第0列元素为0，那么当前元素置为0
        4. 最后别忘记 第一行第一列对自身影响
        
        时间复杂度O(m*n)
        空间复杂度O(1)
        '''
        """

        m,n =len(matrix), len(matrix[0])
        row, col = False, False
        for i in range(m):
            if matrix[i][0]==0:
                row = True
        for j in range(n):
            if matrix[0][j]==0:
                col = True

        # 遍历除了第一行第一列的元素，若x[i,j]为0，将x[i,0], x[0,j]置为0
        for i in range(1,m):
            for j in range(1,n):
                if matrix[i][j]==0:
                    matrix[i][0]=matrix[0][j]=0
        # 再次遍历除了第一行第一列的元素，若x[i,0]或者x[0,j]为0 则将x[i,j]置为0
        for i in range(1,m):
            for j in range(1,n):
                
                if  matrix[i][0]==0 or matrix[0][j]==0:
                    matrix[i][j]=0
        
        if row:
            for i in range(m):
                matrix[i][0]=0
        if col:
            for j in range(n):
                matrix[0][j]=0
        
```





## **[54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)

不考虑时间复杂度的一种抽象做法

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """
        m,n

        (0,0) - > (0,n-1) -> (m-1,n-1) -> (m-1,0) #左闭右开区间
        (1,1) -> (1, n-2) ->(m-2,n-2) ->(m-2,1)
        """
        ans = []
        while matrix:
            ans += matrix.pop(0)
            matrix = list(zip(*matrix))[::-1]

        return ans
```



之前写法太离谱了，按照图来做吧

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """
        根据题意，我们从左上角 (0,0) 出发，按照「右下左上」的顺序前进：

        首先向右走，如果到达矩阵边界，则向右转 90 
        ∘
        ，前进方向变为向下。
        然后向下走，如果到达矩阵边界，则向右转 90 
        ∘
        ，前进方向变为向左。
        然后向左走，如果到达矩阵边界，则向右转 90 
        ∘
        ，前进方向变为向上。
        然后向上走，先从 7 走到 4，然后从 4 准备向上走，但上面的 1 是一个已经访问过的数字，那么向右转 90 
        ∘
        ，前进方向变为向右。
        重复上述过程，直到答案的长度为 mn。
        """
        DIRS = [(0,1),(1,0),(0,-1),(-1,0)]
        m,n = len(matrix), len(matrix[0])
        ans = []
        i=j=di=0
        for _ in range(m*n): #一共走mn步
            ans.append(matrix[i][j]) # 表示已经访问过

            matrix[i][j]=None
            x,y = i+DIRS[di][0], j+DIRS[di][1]
            if x<0 or x>=m or y<0 or y>=n or matrix[x][y] is None:
                # 如果下一个位置 越界或者已经访问过，转向
                di = (di+1)%4
            i,j = i+DIRS[di][0], j+DIRS[di][1]

        return ans
```



## [48. 旋转图像](https://leetcode.cn/problems/rotate-image/)

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        非正经解法
        matrix[::] = [ row[::-1] for row in zip(*matrix)]
        
        下面是普通解法：
        1.转置
        m[i,j]=m[j,i]
        2.反转每一行

        """
        for i in range(len(matrix)):
            for j in range(i):
                matrix[i][j],matrix[j][i] = matrix[j][i], matrix[i][j]
        
        for i in range(len(matrix)):
            j,k = 0, len(matrix[0])-1
            while j<k:
                matrix[i][j], matrix[i][k]= matrix[i][k], matrix[i][j]
                j,k =j+1,k-1
```



## **[240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        从右上角开始找
         1 3 4 5
         2 6 9 10
         3 8 12 15 

         假设要找6，
         5<6，则第一行可以被排除
         2 6 9 10
         3 8 12 15
         6<10， 则 最后一列可以被排除
        """
        m,n = len(matrix), len(matrix[0])
        i,j = 0,n-1
        while i<m and j>=0:
            if matrix[i][j]<target:
                i+=1
            elif matrix[i][j]>target:
                j-=1
            else:
                return True
        return False
```



## **[160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        p,q = headA, headB
        while p is not q:
            p = p.next if p else headB
            q = q.next if q else headA
        return p
```





![lc160-3-c.png](../picture.asset/1729473968-TLOxoH-lc160-3-c.png)



## [206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummyHead = ListNode(-1)

        # 头插法
        p =head
        q = dummyHead
        def appendToHead(headnode, node):
            node.next = headnode.next
            headnode.next = node

        while p:
            tmp = p.next  # 注意提取保存 p.next
            appendToHead(q, p)
            p = tmp
        return dummyHead.next

```

## **[234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/)

 ```python
 # Definition for singly-linked list.
 # class ListNode:
 #     def __init__(self, val=0, next=None):
 #         self.val = val
 #         self.next = next
 class Solution:
     def middleNode(self, head: Optional[ListNode]) -> bool:
         slow,fast = head, head
         while fast.next and fast.next.next:
             slow, fast = slow.next, fast.next.next
         if fast.next:
             slow=slow.next
         
         return slow
 
     def reverseList(self, head: Optional[ListNode]) -> bool:
         dummyhead = ListNode(-1)
         p = head
         while p:
             nxt = p.next 
             p.next = dummyhead.next
             dummyhead.next = p 
             p = nxt
         return dummyhead.next
 
 
     def isPalindrome(self, head: Optional[ListNode]) -> bool:
         """
         时间复杂度O(n),空间复杂度O(1)
         则反转后半部分的链表，然后与前半部分对比即可
         1,2,3,2,1   奇数指向最中间
         1,2,3,3,2,1 偶数指向中间偏右侧
         """
         middle = self.middleNode(head)
         reverse_node = self.reverseList(middle)
        
         p,q = head, reverse_node
         while p and q:
             if p.val != q.val:
                 return False 
             p,q = p.next, q.next
         return True
 
 
 ```



## [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """
            快慢指针， 如果能够相遇则有环，否则无环
        """
        if head is None:
            return False
        slow, fast = head, head 
        while fast.next and fast.next.next:
            slow , fast = slow.next , fast.next.next 

            if slow is fast:
                return True
        return False
```



## [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        list    3 2 0 -4
        pos:      1 
            设 3  长度x， 2，0，-4部分长度
            假设快慢指针初次相遇点 为0,   2，0部分长度设置为y， -4部分长度为z

            2(x+y) = x+y+k(z+y)
            即x = z + k2(z+y)

            所以，指针从起点出发走x步  与 指针从上面的相遇点出发走 z+k2(z+y)步相同
            即为两者初次相遇点
        """
        if head is None:
            return None 
        slow,fast = head, head

        des = None
        while fast.next and fast.next.next:
            slow, fast = slow.next , fast.next.next

            if slow is fast:
                des = slow
                break
        if des is None:
            # 如果无环
            return None 

        p1, p2 = head, des 
        while p1 and p2:
            if p1 is p2:
                return p1 
            p1 ,p2 = p1.next, p2.next 
        return None

```



## [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
            类似归并排序的思想即可
        """
        p1,p2 = list1,list2
        dummy = ListNode(-1)
        p = dummy
        while p1 and p2:
            if p1.val <= p2.val:
                p.next = p1 
                p = p.next 
                p1 = p1.next 
            else:
                p.next = p2
                p = p.next 
                p2 = p2.next 

        if p1: 
            p.next = p1 
        elif p2:
            p.next = p2

        return dummy.next        
```

## [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        
        p1, p2 =l1, l2
        carry = 0
        dummy = ListNode(-1)
        p3=dummy
        while p1 or p2:
            if p1:
                carry += p1.val
                p1=p1.next 
            if p2:
                carry += p2.val
                p2=p2.next
            
            tmp = ListNode(carry%10)
            p3.next = tmp 
            p3 = p3.next 

            carry//=10
            

        while carry>0:
            tmp = ListNode(carry%10)
            p3.next = tmp 
            p3 = p3.next 
            carry//=10
        return dummy.next

```



## **[19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
            简简单单，p1,p2指针， p2指针指向头节点， 然后p2先走n步
            p1,p2同时向后走，p2为null时，p1恰好为导数第n个节点
            p2.next 为null时，p1为倒数第n+1个节点
            需要格外注意：由于可能会删除链表头部，用哨兵节点简化代码
        """
        # 由于可能会删除链表头部，用哨兵节点简化代码
        dummy = ListNode(next=head)
        p1, p2= dummy,dummy
       
        for _ in range(n):# 现在的倒数第n个节点
            p2 = p2.next 
        
        while p2.next:
            pre = p1
            p1, p2 = p1.next, p2.next 
        
        p1.next =p1.next.next 
        return dummy.next
```



## [24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        def swapNodes(pre, nod1, nod2):
            nxt = nod2.next 
            pre.next = nod2 
            nod2.next = nod1
            nod1.next = nxt 
        
        dummy = ListNode(next=head)

        p = head 
        pre = dummy
        while p and p.next:

            swapNodes(pre, p, p.next)
            
            # 这里需要注意  p指向的节点已经被反转了
            pre, p = pre.next.next , p.next

        return dummy.next
```



## [25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if k == 1:
            return head

        
        def reverseK(node):
            # 使用头插法反转node
            dummy = ListNode(next=None)

            p,q = node,dummy 
            while p:
                tmp = p.next 
                p.next = q.next 
                q.next = p 

                p=tmp 
            return dummy.next, node 
        
        dummy = ListNode(next = head)
        p = dummy
        q = p
        while q:
            for _ in range(k):
                q=q.next if q else None 
            if q is None:
                break
            else:
                nxt = q.next 
                q.next = None 
                he, ta  = reverseK(p.next)
                p.next,ta.next = he,nxt 

                # 持续循环下去
                p,q= ta, ta
        return dummy.next 
```



## *[138. 随机链表的复制](https://leetcode.cn/problems/copy-list-with-random-pointer/)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        """
            先仅仅考虑next赋值，同时用hash记录对应关系，最后再过一遍给random赋值
        """
        hashmap = {None: None}  
        dummy = Node(x=-1)

        h, d = head, dummy 
        while h:
            t = Node(x=h.val)
            d.next=t 
			# 使用hash表记录对应位置的 节点对应关系
            hashmap[h] = t
            h,d = h.next, d.next
        
        h,d = head,dummy.next 
        while h:
            # 再次遍历时，赋值random指针
            d.random=hashmap[h.random]# 注意这里是指向h.random
            h,d = h.next,d.next
        return dummy.next
```

## [148. 排序链表](https://leetcode.cn/problems/sort-list/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
            选用归并排序

        """

        def merge(nod1: ListNode,nod2: ListNode):
            """
                将nod1和nod2按从小到大的值顺序合并
            """
            dummy = ListNode(val=-1)
            p3 = dummy
            p1, p2 = nod1, nod2
            while p1 and p2:
                if p1.val <= p2.val:
                    p3.next = p1
                    p1 = p1.next
                else:
                    p3.next = p2 
                    p2 = p2.next 
                p3=p3.next
            if p1:
                p3.next = p1 
            elif p2:
                p3.next = p2 
            
            return dummy.next

        def middleNode(nod: ListNode):
            """
                返回中间的节点：快慢指针的方法
                1,2,3,4
                偶数返回中间靠后的节点，如上返回3的节点
            """
            pre = None 
            slow, fast = nod, nod
            while fast and fast.next:
                pre = slow
                slow,fast = slow.next, fast.next.next
            
            # 这里要断链处理的
            pre.next = None 
            return slow
        
        def mergeSort(nod: ListNode):
            """
                递归排序左右两部分节点
            """
            if nod is None or nod.next is None:
                return nod

            middle = middleNode(nod)

            p1, p2 = nod, middle

            p1, p2 = mergeSort(p1), mergeSort(p2)
            return merge(p1, p2)
        
        return mergeSort(head)
```

## *[23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
            基本思路： 两两合并
        """

        def merge(nod1: ListNode, nod2: ListNode):
            dummy = ListNode(val=-1)# 虚拟头节点， 便于处理 头为None的链表

            p1,p2 = nod1, nod2 
            p3 = dummy
            while p1 and p2:
                if p1.val <= p2.val:
                    p3.next = p1
                    p1, p3 = p1.next, p3.next 
                else:
                    p3.next = p2
                    p2, p3 = p2.next, p3.next 
            if p1:
                p3.next = p1 
            if p2:
                p3.next = p2 
            return dummy.next 
        while len(lists) > 1:
            tmp = []
            i,j = 0, len(lists)-1
            while i<j:
                p1 ,p2 = lists[i], lists[j]
                p = merge(p1,p2)
                tmp.append(p)
                i,j =i+1,j-1
            if i==j:
                tmp.append(lists[i])
            lists = tmp
        return lists[0] if lists else None
```



## [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)



```python
class Node:
    def __init__(self, key,val, pre = None, nxt= None):
        self.key = key
        self.val = val 
        self.next = nxt
        self.pre = pre

def deleteNode(nod):
    nod.pre.next = nod.next 
    nod.next.pre = nod.pre 
 
def insertToHead(head,nod):
    nod.next = head.next 
    head.next = nod 
    nod.pre = head 
    nod.next.pre = nod 

def moveTOhead(head, nod):
    deleteNode(nod)
    insertToHead(head,nod)



class LRUCache:
    """
        使用双链表存储，便于将目标节点提到队头，从链表头到链表尾依次越来越不常用

        此外，为满足get为O(1)复杂度， 再使用hash表记录key及其对应节点
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dummy = Node(key=-1,val=-1) 
        self.tail = Node(key=-1,val=-1)
        self.dummy.next, self.tail.pre = self.tail, self.dummy
        self.hashmap = {}

    def get(self, key: int) -> int:
        if key in self.hashmap:
            # 返回前需要将该节点添加到链表头
            nod = self.hashmap[key]
            moveTOhead(self.dummy, nod)
            return nod.val
        return -1

    def put(self, key: int, value: int) -> None:
        # 如果关键字已经存在
        if key in self.hashmap:
            nod = self.hashmap[key]
            nod.val = value  #变更其值
            moveTOhead(self.dummy, nod)
        else:
            # 如果关键字不存在，插入关键字 头插
            nod = Node(key=key,val=value)
            insertToHead(self.dummy, nod)
            # 注意用hash记录
            self.hashmap[key] = nod

            # 如果lru容量过大，除掉队尾 tail前的元素即可
            if len(self.hashmap) > self.capacity:
                tail_nod = self.tail.pre 
                deleteNode(tail_nod)
                # 注意从hash表中也删掉
                del self.hashmap[tail_nod.key]
                del tail_nod


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```



-----

二叉树

## [二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        ans = []
        ans += self.inorderTraversal(root.left)
        ans.append(root.val)
        ans += self.inorderTraversal(root.right)
        return ans
        
```

## [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
            二叉树问题天然具有 子问题的递归特性
        """
        if root is None:
            return 0
        return 1+max( self.maxDepth(root.left), self.maxDepth(root.right)) 

```

## [226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
            翻转二叉树 即翻转左右节点

            1.递归翻转左右子树
            2.左右子树互换
        """
        if root is None:
            return None 
        
        lef,rig = self.invertTree(root.left), self.invertTree(root.right)

        root.left , root.right= rig, lef 
        return root 
        
```

## ** [101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/)

> 这个递归处理的很神奇

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """
            方法一： 层序遍历 每层是否对称

            方法二： 翻转二叉树，然后逐一检查，

            方法三：正儿八经考虑 子问题分解的事情

                判断一棵树是否是轴对称，即
                1.左右子树根节点是否相同，
                2.左子树的左侧对应右子树的右孩子， 左子树的右孩子对应右子树的左孩子，
                    这里是递归处理
        """
        def isSymm(p1, p2):
            # 首先p1,p2的值相同
            if p1 is None:
                return p2 is None 
            if p2 is None:
                return p1 is None
            
            return p1.val == p2.val and isSymm(p1.left, p2.right) and isSymm(p1.right, p2.left)
        
        return isSymm(root.left, root.right)       
```

## [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        """
            枚举经过每一个节点的最长路径包含节点的数目
            第i个节点的直径 = left节点深度 + 1 + rigth节点深度
        """
        ans = 1 # dfs的同时更新直径
        def dfs(root):
            """
                dfs返回该节点的高度， 遍历过程中利用高度更新ans
            """
            if root is None:
                return 0
            
            lef, rig = dfs(root.left), dfs(root.right)
            nonlocal ans
            ans = max(ans, lef+rig+1)
            return max(lef,rig) +1 
        dfs(root)
        return ans-1 # 直径是任意两个节点之间最长路径的 长度， 即节点数目-1
            
```



## [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        	层序遍历可以选择不使用队列
        """
        if root is None:
            return []
        ans = []
        layer = [root]
        while layer:
            ans_layer = []
            new_layer = []
            
            for nod in layer:
                if nod.left:
                    new_layer.append(nod.left) 
                if nod.right:
                    new_layer.append(nod.right) 
                ans_layer.append(nod.val)
            layer = new_layer
            ans.append(ans_layer)
        return ans
```



## [108. 将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        """
            这里选择emm 中点构建即
                每次选择数组中点作为根节点，然后递归构建左右子树，最后返回根节点
        """
        def middleNode(nums: List[int]):
            if len(nums)==0:
                return None 
            
            mid = len(nums) // 2
            root = TreeNode(val= nums[mid])
            root.left, root.right = middleNode(nums[:mid]), middleNode(nums[mid+1:])
            return root 
        return middleNode(nums)
```



## **[98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """
            可以这样考虑从而确定递归关系

            假设左子树、右子树已经是二叉搜索树，此时需要满足什么条件该棵树是二叉搜索树
            即左子树最大值 小于根节点 右子树最小值 大于根节点
        """  
        
        def dfs(root: Optional[TreeNode]) -> Tuple:
            """
                搜索root是否为二叉搜索树，
                同时最后返回该根节点下最大值和最小值
                (最小值，最大值)
            """
            if root is None:
                return inf, -inf

            lmin,lmax= dfs(root.left)
            rmin,rmax = dfs(root.right)

            if lmax < root.val < rmin:
                return min(lmin,root.val), max(rmax, root.val)
            return -inf, inf
        return dfs(root)[1] != inf


```

下面的先序遍历可能更好理解一些

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode],left=-inf,right=inf) -> bool:
        if root is None:
            return True 
        
        x = root.val 
        return left<x<right and \
                self.isValidBST(root.left, left=left, right=x) and \
                self.isValidBST(root.right,left=x,right=right)
```

## *[230. 二叉搜索树中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)



最简单的方法是遍历一遍

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        """
            二叉搜索树中序遍历是有序的，此时做一个计数操作试试
        """
        lists = []
        def dfs(nod):
            """
                返回节点数目， 中序遍历
            """
            if nod is None:
                return
            
            lef = dfs(nod.left)
            nonlocal lists
            lists.append(nod.val)
            rig = dfs(nod.right)
            return
        dfs(root)

        return lists[k-1]
```



## [199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)

层序遍历的较为容易理解

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        ans = []
        layer= [root]
        while layer:
            new_layer=[]
            for nod in layer:
                if nod.left:
                    new_layer.append(nod.left)
                if nod.right:
                    new_layer.append(nod.right)
            ans.append(layer[-1].val)
            layer = new_layer
        return ans
```



递归的 则需要明确一个关键点， 即某个深度首先到达的加入ans并且先递归右子树

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        depth = 0
        def dfs(nod,d):
            """
                先右子树，再递归左子树，这个深度首次达到，则添加进ans
            """
            if nod is None:
                return 
            nonlocal depth
            if d==depth:
                ans.append(nod.val)
                depth += 1
            dfs(nod.right, d+1)
            dfs(nod.left, d+1)
        dfs(root,0)
        return ans
```

## [114. 二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        考虑将一颗二叉树先序展开为链表，
            其递归左子树返回的链表，递归右子树返回的链表
        其链接顺序为 根、左子树链表、右子树链表
        """

        def dfs(nod):
            """
                这里设计为返会链表 头节点和尾节点
            """
            if nod is None:
                return None, None 
            root = nod
            lhead,ltail = dfs(nod.left)
            rhead,rtail = dfs(nod.right)
            
            nod.left = None
            head, tail = nod, nod 
            if lhead and ltail:
                tail.right = lhead 
                tail = ltail 
            if rhead and rtail:
                tail.right = rhead
                tail = rtail
            return head, tail
        return dfs(root)[0]
```



## [105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
            基本思路就是， preorder首位元素为 根节点，
             依据这个根节点在inorder中找到该根节点， 以此一分为二，左侧为left子树，右侧为right子树，
             然后递归构造左右子树
        """

        if not preorder:
            return None 
        
        root = preorder[0]
        idx = inorder.index(root) # 这一步很麻烦
        inlef, inrig = inorder[:idx], inorder[idx+1:]
        prelef, prerig = preorder[1:1+len(inlef)], preorder[1+len(inlef):]
        lef_nod, rig_nod = self.buildTree(prelef,inlef), self.buildTree(prerig,inrig)

        root=TreeNode(val = root)
        root.left, root.right = lef_nod, rig_nod
        return root
```



## **[437. 路径总和 III](https://leetcode.cn/problems/path-sum-iii/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        """
            前缀和 + 哈希表  做法令人震惊

            类似于 数组上的题： 和为 K 的子数组

            在二叉树上，前缀和即为从根节点开始到达该节点的路径和，
                用哈希表cnt统计前缀和出现次数， 然后此时寻找 [前缀和s-targetSum]在哈希表中是否出现即可
                同样的，初始化需要 {0:1}, （这是为了弥补 s == targetSum的情况） 
        """
        ans = 0 # 记录答案
        cnt = defaultdict(int)
        cnt[0]=1
        """
            想把任意路径和都表示成两个前缀和的差，必须添加一个 0，否则当路径是前缀时（从根节点开始的路径），没法减去一个数
        """
        def dfs(nod, path:int):
            # 属于前序遍历中操作
            if nod is None:
                return 
            path += nod.val 
            
            nonlocal ans
            ans += cnt[path-targetSum] if path-targetSum in cnt else 0 

            cnt[path] += 1 # 先更新ans， 再更新cnt
            """
                如果 targetSum=0，假设根节点值为 1，如果先把 cnt[1] 增加 1，再把 ans 增加 cnt[s−targetSum]=cnt[1]=1，就相当于我们找到了一条和为 targetSum=0 的路径，但和为 0 的路径是不存在的。另一种理解方式是，空路径的元素和等于 0，我们把这个 0 当作了符合要求的路径，但题目要统计的是非空路径。
            """
            dfs(nod.left, path)
            dfs(nod.right, path)
            cnt[path] -= 1 # 恢复现场
            """
            如果不恢复现场，当我们递归完左子树，要递归右子树时，cnt 中还保存着左子树的数据。但递归到右子树，要计算的路径并不涉及到左子树的任何节点，如果不恢复现场，cnt 中统计的前缀和个数会更多，我们算出来的答案可能比正确答案更大。
            """

        dfs(root,0)
        return ans
```

## [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)



```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
            无外乎三种情况 
            p 为一个根节点，q为其孩子
            同上不过q为根节点，p为孩子
            一个根节点，p，q为其左右孩子

            采用后序遍历，
                函数dfs 返回节点是否包含p或q，包含则返回p或q，否则返回None
        """
        ans = None 

        def dfs(nod):

            if nod is None:
                return None 
            
            lef, rig = dfs(nod.left), dfs(nod.right)
            nonlocal ans 
            #1.p 为一个根节点，q为其孩子
            if p==nod and q in [lef,rig]:
                ans = p 
                return None # 终结了
            # 2.
            elif q ==nod and p in [lef,rig]:
                ans = q 
                return None
            # 3.
            elif p in [lef,rig] and q in [lef,rig]:
                ans = nod 
                return None
            else:
                if p in [lef,rig,nod]:
                    return p 
                if q in [lef,rig,nod]:
                    return q
                return None
        dfs(root)
        return ans
            

```



## [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        """
            依据路径定义， 想法是遍历拐点

            后序遍历，然后对于每一个点，获取其左右子树的最大垂直路径和必须连续到底，考虑是否在该拐点处相接

        """
        ans = -inf
        def dfs(nod):
            """
            返回该根节点的最大垂直路径和
                同时更新答案ans查看总的最大路径和
            """
            if nod is None:
                return -1000
            lef,rig = dfs(nod.left), dfs(nod.right)
            
            nonlocal ans 
            ans = max(ans, lef, lef+nod.val, lef+nod.val+rig, nod.val+rig, rig, nod.val) 

            return max(nod.val + lef, nod.val + rig, nod.val)
        dfs(root)
        return ans
```

上面写法不太优雅：

下面是正解

```python
链：从下面的某个节点（不一定是叶子）到当前节点的路径。把这条链的节点值之和，作为 dfs 的返回值。如果节点值之和是负数，则返回 0。
直径：等价于由两条（或者一条）链拼成的路径。我们枚举每个 node，假设直径在这里「拐弯」，也就是计算由左右两条从下面的某个节点（不一定是叶子）到 node 的链的节点值之和，去更新答案的最大值。
⚠注意：dfs 返回的是链的节点值之和，不是直径的节点值之和。


class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        ans = -inf
        def dfs(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0  # 没有节点，和为 0
            l_val = dfs(node.left)  # 左子树最大链和
            r_val = dfs(node.right)  # 右子树最大链和
            nonlocal ans
            ans = max(ans, l_val + r_val + node.val)  # 两条链拼成路径
            return max(max(l_val, r_val) + node.val, 0)  # 当前子树最大链和（注意这里和 0 取最大值了）
        dfs(root)
        return ans
```



--------

图论

## [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
            遍历每一个岛屿，遍历时将相邻岛屿置-1，最后统计遍历到的数量即可
        """
        m,n = len(grid), len(grid[0])
        ans = 0
        Dir = [[-1,0],[1,0],[0,-1],[0,1]]
        def dfs(grid, i,j):

            grid[i][j] = -1 #当前岛屿置-1
            for x,y in Dir:
                nx,ny = x+i, y+j
                if 0<=nx<m and 0<=ny<n and grid[nx][ny] == "1":
                    dfs(grid,nx,ny)
        cnt = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    cnt += 1 
                    dfs(grid,i,j)
        return cnt
        
```



## [994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        """
            很容易想到bfs， 然后用队列（这里可以直接用list代替）收集所有的腐烂橘子，
            更新新鲜橘子为腐烂橘子，直到不能更新为止
        """

        Dir = [[-1,0],[1,0],[0,1],[0,-1]]
        m,n = len(grid), len(grid[0])
        ans = -1

        cnt = 0


        q = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    q.append((i,j))
        
        while q:
            new_q = []
            for (i,j) in q:
                for dx,dy in Dir:
                    nx,ny = dx+i, dy+j
                    if 0<=nx<m and 0<=ny<n and grid[nx][ny] == 1:
                        grid[nx][ny] = 2# 变成腐烂橘子
                        new_q.append((nx,ny))
            q=new_q
            cnt += 1 if new_q else 0
   
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    cnt = -1
        ans = cnt
        return ans

```



## [207. 课程表](https://leetcode.cn/problems/course-schedule/)

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        ind, outd = [0]*numCourses, [0]*numCourses # 入度，出度
        for cour, precour in prerequisites:
            ind[cour]  +=1
            # outd[precour] += 1
        adj = [[] for _ in range(numCourses)] # 邻接表
        for c, pre in prerequisites: 
            adj[pre].append(c)

        q = [ i for i,cour in enumerate(ind) if cour==0]
        
        while q:
            new_q = []
            for cour in q:
                for adjacent in adj[cour]:
                    ind[adjacent] -= 1
                    if ind[adjacent] == 0:
                        new_q.append(adjacent)
            q = new_q
        
        for ind_cour in ind:
            if ind_cour != 0:
                return False
        return True
```



## [208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/)

```python
class Node:
    __slots__ = 'son', 'end'

    def __init__(self):
        self.son = {}
        self.end = False

class Trie:
    """
        对于Trie前缀树，每个节点有26个节点与 a-z一一对应即可， 如果还有下一个字母，则对应构造一个节点，否则为None
    """
    def __init__(self):
        self.root = Node()

    def insert(self, word: str) -> None:
        p = self.root
        for w in word:
            if w not in p.son:
                p.son[w]= Node()
            p = p.son[w]
        p.end = True

    def search(self, word: str) -> bool:
        p = self.root
        for w in word:
            if w not in p.son:
                return False 
            else:
                p =p.son[w]
        return p.end    

    def startsWith(self, prefix: str) -> bool:
        p = self.root
        for w in prefix:
            if w not in p.son:
                return False 
            else:
                p =p.son[w]
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```



------

回溯



## [46. 全排列](https://leetcode.cn/problems/permutations/)



下面是根据树进行构造的写法，主要参考依据为代码随想录，

但是总觉得此问题思考不够清晰，再往下有递归写法

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        """
           一棵树
           1  2  3
          23 13 12
         3 2 3 1 2 1
        """
        """
        	从递归子问题的角度思考较好
            [1,2]的全排列即为
                1 + [2]的排列
                2 + [1]的排列
        """
        n = len(nums)
        visit = [False] *n #标记数组 避免了重复重新构造数组
        path = []
        ans = []
        def dfs():
            if len(path) == n:
                ans.append(path[:])
                return 

            for i, num in enumerate(nums):
                if not visit[i]:
                    path.append(num)
                    visit[i]=True 

                    dfs()

                    path.pop()
                    visit[i]=False
            
        dfs()
        return ans
```







## [78. 子集](https://leetcode.cn/problems/subsets/)

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
            [1,2,3]的子集等于
                     [2,3]的子集等于
                            [3]的子集
                        2 、[3]的子集

                1   、 [2,3]的子集等于

        """

        n = len(nums)
        ans = []
        path = []

        def dfs(i):
            if i==n:
                ans.append(path.copy())
                return 
            
            # 不选
            dfs(i+1)

            # 选
            path.append(nums[i])
            dfs(i+1)
            path.pop()
                

        dfs(0)
        return ans
```



## [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        
        hashmap = {'2':"abc",
                    "3":"def",
                    "4":"ghi",
                    "5":"jkl",
                    "6":"mno",
                    "7":"pqrs",
                    "8":"tuv",
                    "9":"wxyz"}
        """
            "23"
            从对应hashmap中选一个即可
        """
        if len(digits) == 0:
            return []
        ans = []
        path = []#路径
        n = len(digits)

        def dfs(i):
            # i 表示已经选了i-1个字母,再考虑第i个字母
            if i == n:
                ans.append("".join(path))
                return 
            for alp in hashmap[digits[i]]:
                path.append(alp)
                dfs(i+1)

                #回溯，平等的选择同一个数字对应的下一个字母
                path.pop()
        dfs(0)
        return ans
```



## [39. 组合总和](https://leetcode.cn/problems/combination-sum/)

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
            递归子问题即可
        """

        ans = []
        path = []
        n = len(candidates)

        def dfs(i, sums):
            # i 表示已经考虑了前i-1个元素
            if i==n:
                if sums == target:
                    ans.append(path.copy())
                return
            if sums == target:
                ans.append(path.copy())
                return 
            if sums > target:
                return 
            
            #不选第i个
            dfs(i+1,sums)
            #选第i个
            path.append(candidates[i])
            dfs(i,sums+candidates[i])
            path.pop()
            
        dfs(0,0)
        return ans
```



## **[22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)

下面是错误做法：

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        """
            n对生成有效括号的组合 记作 dn
            在n=2时，答案为
            ()d1  d1() (d1) 
            如此递归子问题下去即可
        """

        if n==1:
            return ["()"]
        
        ans = self.generateParenthesis(n-1)
        
        result = set()
        for an in ans:
            result.add("()"+an)
            result.add("("+an+")")
            result.add(an+"()")
        return list(result)
```

![img](../picture.asset/QQ_1737121847882.png)



下面是正确的枚举写法：

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        """
            枚举填左括号还是右括号
                n对括号总共有2*n个位置
                dfs(i,lef,rig)表示当前考虑到第i位置，已经填了lef个左括号、rig个右括号
                    1.**对应任意位置， lef >= rig必然成立
                    **换句话说 lef<rig则不匹配
                    2.** lef个数小于等于n
        """

        ans = []
        path = [''] * (2*n) #采用直接覆盖的方式进行回溯

        def dfs(i,lef,rig):
            if lef<rig:
                return
            if lef >n:
                return 
            
            if lef+rig == n*2 and lef==rig:
                ans.append(''.join(path))
                return 
            
            #选左括号
            path[i]='('
            dfs(i+1,lef+1,rig)
            #选右括号
            path[i]=')'
            dfs(i+1,lef,rig+1)
        dfs(0,0,0)
        return ans
                
```



## **[79. 单词搜索](https://leetcode.cn/problems/word-search/)



下面这个是错误解法：

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        """
            1.查找开头字母
            2.回溯搜索即可
        """
        m,n = len(board), len(board[0])
        Dire = [[-1,0],[1,0],[0,1],[0,-1]]
        visit =[[False] * n] *m
        def dfs(i,j,k):
            """
                入口一定是 board[i][j]==word[0]
                注意 进入时置visit为True
            """
            if k==len(word)-1:
                return True

            for dx,dy in Dire:
                nx,ny = dx+i, dy+j
                if 0<=nx<m and 0<=ny<n and (not visit[nx][ny]) and board[nx][ny]==word[k+1]:
                    visit[nx][ny]=True
                    flag =  dfs(nx,ny,k+1)
                    if flag:
                        return True
                    visit[nx][ny]=False
            return False
        
        for i in range(m):
            for j in range(n):
                if board[i][j]==word[0]:
                    visit[i][j]=True
                    flag = dfs(i,j,0)
                    if flag:
                        return True
                    visit[i][j]=False # 这里也应该回溯 注意：
        return False
```

>代码最大错误点在于这里：
>        错误：visit =[[False] * n] *m
>        正确：visit = [[False for _ in range(n)] for _ in range(m)]
>
>详细解释如下：
>
>当你使用 `[[False] * n] * m` 来创建 `visit` 列表时，实际上是创建了一个 `m` 行 `n` 列的二维列表。然而，这里存在一个潜在的问题，即<font color='red'>内部列表的复制方式是浅复制</font>。具体来说，Python 只是复制了内部列表的引用，而不是创建新的列表对象。这意味着 `visit` 列表中的 `m` 个子列表实际上都指向同一个列表对象。
>
>例如，如果你修改 `visit[0][0] = True`，由于它们都指向同一个列表，那么 `visit[1][0]`、`visit[2][0]` 等都会变成 `True`，因为它们实际上是同一个列表对象的不同引用。

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        """
            1.查找开头字母
            2.回溯搜索即可
        """
        m,n = len(board), len(board[0])
        Dire = [[-1,0],[1,0],[0,1],[0,-1]]
        # visit =[[False] * n] *m
        visit = [[False for _ in range(n)] for _ in range(m)]
        def dfs(i,j,k):
            """
                入口一定是 board[i][j]==word[0]
                注意 进入时置visit为True
            """
            if k==len(word)-1:
                return True

            for dx,dy in Dire:
                nx,ny = dx+i, dy+j
                if 0<=nx<m and 0<=ny<n and (not visit[nx][ny]) and board[nx][ny]==word[k+1]:
                    visit[nx][ny]=True
                    flag =  dfs(nx,ny,k+1)
                    if flag:
                        return True
                    visit[nx][ny]=False
            return False
        
        for i in range(m):
            for j in range(n):
                if board[i][j]==word[0]:
                    visit[i][j]=True
                    flag = dfs(i,j,0)
                    if flag:
                        return True
                    visit[i][j]=False # 这里也应该回溯 注意：
        return False
```



## *[131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def isValid(st):
            i,j =0,len(st)-1
            while i<j:
                if st[i]!=st[j]:
                    return False
                i,j = i+1,j-1
            return True
        ans = []
        path = []

        def dfs(i,j):
            """
                将s[i:j)切割成每个子串都是回文串的可能的分割方案
            """
            if i>=j:
                ans.append(path.copy())
                # return 1
                return 
            # res = 0

            for k in range(i+1,j+1):
                tmp = s[i:k]
                if isValid(tmp):
                    # res += dfs(k,j)
                    path.append(tmp)
                    dfs(k,j)
                    path.pop()#回溯
            # return res
        dfs(0,len(s))
        
        return ans
```

## [51. N 皇后](https://leetcode.cn/problems/n-queens/)



```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        
        ans = []
        path = [["."]*n  for _ in range(n)]

        # def check(x,y):
        #     for i in range(n):
        #         if i!=y and path[x][i] == 'Q':
        #             return False
        #     for i in range(n):
        #         if i!=x and path[i][y]=='Q':
        #             return False
        #     for i,j in zip(range(x,-1,-1), range(y,-1,-1)):
        #         if i!=x and j!=y and path[i][j] == 'Q':
        #             return False
        #     # for i,j in zip(range(x,n), range(y,n)): # 这个方向还没有放置无需添加
        #     #     if i!=x and j!=y and path[i][j] == 'Q':
        #     #         return False
        #     # for i,j in zip(range(x,n), range(y,-1,-1)):
        #     #     if i!=x and j!=y and path[i][j] == 'Q':
        #     #         return False
        #     for i,j in zip(range(x,-1,-1), range(y,n)):
        #         if i!=x and j!=y and path[i][j] == 'Q':
        #             return False
        #     return True
        def check(x,y):
            for i in range(n):
                if i!=y and path[x][i] == 'Q':
                    return False
            for i in range(n):
                if i!=x and path[i][y]=='Q':
                    return False
            for i,j in zip(range(x-1,-1,-1), range(y-1,-1,-1)):
                if path[i][j] == 'Q':
                    return False
            for i,j in zip(range(x-1,-1,-1), range(y+1,n)):
                if  path[i][j] == 'Q':
                    return False
            return True

        def dfs(i):
            """
                枚举第i行皇后位置放置规则
            """
            if i == n:
                res = ["".join(p) for p in path]
                ans.append(res)
                return

            for j in range(n):
                path[i][j]='Q'
                if check(i,j):
                    dfs(i+1)
                path[i][j]='.'
        dfs(0)
        return ans
```



------------

二分查找



## **[35. 搜索插入位置](https://leetcode.cn/problems/search-insert-position/)

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        """
            将数组一分为二，返回第一个 >= target的位置
        """

        l,r = 0,len(nums)-1#闭区间  
        while l<r:
            mid = (l+r)//2
            if nums[mid] >= target:
                r=mid 
            else:
                l=mid+1
        # 如果整个数组都小于taget的话，r停滞在 len(nums)-1 的位置， 导致出错
        if nums[r] < target:
            r+=1 
        return r
```



使用下面的更好点（学了半吊子实在害人不浅），<font color='red'> 使用再下面的，一切向标准库看齐。</font>

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        """
            将数组一分为二，返回第一个 >= target的位置
        """

        # 首先处理边界问题  确定闭区间[l,r]包含答案
        if nums[0] > target:
            return 0
        if nums[len(nums)-1] < target:
            return len(nums)

        l,r = 0,len(nums)-1#闭区间
        while l<r:
            mid = (l+r)//2
            if nums[mid] >= target:
                r=mid 
            else:
                l=mid+1
        return r
```

>  一切向标准库看齐。



```python
def lower_bound(nums: List[int], target: int) -> int:
    """
        经典lower_bound， 
            对非递减的数组nums而言，
            在这里是返回最小的nums[i]>=target的i，
            attention!!!:: 如果数组为空，或者所有数字<taget，返回len(nums)
    """
    l,r = 0,len(nums)-1 # 闭区间[l,r]

    while l<=r: #区间不为空
        mid = (l+r)//2
        if nums[mid] < target:
            l=mid+1# 循环不变量 nums[l-1] < taget
        else: 
            r=mid-1# 循环不变量  nums[r+1] >= target
    # 退出循环后 l=r+1
    return l

class Solution:
    def searchInsert(self, arr: List[int], tar: int) -> int:
        """
            将数组一分为二，返回第一个 >= target的位置
        """
        
        return lower_bound(arr,tar)
```

>```python
>class Solution:
>    def searchInsert(self, arr: List[int], tar: int) -> int:
>        """
>            将数组一分为二，返回第一个 >= target的位置
>        """
>        
>        return bisect_left(arr,tar)
>```





## **[74. 搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/)

这题很有意思，联想到前面的[240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

可以对比对比差异

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
            明显使用二分即可
        """

        m,n = len(matrix), len(matrix[0])


        # 先二分行  一分为二，上边小于等于target，下边大于target， 找到上面的最下边
        # 即upper_bound-1

        l,r = 0, m-1 # 闭区间 

        while l<=r:
            mid = (l+r)//2
            if matrix[mid][0] > target:
                r=mid-1 # [r+1] > target
            else:
                l=mid+1 # [l-1] <= target
        # l=r+1
        row = r # 即upperbound -1 

        # 二分列 一分为二 左边小于target 右边大于等于target 找到右边的最左边
        # 即lower_bound
        l,r = 0,n-1 
        while l<=r:
            mid = (l+r)//2
            if matrix[row][mid] >= target:
                r=mid-1 # [r+1] >= target
            else:
                l=mid+1# [l-1] < target
        # l =r+1
        if l == n :# 没找到
            return False
        col = l
        return matrix[row][col] == target
```



## **[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

```python
def lower_bound(nums,tar):
    """
        对于非递减数组nums而言，
        返回第一个nums[i]>=tar的i
    """
    l,r = 0,len(nums)-1#闭区间
    while l<=r: #区间有效
        mid = (l+r)//2
        if nums[mid] >= tar:
            r=mid-1 #nums[r+1]>=tar
        else:
            l=mid+1 #nums[l-1]<tar
    #l=r+1
    return l

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        st = lower_bound(nums,target)
        if st == len(nums) or nums[st] != target:
            return [-1,-1]
        
        ed = lower_bound(nums,target+1)-1
        return [st,ed]

```



## **[153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

本题可以与**[189. 轮转数组](https://leetcode.cn/problems/rotate-array/) 相互照应，即数组的旋转方法

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        """
        题目已经确保进行了至少一次旋转
            3 4 5 1 2
            取 3+2 / 2= 2.5 作为tar值

            前半段值都大于2.5， 后半段值都小于2.5
            如果mid>2.5 说明处于前半段， l=mid+1
                mid<=2.5，说明处于后半段, r=mid-1
            
            1 2 4 5 6 顺序数组则不能适用于上面的规则
            因此对于此类旋转数组，需要特别判断
        """

        l,r= 0, len(nums)-1
        if nums[l] < nums[r]:#旋转数组是否等同原数组
            return nums[l]
        tar = (nums[l]+nums[r])//2
        while l<=r:
            mid = (l+r)//2
            if nums[mid] > tar:
                l=mid+1 # nums[l-1]>tar
            else:
                r=mid-1
        return nums[l]
        
            
```



## **[33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

经过本题后，请重新反思该题<font color='red'> 153. 寻找旋转排序数组中的最小值</font>

```python
# bisect_left
def lower_bound(nums,l,r,tar):
    # l,r = 0, len(nums)-1
    l,r = l,r
    while l<=r:
        mid = (l+r)//2
        if nums[mid]<tar:
            l=mid+1
        else:
            r=mid-1
    return l

class Solution:
    def search(self, nums: List[int], target: int) -> int:

        """
            1.如果轮转后数组等同于排序数组，直接二分

            2. 否则首先寻找轮转后数组的最小值，恰好将数组分为两部分，前半部分A 都大于某值，后半部分B都小于某值
                根据target于nums[0]的关系 判断target在A还是B，
                然后二分即可
        """
        n=len(nums)
        idx = -1
        # 正常排序数组
        if nums[0]<nums[n-1]:
            idx =  lower_bound(nums,0,n-1,target)
        else:

            # 轮转数组
            # 首先通过找最小值 将该数组分为A,B两部分
            l,r = 0, n-1
            cut = nums[0] # 找到小于nums[0]的最左边
            while l<=r:
                mid = (l+r)//2
                if nums[mid] < cut:
                    r=mid-1
                else:
                    l=mid+1
            mid = r+1
            l,r = 0, n-1
            # 通过比较确定是在A部分[l,mid-1] 还是B部分[mid,r]
            if target < cut:
                idx= lower_bound(nums,mid,r,target)
            else:
                idx =lower_bound(nums,l,mid-1,target)
        print(idx)
        return idx if idx<n and nums[idx]==target else -1
```



## ？**[4. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/)

初次看到这个题时，只能想到  双指针，类似归并排序似的不断移动i,j到一半长度。这样复杂度还是O(n)的

<font color='red'>怎么达到 O(logn)呢？</font>

参见评论区大牛的题解如下：



![img](../picture.asset/QQ_1737469265329.png)

1. 暴力排序：

   <font color='red'>中位数： 可将数值集合划分为大小相等的两部分。</font>

![lc4-1-c.png](../picture.asset/1728911111-PpxpUR-lc4-1-c.png)

2. **<font color='red'>均匀分组」，每组 5 个数，只要第一组的最大值 ≤ 第二组的最小值，我们就找到了答案。</font>**

![lc4-2-c.png](../picture.asset/1728911257-NIolPh-lc4-2-c.png)



---------

栈



## [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/)

```python
class Solution:
    def isValid(self, s: str) -> bool:
        """
            经典栈解法即可
            遇到左括号，直接入栈
            遇到右括号，与栈顶匹配掉即可
            最后查看栈中是否为空
        """
        mat = {'[':']','{':'}','(':')'}
        st = []
        for c in s:
            if c in mat.keys():
                st.append(c)
            else:
                if not st: 
                    #栈为空没有匹配的
                    return False
                if mat[st.pop()] != c:
                    # 匹配不上
                    return False
        return not st 
```

## [155. 最小栈](https://leetcode.cn/problems/min-stack/)

```python
class MinStack:
    """
        准备两个栈，一个普通栈，一个记录最小值的栈即可
    """
    def __init__(self):
        self.min_st = []
        self.st = []
        

    def push(self, val: int) -> None:
        self.st.append(val)
        val = val if not self.min_st or val < self.min_st[-1] else self.min_st[-1]
        self.min_st.append(val)

    def pop(self) -> None:
        self.min_st.pop()
        self.st.pop()

    def top(self) -> int:
        return self.st[-1]

    def getMin(self) -> int:
        return self.min_st[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

## [394. 字符串解码](https://leetcode.cn/problems/decode-string/)

```python
class Solution:
    def decodeString(self, s: str) -> str:
        """
        简易思路
            一个数字栈一个字母栈
            遇到]时将字母栈弹出直到遇到[
            重复足够次数后入栈即可
        """
        def repetition(string,k):
            res = ''
            for i in range(k):
                res+=string
            return res
        num_st = []
        alp_st = []

        buffer = 0
        for c in s:
            # print(num_st)
            # print(alp_st)
            # print('---------')
            if '0'<=c<='9':
                buffer = buffer*10+ord(c)-ord('0')
            elif c == '[':
                num_st.append(buffer)# [前面必定是数字，恰好标志数字结束
                alp_st.append(c)
                buffer=0
            elif 'a'<=c<='z':
                alp_st.append(c)
            else:# c==']'
                rep = num_st.pop()
                str_rep = ''
                while alp_st[-1] != '[':
                    str_rep += alp_st.pop()
                alp_st.pop()

                str_rep = repetition(str_rep, rep)
                for t in str_rep[::-1]:
                    alp_st.append(t)

        return ''.join(alp_st)
```



## [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/)

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """
            对于一个序列 73 74 75 71 69 72而言
                想要找到对于第i天， 其下一个更高温度在第几天

                对上述序列 74可以更新掉73
                        75又可以更新掉74
                        而71无法更新75，留置
                        69无法更新71，留置
                        到了72， 它可以更新留置序列 75 71 69中的 71和69
                该序列即为单调递减栈

                思路即为：
                    对于当前数字，持续更新栈直到栈顶数字大于等于当前数字，然后入栈
        """
        st = []
        answer=[0] * len(temperatures)

        for i,t in enumerate(temperatures):
            while st and temperatures[st[-1]] < t:
                answer[st[-1]] = i-st[-1]
                st.pop()
            st.append(i)
        return answer

```



## *[84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        """
            矩形面积 为宽(平行x轴)*高(平行y轴)

            heights = [2,1,5,6,2,3]
            1.
            对于第i根柱子，例如5而言，
                对于此高度5， 需要找到右侧第一个比5小的位置j,则[i,j-1]是右侧的宽
                而左侧同理，找到第一个比5小的位置k，[k,i-1]是左侧宽度
                最终对于这个柱子高度5，勾勒出来的最大面积即为 (j-1-k+1)*5
            遍历所有柱子即可

            2.
            对于上面的找到左侧第一个比height小的位置
                对于heights = [2,1,5,6,2,3]而言
                2左侧显然没有比其小的，留置
                1比2小，并且靠近右侧，所以右侧的数字一定不会找2，而是找1，所以2无用了，pop调
                5左侧目标位置为1的位置
                6的目标位置为5 [1,5,6]
                到了2，同样的2比5，6小，则右侧数字一定找2而不是5，6所以->[1,2]
                此为单调递增栈  每个数需要将递增序列尾部不断更新保留一个单调递增栈

            对于上面的找到右侧第一个比height小的位置
                逆序处理heights数组即可
        """

        n = len(heights)
        lef = [-1]*n
        rig = [n]*n

        # 先处理左边
        st=[]
        for i,t in enumerate(heights):
            while st and heights[st[-1]] >= t:
                st.pop()
            if st:
                lef[i]=st[-1]
            st.append(i)
        
        #再处理右边
        st=[]
        for i in range(n-1,-1,-1):
            while st and heights[st[-1]] >= heights[i]:
                st.pop()
            if st:
                rig[i]=st[-1]
            st.append(i)
        # print(lef)
        # print(rig)
        res = 0
        for i in range(n):
            width = (rig[i]-1) - (lef[i]+1) +1
            res = max(res,width*heights[i])
            # print(width*heights[i], end=' ')
        return res
```



--------



堆



## **[215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
            可以快排或者利用堆

            对于利用堆而言，
                创建一个k个数的小根堆，超过k后pop掉最小的数，
                那么最后遗留下来的就是前k大数，此时同样pop最小的数，即为第k大数
        """

        heap = []
        for num in nums:
            heapq.heappush(heap,num)

            if len(heap) > k:
                heapq.heappop(heap)
        return heapq.heappop(heap)

```



## [347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
            构造小根堆，大小大于k时pop掉最小的，最后留下的k个整数即为前k高
        """
        cnt = defaultdict(int)
        for num in nums:
            cnt[num] += 1

        heap = []
        for key,val in cnt.items():
            num=(val,key)
            heapq.heappush(heap,num)
            if len(heap)>k:
                heapq.heappop(heap)
        
        return [k for v,k in heap]
```



## *[295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)

```python
class MedianFinder:
    """
        中位数 即均分数组 A B
        A = [1,2] B=[3,4]
        A中最大值 2 < B中最小值3
        中位数 2+3//2 均分了A+B

        这里我们使用一个最大堆A， 一个最小堆B
        规定 len(A)==len(B) 或者len(A)==len(B)+1

        对于数据流中来的一个新的值
            将其与左侧值mx比较，如果小于mx，直接加入左侧A中
            否则将该值从右侧B中置换出一个最小值tmp出来，然后再加入左侧A中

            最后判断AB是否能够仍然满足长度关系
    """
    def __init__(self):
        self.mx = [] #需要加-号 模拟大根堆
        self.mn = [] #默认小根堆

    def addNum(self, num: int) -> None:
        mx,mn = self.mx, self.mn
        #查看是否需要置换
        if mx and -mx[0] < num:
            heappush(mn,num)
            num = heappop(mn)

        heappush(mx, -num)

        if len(mx) > len(mn)+1:
            tmp = -heappop(mx)
            heappush(mn, tmp)


    def findMedian(self) -> float:
        mx,mn = self.mx, self.mn
        if len(mx) == len(mn):
            return (-mx[0] + mn[0])/2
        
        return -mx[0] 



# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```





-------

贪心



## [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
            7 1 5 3 6 4

            显然对于第i天 比如1
            需要直到右侧最高的值，
            此时就可以获得改天最大利润
        """

        res = 0
        tmp = 0
        n = len(prices)
        mx = [0] * n
        mx[n-1]=prices[n-1]
        for i in range(n-2,-1,-1):
            mx[i]=max(prices[i], mx[i+1])
        
        for i in range(n):
            tmp = mx[i]-prices[i]
            res = max(res, tmp)
        return res
```



## *[55. 跳跃游戏](https://leetcode.cn/problems/jump-game/)

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        """
        从左到右遍历 nums，同时维护能跳到的最远位置 mx，初始值为 0。
        如果 i>mx，说明无法跳到 i，返回 false。
        否则，用 i+nums 更新 mx 的最大值。
        如果循环中没有返回 false，那么最后返回 true。

        """
        mx = 0 
        for i, jump in enumerate(nums):
            if i > mx:
                return False 
            
            mx = max(mx, i+jump)
        return True
```



## **[45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/)



下面是我最初的想法dp

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        """
            dp

            对于第i个位置，其可以更新到i+nums[i]

            dp[i+nums[i]] = min(dp[i]+1, dp[i+nums[i]])即可
        """
        n = len(nums)
        dp = [inf]*n
        dp[0]=0
        for i,jmp in enumerate(nums[:-1]):
            rig = min(n-1, i+jmp)
            for k in range(i+1,rig+1):
                dp[k] = min(dp[k], dp[i]+1)
        return dp[n-1]
```



然后下面是题解 贪心

就一句话  跳跃途中更新右边界， 每次选择跳到右边界的步骤即可

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        """
            每次跳跃时，都有可能会更新有边界

            因此，每次在区间范围内 选择能够跳到最远距离的地方起跳即可
            
        """

        idx = 0
        right = 0 #[idx, right]为此时可跳跃范围
        new_right = 0 #new_right用于 寻找过程中新的最大右边界
        ans = 0
        for idx in range(len(nums)):
            if idx==len(nums)-1:
                return ans

            new_right = max(new_right, nums[idx]+idx)

            if idx == right:      
                right = new_right
                ans += 1

        return ans
```



## [763. 划分字母区间](https://leetcode.cn/problems/partition-labels/)

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        """
            统计所有字母的区间

            此问题转换为 区间合并问题

            -------------
                ------------
                 -----
            基本思路为 按左端点排序，当前右端点如果大于等于接下来线段左端点，就更新该右端点
                        无可更新时，区间数目加一

                        左端点排序是因为
                            首先一个区间 [lef,rig] 
                                lef一定是最左端，一定不会更新
                                新的区间由于是按照左端点排序，所以左端点一定更大，
                                    如果新的左端点大于此时右端点rig，说明后面所有的线的左端点都大于，需要新的区间合并
                                    反之，可以合并新的线段，但是无法保证后面的所有的线段的情况
        """

        levels = [[-1,-1] for _ in range(26)]
        for i,c in enumerate(s):
            tmp = ord(c)-ord('a')
            lev = levels[tmp]
            if lev == [-1,-1]:
                lev = [i,i]
            else:
                lev[1]=max(lev[1],i)
            levels[tmp] = lev # 必须加上
        # 左端点从小到大排序
        levels = sorted(levels, key= lambda x: x[0])
        # print(levels)
        ans = []
        lef,rig = -1,-1
        for lev in levels:
            if lev == [-1,-1]:
                continue
            
            l,r = lev 
            if l>rig:
                if rig!=-1:
                    ans.append(rig-lef+1)
                lef,rig = l,r 
            else:
                rig = max(rig,r)
        ans.append(rig-lef+1)#最后会残留一个区间
        return ans
```





------

dp

## [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/)

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        """
            do[i]表示到达第i个楼梯不同方法数
            dp[i]=dp[i-1]+dp[i-2]
        """
        if n==1:
            return 1
        if n==2:
            return 2
        dp = [0]*n
        dp[0], dp[1]=1,2
        for i in range(2,n):
            dp[i]=dp[i-1]+dp[i-2]
        return dp[n-1]

```



## [118. 杨辉三角](https://leetcode.cn/problems/pascals-triangle/)

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        """
            1
            1 1
            1 2 1
            1 3 3 1

            dp[m][n] = dp[m-1][n] + dp[m-1][n-1]
        """

        ans = [[1 for j in range(i+1)] for i in range(numRows)]
        for i in range(numRows):
            for j in range(1,i):
                ans[i][j]=ans[i-1][j]+ans[i-1][j-1]
        return ans
               
```



## [198. 打家劫舍](https://leetcode.cn/problems/house-robber/)

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        """
            dp[i]表示偷到第i家后得到的最高金额

            dp[i] = {dp[i-2]+nums[i], dp[i-1]}
        """
        n=len(nums)
        if n==1:
            return nums[0]
        elif n==2:
            return max(nums[0],nums[1]) 

        dp = [0]*n 
        dp[0], dp[1]= nums[0], max(nums[0],nums[1])
        for i in range(2,n):
            dp[i]=max(dp[i-1],dp[i-2]+nums[i])
        return dp[n-1]

        
```



## *[279. 完全平方数](https://leetcode.cn/problems/perfect-squares/)



先写了下面这个解法

```python
class Solution:
    def numSquares(self, n: int) -> int:
        """
            和为n的最少数量

            dfs(n) 
        """

        sels = [i**2 for i in range(1,int(math.sqrt(n))+1)]

        @cache
        def dfs(n):
            if n<0:
                return inf
            elif n==0:
                return 0
            res = n
            for sel in sels:
                if n>=sel:
                    res = min(res, 1+dfs(n-sel))
            return res
        return dfs(n)
```



看了题解，又下面的:视作背包问题：

```python
# 写在外面，多个测试数据之间可以共享，减少计算量
@cache# 缓存装饰器，避免重复计算 dfs 的结果（记忆化）
def dfs(i,j):
    """
        从前i个完全平方数中选择一些数字，其值等于j，最少要选的数字个数
    """
    if i==0:
        return 0 if j==0 else inf 

    if j<i*i:
        return dfs(i-1,j) #无法选

    return min(dfs(i-1,j), dfs(i,j-i*i)+1) #注意可以重复选


class Solution:
    def numSquares(self, n: int) -> int:
        
        return dfs(isqrt(n),n)
        # math.isqrt返回的是  小于等于n的最大完全平方根

```





## [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        """
            完全背包问题
        """
        n=len(coins)
        @cache
        def dfs(i,j):
            """
                返回前i种硬币，凑成金额j， 所需的最小的硬币个数
            """

            if i==-1:
                return 0 if j==0 else inf
            
            if coins[i] > j:
                return dfs(i-1,j)
            
            return min(dfs(i-1,j), dfs(i,j-coins[i])+1)#注意可以接着重复选
        ans = dfs(n-1,amount)
        return ans if ans != inf else -1
```



## [139. 单词拆分](https://leetcode.cn/problems/word-break/)

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
            同样可以视作完全背包问题
        """
        wd = set(wordDict)
        n = len(s)

        @cache
        def dfs(i):
            """
                考虑到s的第i个位置,查看是否能够匹配成功
            """
            if i==n:
                return True
            ans = False
            # 考虑到 ap  apple 都算作互不相同的字符串
            for j in range(i+1,n+1):
                split = s[i:j]
                if split in wd:
                    ans = ans or dfs(j)
            return ans 
        return dfs(0)

```



## **[300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)





下面是基本解法，基本注意点是这里是 子序列而不是子数组，子数组具有连续的性质

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """
            dp[i]记作以nums[i]结尾的最长严格递增子序列长度

        """

        ans = 1
        dp = [1] * len(nums)

        for i in range(1,len(nums)):
            for j in range(i):
                if nums[i]>nums[j]:
                    dp[i]=max(dp[i], dp[j]+1)
            ans = max(ans,dp[i])
        return ans
```



然后是一个进阶解法，贪心+二分查找

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """
            定义g[i]表示长度为i+1的递增子序列的末尾元素的最小值
                
        """
        g = []
        for x in nums:
            j = bisect_left(g,x) # 相当于lower_bound 即从从小到大的数组中二分到大于等于x的最小值位置

            if j==len(g):
                g.append(x)
            else:
                g[j]=x
        return len(g)

```



## [152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        """
            此问题需要考虑负数的问题
                尤其一个最大正数乘以一个负数就变成最小的了

            mx[i]表示以nums[i]结尾的乘积最大子数组
            mn[i]表示以nums[i]结尾的乘积最小子数组

                由于是子数组， 具备连续的性质
            
            则有 if nums[i]<0:
                    mx[i],mn[i] = nums[i]*mn[i-1], nums[i]*mx[i-1]
                else:
                    mx[i],mn[i] = nums[i]*mx[i-1], nums[i]*mn[i-1]
        """
        mx,mn = [1]*len(nums), [1]*len(nums)
        mx[0],mn[0]= nums[0],nums[0]
        ans = nums[0]
        for i in range(1,len(nums)):
            #乘
            if nums[i]<0:
                mx[i],mn[i] = nums[i]*mn[i-1], nums[i]*mx[i-1]
            else:
                mx[i],mn[i] = nums[i]*mx[i-1], nums[i]*mn[i-1]
            #不乘
            mx[i],mn[i]= max(nums[i],mx[i]), min(nums[i],mn[i])
            ans = max(ans, mx[i])
        return ans
```



## [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        """
            首先求出sum
            问题转换为 01背包问题，值为sum/2(sum必须为偶数)

        """
        n=len(nums)
        s = sum(nums)
        if s%2==1:#sum必须为偶数
            return False
        @cache
        def dfs(i,j):
            """
                从0开始考虑到了第i个数，接下来能否凑够价值j， return Bool
            """

            if i==n:
                return True if j==0 else False
            
            if nums[i] > j:
                return dfs(i+1,j)
            return dfs(i+1,j) or dfs(i+1,j-nums[i])
        return dfs(0,s//2)
```



## **[32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)



下面是最开始的错误想法：

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        """
            括号匹配 明显使用栈比较好

            这里的问题在于总不能每个位置都用栈做一下

            可以只用栈进行一遍括号匹配，
                对于(()而言
                    匹配完毕后剩余( 不作数
                对于)()())而言
                    需要一个记录连续性的东西即可
        """

        st = []
        ans = 0 
        tmp = 0
        for c in s:
            if c=='(':
                st.append(c)
            else:
                if not st or st[-1] !='(':
                    #无法匹配掉
                    ans = max(ans,tmp)
                    tmp=0
                    st.append(')')
                else:
                    tmp+=2
                    st.pop()
        ans = max(ans,tmp)#如果最后一个匹配成功， tmp没有更新ans
        return ans
```

<img src="../picture.asset/image-20250124181344475.png" alt="image-20250124181344475" style="zoom:50%;" />

这个解法的问题在于只考虑了 ‘)'作为分割边界，由此导致了中间可以穿插(的问题

下面是一个容易理解的题解：



题目为找到括号匹配成功的最长连续字符，我们可以进行问题转化。这里有两个关键点。
1.匹配成功
2.最长连续
因此，分别模拟以上两个过程即可

首先是“匹配成功”，这个用栈来实现。为了给第2步做准备，我们要在匹配成功时做个记号，这里开辟一个数组，匹配成功时，在'('和")'的索引位置处记为1。
然后统计数组里面连续1的个数，最长的那个就是结果

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:

        st = []
        flag = [0]*len(s)
        for i,c in enumerate(s):
            if c=='(':
                st.append(i)
            else:
                if st and s[st[-1]]=='(':
                    flag[st[-1]],flag[i]=1,1
                    st.pop()
                else:
                    st.append(i)
               
        # print(flag)
        ans = 0
        cur = 0
        for f in flag:
            if f==0:
                ans = max(ans,cur)
                cur=0
            else:
                cur+=1
        ans = max(ans,cur)
        return ans
```



## [62. 不同路径](https://leetcode.cn/problems/unique-paths/)

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        """
            dp[i][j]记作到达(i,j)的不同路径条数

            dp[i][j]  <-  d[i-1][j] + dp[i][j-1]
        """

        dp = [[0 for i in range(n)] for j in range(m)]
        dp[0][0]=1
        for i in range(m):
            for j in range(n):
                if i:
                    dp[i][j]+=dp[i-1][j]
                if j:
                    dp[i][j]+=dp[i][j-1]
        return dp[m-1][n-1]

```



## [64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/)

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        """
            dp[i][j]表示到达(i,j)位置的最小数字总和

            dp[i][j]  <-- min(dp[i-1][j] , dp[i][j-1])+nums[i][j]
        """
        m,n = len(grid), len(grid[0])
        dp = [[inf for i in range(n)] for j in range(m)]
        dp[0][0]=grid[0][0]
        for i in range(m):
            for j in range(n):
                if i:
                    dp[i][j]=min(dp[i][j],dp[i-1][j]+grid[i][j])
                if j:
                    dp[i][j]=min(dp[i][j],dp[i][j-1]+grid[i][j])
        return dp[m-1][n-1]

```



## *[5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        """
            s[i:j]是否为回文子串 <- s[i+1:j-1] and s[i]==s[j]
        """
        n=len(s)
        dp = [[0]*n for _ in range(n)]
        ans = ""
        num = 0
        for i in range(n-1,-1,-1):
            for j in range(i,n): #注意递推方向
                if i==j:
                    dp[i][j]=True
                elif i+1==j:
                    dp[i][j]= s[i]==s[j]
                else:
                    dp[i][j]= (s[i]==s[j])and dp[i+1][j-1]
                
                if dp[i][j] and j-i+1>num:
                    ans = s[i:j+1]
                    num = j-i+1
        return ans
```



## [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        
        @cache
        def dfs(m,n):
            # 返回text1[:m]和text2[:n]的最长公共子序列

            if m<0 or n<0:
                return 0
            
            if text1[m]==text2[n]:
                return dfs(m-1,n-1)+1
            else:
                lef,rig = dfs(m-1,n), dfs(m,n-1)
                return max(lef,rig)
        return dfs(len(text1)-1, len(text2)-1)
```



## [72. 编辑距离](https://leetcode.cn/problems/edit-distance/)

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        
       
        @cache
        def dfs(m,n):
            #return word1[:m]转换成word2[:n]所需最小操作数
            if m<0:
                return n+1
            if n<0:
                return m+1

            if word1[m]==word2[n]:
                return dfs(m-1,n-1)
            else:
                return min(dfs(m-1,n)+1, dfs(m,n-1)+1, dfs(m-1,n-1)+1)
                # w1[m]与w2[n]不相等
                # 为把w1转换为w2
                # 可以删除一个字符 dfs(m-1,n)
                # 可以插入一个字符 dfs(m,n-1)
                # 可以替换一个字符 dfs(m-1,n-1)
        m,n = len(word1), len(word2)
        return dfs(m-1,n-1)
```



-----

技巧



## *[136. 只出现一次的数字](https://leetcode.cn/problems/single-number/)

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        """
            由于异或的优良性质 
             x xor x = 0
            所有我们可以把所有的数字xor一遍，剩下来的就是只出现了一次的元素
        """

        ans = 0
        for num in nums:
            ans = ans ^ num 
        return ans
```





## **[169. 多数元素](https://leetcode.cn/problems/majority-element/)

简单思考 用 hash统计一下就好了， 但是这样子时空间复杂度均为 O（N）

题目需要时间复杂度为O(n)而空间复杂度为O(1)的算法，

在评论区找到了 [Krahets](https://leetcode.cn/u/jyd/) 的详细题解，如下

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        votes = 0
        x = 0
        for num in nums:
            if votes == 0:
                x=num 
            
            if num == x:
                votes += 1
            else:
                votes -= 1
        return x
```



> 在本文中，“数组中出现次数超过一半的数字” 被称为 **“众数”** 。
>
> 需要注意的是，数学中众数的定义为 “数组中出现次数最多的数字” ，与本文定义不同。

本题常见的三种解法：

哈希表统计法： 遍历数组 nums ，用 HashMap 统计各数字的数量，即可找出 众数 。此方法时间和空间复杂度均为 O(N) 。
数组排序法： 将数组 nums 排序，数组中点的元素 一定为众数。
摩尔投票法： 核心理念为 票数正负抵消 。此方法时间和空间复杂度分别为 O(N) 和 O(1) ，为本题的最佳解法。



![Picture1.png](../picture.asset/1603612327-bOQxzq-Picture1.png)

摩尔投票：
设输入数组 nums 的众数为 x ，数组长度为 n 。

推论一： 若记 众数 的票数为 +1 ，非众数 的票数为 −1 ，则一定有所有数字的 票数和 >0 。

推论二： 若数组的前 a 个数字的 票数和 =0 ，则 数组剩余 (n−a) 个数字的 票数和一定仍 >0 ，即后 (n−a) 个数字的 众数仍为 x 。



## **[75. 颜色分类](https://leetcode.cn/problems/sort-colors/)

首先想到了类似快排，但是存在一个问题

之前的快排代码是只能将区间二分， 即左侧小于flag，右侧大于等于flag

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        flag = 1
        i,j = 0,len(nums)-1
        while i<=j:
            while i<=j and nums[i]<flag:
                i+=1 
            while i<=j and nums[j]>=flag:
                j-=1
            if i<=j:
                nums[i], nums[j]= nums[j], nums[i]
        
```

![image-20250125151027634](../picture.asset/image-20250125151027634.png)

下面是一个正确做法，三指针

cur指针指向正在处理的元素，然后规定[0,lef) (rig,len(nums)]分别为0，2区间

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        lef,rig = 0, len(nums)-1 # [0,lef)为0  (rig,len(nums)]为2
        cur = 0 #[lef,cur)为1

        while cur<=rig: #一直处理到 rig
            if nums[cur]==0:
                # 与left交换
                nums[lef],nums[cur]=nums[cur],nums[lef]
                lef+=1
                # 由于left<=cur  即left是已经处理过的元素 cur++
                cur+=1
            elif nums[cur]==1:
                cur+=1
            elif nums[cur]==2:
                #与右侧交换
                nums[rig],nums[cur]=nums[cur],nums[rig]
                rig-=1
                #由于rig元素并未处理过所以不移动cur
```



## ?**[31. 下一个排列](https://leetcode.cn/problems/next-permutation/)







## [287. 寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/)

本题目与 41. 缺失的第一个正数 的思维比较想像

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        """
            利用数组本身作为标记

            [1,3,4,2,2]
            包含(n=4)+1 共5个数字，数字都在[1,n]之间

            对第i个数， 将其作为下标索引， 对应位置数字置为负数，如遇到两个负数则找到重复的数字
        """

        for i,num in enumerate(nums):
            tag = num if num > 0 else -num 

            if nums[tag] < 0:
                return tag
            else:
                nums[tag] = -nums[tag]
        return -1
```



2025.1.25

只剩下 下一个排列 以及 寻找两个正序数组的中位数 还是一头雾水

当然，中间也有一些不是完全理解的，也不一定是最优解



至此，第一遍速通算是完成。

令人最印象深刻的是  递归子问题，以及加上@cache来解决重叠子问题的dp， 越来越喜欢dp了

2025.1.25记

------------------

