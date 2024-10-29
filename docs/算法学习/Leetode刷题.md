# Leetcode进阶之旅


> 目标2000分


## 2024.10.18

做了两题
[1004. 最大连续1的个数 III](https://leetcode.cn/problems/max-consecutive-ones-iii/description/)
利用滑动窗口做了
```c++
 int longestOnes(vector<int>& nums, int k) {
        //可以考虑滑动窗口求解
        //窗口大小即为k， 表示窗口中对于0的最大容忍度为k
        
        int res=0;
        int l=0,r=0;//窗口 [l,r]
        int cnt=0;//0的数量
        for(r=0;r<nums.size();r++){
            //先添加进窗口中
            if(nums[r]==0)cnt++;

            //对于闭区间[l,r]中
            //收缩左边界
            // while(cnt>k&&l<=r){
            while(cnt>k){
                if(nums[l]==0)cnt--;
                l++;
            }

            res=max(res,r-l+1);
        }
        return res;

    }

```


[1594. 矩阵的最大非负积](https://leetcode.cn/problems/maximum-non-negative-product-in-a-matrix/description/)
```c++
int maxProductPath(vector<vector<int>>& grid) {
        /*
            简单dp
            注意维护两个，一个最小值、一个最大值即可
            grid[i,j] >= 0 时
                mi[i,j]=min(mi[i-1,j],mi[i,j-1])*g[i,j]
                ma[i.j]=max(ma[i-1,j],ma[i,j-1])*g[i,j]
            grid[i,j] < 0
                mi[i,j]=g[i,j]* max(ma[i-1,j],ma[i,j-1])
                ma[i,j]=g[i,j]* min(mi[i-1,j],mi[i,j-1])
        */
        const int INF=1e9+7;
        int m=grid.size(),n=grid[0].size();
        vector<vector<long>>mi(m,vector<long>(n,0));
        vector<vector<long>>ma(m,vector<long>(n,0));

        mi[0][0]=grid[0][0];
        ma[0][0]=grid[0][0];

        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(!i&&!j)continue;
                if(!i){
                    if(grid[i][j] >=0){
                        mi[i][j]=grid[i][j]*mi[i][j-1];
                        ma[i][j]=grid[i][j]*ma[i][j-1];
                    }else{
                        mi[i][j]=grid[i][j]*ma[i][j-1];
                        ma[i][j]=grid[i][j]*mi[i][j-1];
                    }
                }else if(!j){
                    if(grid[i][j] >=0){
                        mi[i][j]=grid[i][j]*mi[i-1][j];
                        ma[i][j]=grid[i][j]*ma[i-1][j];
                    }else{
                        mi[i][j]=grid[i][j]*ma[i-1][j];
                        ma[i][j]=grid[i][j]*mi[i-1][j];
                    }
                }else{
                    if(grid[i][j] >=0){
                        mi[i][j]=grid[i][j]*min(mi[i-1][j],mi[i][j-1]);
                        ma[i][j]=grid[i][j]*max(ma[i-1][j],ma[i][j-1]);
                    }else{
                        mi[i][j]=grid[i][j]*max(ma[i-1][j],ma[i][j-1]);
                        ma[i][j]=grid[i][j]*min(mi[i-1][j],mi[i][j-1]);
                    }
                }
                
            }
        }
        return ma[m-1][n-1] < 0? -1: ma[m-1][n-1]%INF;

    }

```

## 2024.10.19

[1685. 有序数组中差绝对值之和](https://leetcode.cn/problems/sum-of-absolute-differences-in-a-sorted-array/)

前缀和类型
```c++
    class Solution {
public:
    vector<int> getSumAbsoluteDifferences(vector<int>& nums) {
        /*
        
        对于num[i]而言  sum( |num[i]-num[j]| )可以分为两个部分进行计算
                    小于i即左侧的部分  sum(num[i]-num[j])
                    大于i即右侧的部分  sum(num[j]-num[i])
                    由此可用前缀和做
                    (i+1)*num[i] - s[i]
                    -(len-1-i)*num[i]+ (s[len-1]-s[i])
        */
        int n=nums.size();
        vector<int> res,s;
        for(int i=0;i<n;i++){
            if(!i)s.push_back(nums[i]);
            else s.push_back(s[i-1]+nums[i]);
        }
        for(int i=0;i<n;i++){
            int lef=(i+1)*nums[i]-s[i],
                rig=-(n-1-i)*nums[i] + s[n-1] - s[i];
            res.push_back(lef+rig);
        }
        return res;



    }
};

```
