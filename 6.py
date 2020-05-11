class Solution:
    def uniquePaths(self, m, n):
        dp = [[0] *n] *m
        dp[1][0] = 1
        dp[0][1] = 1
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]

if __name__ == "__main__":
    a = Solution()




    res = a.uniquePaths(3,2)

    print(res)#res = ['2413', '3142']