from collections import deque
from collections import defaultdict
class Solution:
    def minimalSteps(self, maze):
        def bfs(begin,end):
            visited = [[True * cols] * rows]
            location = [[0, 1], [1, 0], [-1, 0], [0, -1]]
            bi,bj = begin
            ei,ej = end

            min = 0
            while que:
                min += 1
                x, y = que.popleft()

                for l in location:
                    i = x + l[0]
                    j = y + l[1]
                    if i >= bi and i <= ei and j >= bj and j < ej:
                        if visited[i][j]:
                            que.append((i, j))
                            visited[i][j] = False

        rows = len(maze)
        cols = len(maze[0])
        que = deque()
        start = ()
        end = ()
        O = ()
        M = deque()

        barrier = deque()
        for i in range(rows):
            for j in range(cols):
                    if maze[i][j] == "S":
                        start =(i,j)
                    if maze[i][j] == "T":
                        end = (i, j)
                    if maze[i][j] == "#":
                        barrier.append((i,j))

                    if maze[i][j] == "M":
                        M.append((i, j))
                    if maze[i][j] == "O":
                        O = (i, j)
        path1 = bfs(start,O)
        for item in
        path2 = bfs(O)




if __name__ == "__main__":
    a = Solution()
    b = "hit"
    e = "cog"
    w =  ["hot", "dot", "dog", "lot", "log", "cog"]
    res = a.minimalSteps( ["S#O", "M..", "M.T"])


    print(res)