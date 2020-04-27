from collections import defaultdict
class Solution:

    def findLadders(self, beginWord, endWord, wordList) :
        from collections import deque


        l = len(beginWord)
        que = deque([(beginWord)])
        res = []
        general = defaultdict(list)
        mask = defaultdict(bool)
        mask[beginWord] = True
        a = []
        for word in wordList:
            for i in range(l):
                general[word[:i] + "*" + word[i + 1:]].append(word)

        while que:
            curr = que.popleft()
            a.append(curr)
            for i in range(l):
                temp = curr[:i] + "*" + curr[i + 1:]
                next_temp = general[temp]
                for item in next_temp:
                    if item == endWord:
                        res.append(a)

                    if not mask[item]:
                        mask[item] = True
                        que.append((item))

        return [res]
if __name__ == "__main__":
    a = Solution()
    b = "hit"
    e = "cog"
    w = ["hot", "dot", "dog", "lot", "log", "cog"]
    res = a.findLadders(b,e,w)
    print(res)