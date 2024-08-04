import collections
from math import sqrt

class BombDetonation:
    def __init__(self, bombs):
        self.bombs = bombs
        self.graph = self._construct_graph()

    def _can_detonate(self, bomb1, bomb2):
        x1, y1, r1 = bomb1
        x2, y2, _ = bomb2
        distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance <= r1

    def _construct_graph(self):
        n = len(self.bombs)
        graph = collections.defaultdict(list)
        for i in range(n):
            for j in range(n):
                if i != j and self._can_detonate(self.bombs[i], self.bombs[j]):
                    graph[i].append(j)
        return graph

    def _dfs(self, node, visited):
        visited.add(node)
        count = 1
        for neighbor in self.graph[node]:
            if neighbor not in visited:
                count += self._dfs(neighbor, visited)
        return count

    def maximum_detonation(self):
        max_detonations = 0
        for i in range(len(self.bombs)):
            visited = set()
            max_detonations = max(max_detonations, self._dfs(i, visited))
        return max_detonations


if __name__ == '__main__':
    bombs = [[1, 2, 3], [2, 3, 1], [3, 4, 2], [4, 5, 3], [5, 6, 4]]
    detonation = BombDetonation(bombs)
    print(detonation.maximum_detonation())  
