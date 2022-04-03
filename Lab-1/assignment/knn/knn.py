"""KNN Implementation using KDTree for point cloud data"""

import heapq
import math

from operator import itemgetter
from pprint import pformat
from typing import Any, NamedTuple


Point = tuple[int | float, ...]
PointList = list[Point]
Best = NamedTuple("best", point=None, distance=float)
BestType = Best | None


def euclidean_distance(x: Point, y: Point) -> float:
    """Calculate the Euclidean distance between two points"""
    return math.sqrt(sum((x_i - y_i) ** 2 for x_i, y_i in zip(x, y)))


class Node(NamedTuple):
    """Node in a kd-tree"""
    point: Point
    left: Any
    right: Any

    def __repr__(self):
        return pformat(tuple(self))


class KDTree:
    """K-Dimensional Tree"""

    def __init__(self, points: PointList) -> None:
        if not points:
            raise ValueError("Must have at least one point")
        self.k = len(points[0])
        self.root = self.build_tree(points, 0)

    def __repr__(self):
        return pformat(self.root)
    
    def __iter__(self):
        """Iterate through the points in the tree"""
        if not self.root:
            return

        def traverse(node: Node | None):
            if not node:
                return None

            traverse(node.left)
            yield node.point
            traverse(node.right)

        yield from traverse(self.root)

    def build_tree(self, points: list[Point], depth: int) -> Node | None:
        """ Create a K-Dimensional Tree"""
        if not points:
            return None

        # select axis based on depth so that axis cycles through all valid values
        axis = depth % self.k

        # sort point list and choose median as pivot element
        points.sort(key=itemgetter(axis))
        median = len(points) // 2

        # create node and construct subtrees
        return Node(
            point=points[median],
            left=self.build_tree(points[:median], depth + 1),
            right=self.build_tree(points[median + 1:], depth + 1)
        )

    def get_knn(self, point: Point, k: int):
        """ Find the k-nearest neighbors"""

        def get_nearest(
            node: Node | None, 
            point: Point, 
            k:int, 
            heap: list[tuple[float, Point]], 
            i: int=0, 
            tiebreaker: int=1
        ) -> list[tuple[float, Point]]:
            if not node:
                return heap

            axis = i % self.k
            distance = euclidean_distance(point, node.point)

            if len(heap) < k:
                heapq.heappush(heap, (distance, node.point))
            elif distance < heap[0][0]:
                heapq.heappushpop(heap, (distance, node.point))

            if point[axis] < node.point[axis]:
                get_nearest(node.left, point, k, heap, i + 1, tiebreaker)
            else:
                get_nearest(node.right, point, k, heap, i + 1, tiebreaker)

            return heap
        
        return heapq.nsmallest(k, get_nearest(self.root, point, k, []))

    def get_nearest(self, point: Point) -> BestType:
        """Recursively search through the k-d tree to find the
        nearest neighbor.
        """
        if not self.root:
            return None

        best: BestType = None
        depth = 0

        def traverse(node: Node | None, best: BestType, point: Point, depth: int) -> BestType:
            if not node:
                return best

            axis = depth % self.k
            distance = euclidean_distance(point, node.point)

            if best is None or distance < best.distance:
                best = Best(point=node.point, distance=distance)

            if point[axis] < node.point[axis]:
                best = traverse(node.left, best, point, depth + 1)
            else:
                best = traverse(node.right, best, point, depth + 1)
            return best

        return traverse(self.root, best, point, depth)


class KNN:
    """K Nearest Neighbor"""

    def __init__(self, n_neighbors: int) -> None:
        self.k = n_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, x_train: PointList, y_train: PointList) -> None:
        """Train Model"""
        self.x_train = KDTree(x_train)
        self.y_train = y_train

    def predict(self, x_test: PointList):
        """Predict Result"""
        if not self.x_train:
            raise ValueError("Model not trained")
        
        return [self.x_train.get_knn(x, self.k) for x in x_test]


def main():
    """Example usage"""
    point_list: PointList = [(7, 2), (5, 4), (9, 6), (4, 7), (8, 1), (2, 3)]
    # reference_points: PointList = [ (1, 2), (3, 2), (4, 1), (3, 5) ]
    tree = KDTree(point_list)
    # print(tree)
    # print(tree.get_nearest((10, 1)))
    print(tree.get_knn((2, 2), 3))

    knn = KNN(3)
    knn.fit(point_list, point_list)
    print(knn.predict([(2, 2)]))


if __name__ == '__main__':
    main()
