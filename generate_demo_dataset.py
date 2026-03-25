import json, random, csv
from pathlib import Path

PROBLEMS = [
    # ── Graphs ────────────────────────────────────────────────────────────────
    {"name":"Shortest Path Grid","rating":1200,"tags":"graphs,bfs",
     "statement":"Find the shortest path from top-left to bottom-right in a grid with obstacles using BFS."},
    {"name":"Connected Components","rating":1000,"tags":"graphs,dfs",
     "statement":"Count the number of connected components in an undirected graph using DFS or Union-Find."},
    {"name":"Cycle Detection","rating":1400,"tags":"graphs,dfs",
     "statement":"Detect if a directed graph contains a cycle using DFS with coloring."},
    {"name":"Bipartite Check","rating":1300,"tags":"graphs,bfs",
     "statement":"Determine whether an undirected graph is bipartite using 2-coloring BFS."},
    {"name":"Dijkstra Cities","rating":1600,"tags":"graphs,dijkstra,shortest path",
     "statement":"Given a weighted graph of cities, find the minimum cost to travel from source to destination."},
    {"name":"MST Construction","rating":1700,"tags":"graphs,mst,kruskal",
     "statement":"Build the minimum spanning tree of a network using Kruskal's algorithm."},
    {"name":"Topological Sort","rating":1500,"tags":"graphs,topological sort,dag",
     "statement":"Output a valid topological ordering of a DAG representing task dependencies."},
    {"name":"Flood Fill","rating":900,"tags":"graphs,bfs,grid",
     "statement":"Count the size of the region connected to a given cell in a 2D grid."},

    # ── Dynamic Programming ───────────────────────────────────────────────────
    {"name":"Knapsack Classic","rating":1600,"tags":"dp,knapsack",
     "statement":"Given weights and values of items and a capacity, maximize the total value using 0/1 knapsack DP."},
    {"name":"Longest Common Subsequence","rating":1500,"tags":"dp,strings",
     "statement":"Find the length of the longest common subsequence of two strings."},
    {"name":"Coin Change","rating":1400,"tags":"dp,greedy",
     "statement":"Find the minimum number of coins to make a given amount using available denominations."},
    {"name":"Matrix Chain Multiplication","rating":1900,"tags":"dp,interval dp",
     "statement":"Minimize the number of multiplications needed to multiply a chain of matrices."},
    {"name":"Edit Distance","rating":1700,"tags":"dp,strings",
     "statement":"Compute the minimum edit distance (insert/delete/replace) between two strings."},
    {"name":"Longest Increasing Subsequence","rating":1500,"tags":"dp,binary search",
     "statement":"Find the length of the longest strictly increasing subsequence in an array."},
    {"name":"Staircase DP","rating":800,"tags":"dp",
     "statement":"Count the number of ways to climb n stairs taking 1 or 2 steps at a time."},
    {"name":"Partition Equal Subset","rating":1600,"tags":"dp,knapsack",
     "statement":"Determine if an array can be partitioned into two subsets with equal sum."},

    # ── Sorting & Binary Search ───────────────────────────────────────────────
    {"name":"Merge Sort Count","rating":1700,"tags":"sorting,divide and conquer",
     "statement":"Count the number of inversions in an array using a modified merge sort."},
    {"name":"Binary Search Answer","rating":1400,"tags":"binary search",
     "statement":"Binary search on the answer: find the minimum maximum distance between placed points."},
    {"name":"Kth Smallest Element","rating":1300,"tags":"sorting,binary search",
     "statement":"Find the k-th smallest element in an unsorted array in O(n log n)."},
    {"name":"Two Sum Sorted","rating":900,"tags":"binary search,two pointers",
     "statement":"Given a sorted array, find two numbers that add up to a target value."},

    # ── Strings ───────────────────────────────────────────────────────────────
    {"name":"Pattern Matching KMP","rating":1800,"tags":"strings,kmp",
     "statement":"Find all occurrences of a pattern in a text using the KMP algorithm."},
    {"name":"Anagram Groups","rating":1100,"tags":"strings,hashing",
     "statement":"Group a list of strings into sets of anagrams."},
    {"name":"Palindrome Substrings","rating":1500,"tags":"strings,dp",
     "statement":"Count the number of palindromic substrings in a given string."},
    {"name":"Longest Palindromic Subsequence","rating":1600,"tags":"strings,dp",
     "statement":"Find the longest palindromic subsequence of a string using DP."},
    {"name":"Run-Length Encoding","rating":800,"tags":"strings,implementation",
     "statement":"Compress a string using run-length encoding."},

    # ── Data Structures ───────────────────────────────────────────────────────
    {"name":"Segment Tree Range Sum","rating":1900,"tags":"data structures,segment tree",
     "statement":"Answer range sum queries and point updates on an array using a segment tree."},
    {"name":"Fenwick Tree","rating":1800,"tags":"data structures,bit",
     "statement":"Support prefix sum queries and point updates using a Binary Indexed Tree."},
    {"name":"Stack Balanced Parentheses","rating":900,"tags":"data structures,stack",
     "statement":"Check if a string of brackets is balanced using a stack."},
    {"name":"Queue via Two Stacks","rating":1000,"tags":"data structures,stack,queue",
     "statement":"Implement a queue supporting enqueue/dequeue using two stacks."},
    {"name":"Sliding Window Maximum","rating":1700,"tags":"data structures,deque",
     "statement":"Find the maximum in every window of size k in an array using a deque."},

    # ── Math & Number Theory ──────────────────────────────────────────────────
    {"name":"Sieve of Eratosthenes","rating":1100,"tags":"math,number theory,primes",
     "statement":"Find all prime numbers up to N using the Sieve of Eratosthenes."},
    {"name":"GCD and LCM","rating":800,"tags":"math,number theory",
     "statement":"Compute the GCD and LCM of two numbers using Euclid's algorithm."},
    {"name":"Modular Exponentiation","rating":1300,"tags":"math,modular arithmetic",
     "statement":"Compute a^b mod m efficiently using fast exponentiation."},
    {"name":"Euler Totient","rating":1600,"tags":"math,number theory",
     "statement":"Compute Euler's totient function for all numbers up to N."},
    {"name":"Combinatorics Mod P","rating":1500,"tags":"math,combinatorics",
     "statement":"Compute C(n, k) mod p for large n and k using Lucas theorem or precomputed factorials."},

    # ── Greedy ────────────────────────────────────────────────────────────────
    {"name":"Activity Selection","rating":1200,"tags":"greedy,intervals",
     "statement":"Select the maximum number of non-overlapping activities sorted by finish time."},
    {"name":"Fractional Knapsack","rating":1100,"tags":"greedy",
     "statement":"Maximize the total value that can be put into a knapsack if items can be fractionally taken."},
    {"name":"Huffman Encoding","rating":1800,"tags":"greedy,trees",
     "statement":"Build an optimal Huffman encoding tree for a set of characters with given frequencies."},
    {"name":"Jump Game","rating":1300,"tags":"greedy,arrays",
     "statement":"Determine if you can reach the last index given maximum jump lengths at each position."},

    # ── Trees ─────────────────────────────────────────────────────────────────
    {"name":"LCA Binary Tree","rating":1700,"tags":"trees,lca",
     "statement":"Find the lowest common ancestor of two nodes in a binary tree."},
    {"name":"Diameter of Tree","rating":1500,"tags":"trees,dfs",
     "statement":"Find the longest path between any two nodes in an unweighted tree."},
    {"name":"Binary Search Tree Validity","rating":1200,"tags":"trees,bst",
     "statement":"Check if a binary tree satisfies the BST property."},
    {"name":"Level Order Traversal","rating":900,"tags":"trees,bfs",
     "statement":"Print a binary tree level by level using BFS."},

    # ── Geometry ─────────────────────────────────────────────────────────────
    {"name":"Convex Hull","rating":2000,"tags":"geometry,convex hull",
     "statement":"Find the convex hull of a set of 2D points using Graham scan or Jarvis march."},
    {"name":"Point in Polygon","rating":1600,"tags":"geometry",
     "statement":"Determine if a given point lies inside, outside, or on the boundary of a polygon."},

    # ── Miscellaneous ─────────────────────────────────────────────────────────
    {"name":"Game Theory Nim","rating":1800,"tags":"game theory,nim",
     "statement":"Determine who wins the Nim game given pile sizes using XOR analysis."},
    {"name":"Two Pointers Subarray","rating":1200,"tags":"two pointers,arrays",
     "statement":"Find the shortest subarray with sum at least K using two pointers."},
    {"name":"Prefix Sum 2D","rating":1300,"tags":"prefix sums,arrays",
     "statement":"Answer rectangle sum queries on a 2D grid using 2D prefix sums."},
    {"name":"Sparse Table RMQ","rating":1900,"tags":"data structures,sparse table",
     "statement":"Answer range minimum queries in O(1) after O(n log n) preprocessing."},
]

def main():
    out = Path("codeforces_demo.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name","rating","tags","statement","url"])
        writer.writeheader()
        for i, p in enumerate(PROBLEMS, 1):
            p["url"] = f"https://codeforces.com/problemset/problem/{1000+i}/A"
            writer.writerow(p)
    print(f"✅  Demo датасет зачуван: {out}  ({len(PROBLEMS)} задачи)")

if __name__ == "__main__":
    main()
