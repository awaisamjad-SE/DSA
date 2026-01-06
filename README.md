# 📚 COMPLETE DSA GUIDE (Data Structures & Algorithms)
## From Zero to Interview Ready | Easy Language | With Examples | Nothing Skipped

---

# TABLE OF CONTENTS
1. [What is DSA?](#1-what-is-dsa)
2. [Classification of Data Structures](#2-classification-of-data-structures)
3. [Core Data Structures](#3-core-data-structures)
4. [Algorithms](#4-algorithms)
5. [Problem-Solving Patterns](#5-problem-solving-patterns)
6. [How to Solve ANY Problem](#6-golden-method-to-solve-any-problem)
7. [Python Tools & Setup](#7-python-tools-for-dsa)
8. [Preparation Roadmap](#8-preparation-roadmap)
9. [Interview Tips](#9-interview-tips)
10. [Quick Reference](#10-quick-reference)

---

# 1. WHAT IS DSA?

## Definition
**DSA = Data Structures + Algorithms**

- **Data Structure** → How data is stored and organized in memory
- **Algorithm** → Step-by-step logic to solve a problem using that data

## Goal of Learning DSA
✅ Write efficient, fast code
✅ Reduce time complexity (faster execution)
✅ Reduce space complexity (less memory)
✅ Solve any problem systematically
✅ Pass technical interviews

## Real-World Example
**Problem:** Find if a number exists in a list of 1 million numbers

❌ **Without DSA:** Check each number (1 million checks) → SLOW
✅ **With DSA:** Use Binary Search on sorted list (20 checks) → FAST

---

# 2. CLASSIFICATION OF DATA STRUCTURES

## 🔸 A. Primitive Data Structures
Basic building blocks of programming.

```python
int_val = 10
float_val = 3.14
char_val = 'A'
bool_val = True
```

---

## 🔸 B. Non-Primitive Data Structures

### Category 1️⃣: LINEAR DATA STRUCTURES
Data stored in a sequence (one after another)

| Structure | What it is | How stored | When to use |
|-----------|-----------|-----------|------------|
| **Array/List** | Indexed collection | Continuous memory | Fast access |
| **String** | Sequence of characters | Characters array | Text problems |
| **Stack** | LIFO (Last In First Out) | Top pointer | Undo, recursion |
| **Queue** | FIFO (First In First Out) | Front & rear pointer | Scheduling |
| **Linked List** | Nodes with pointers | Non-continuous | Dynamic memory |

#### Example: Visual Representation

```
Array:      [10] [20] [30] [40]  → Access by index: arr[0] = 10
             0    1    2    3

Stack:       ▲ 30 (top - remove first)
            │ 20
            │ 10

Queue:      10 → 20 → 30 → (remove from front, add to back)

Linked List: 10 → 20 → 30 → None
```

---

### Category 2️⃣: NON-LINEAR DATA STRUCTURES
Data stored hierarchically or in networks

| Structure | What it is | When to use |
|-----------|-----------|------------|
| **Tree** | Hierarchical data | Parent-child relationships |
| **Binary Tree** | Max 2 children | Recursion problems |
| **BST** | Left < Root < Right | Fast search |
| **Heap** | Priority-based | Min/Max elements |
| **Graph** | Nodes + edges | Networks, relationships |

#### Example: Visual Representation

```
Tree:              Root
                  /    \
                 /      \
               Node1    Node2
              /    \
            Leaf1  Leaf2

Graph:     A ←→ B
           ↓     ↑
           C ←→ D
```

---

### Category 3️⃣: HASH-BASED STRUCTURES
Fast lookup using keys

| Structure | Python | Use |
|-----------|--------|-----|
| **Hash Table** | `dict` | Frequency counting, lookup |
| **Hash Set** | `set` | Unique values, fast search |

#### Example:

```python
# Hash Table (Dictionary)
freq = {"apple": 3, "banana": 2}  # key → value lookup: O(1)

# Hash Set
unique = {1, 2, 3}  # Fast to check: 2 in unique → O(1)
```

---

# 3. CORE DATA STRUCTURES

## 🔹 3.1 ARRAY / LIST

### What is it?
Stores multiple elements in continuous memory with index-based access.

### Why Important?
- Fastest access → O(1)
- Foundation for other patterns

### Python Code:

```python
# Create
arr = [10, 20, 30, 40]

# Access
print(arr[0])  # 10

# Modify
arr[1] = 25    # [10, 25, 30, 40]

# Add
arr.append(50) # [10, 25, 30, 40, 50]

# Remove
arr.pop()      # Removes last → [10, 25, 30, 40]
```

### Time Complexity:

| Operation | Time |
|-----------|------|
| Access by index | O(1) |
| Search | O(n) |
| Insert | O(n) |
| Delete | O(n) |

### When to Use:
✅ Store multiple values
✅ Random access needed
✅ Known size

---

## 🔹 3.2 STACK (LIFO - Last In First Out)

### What is it?
Last element added is first to come out (like a plate stack).

### Visual:

```
Push 10:    [10]
Push 20:    [10, 20]
Push 30:    [10, 20, 30]
Pop:        [10, 20]  ← Removed 30
Pop:        [10]      ← Removed 20
```

### Python Code:

```python
# Using list as stack
stack = []

# Push (add)
stack.append(10)
stack.append(20)
stack.append(30)

# Pop (remove)
top = stack.pop()  # 30
print(stack)       # [10, 20]

# Peek (see without removing)
print(stack[-1])   # 20
```

### Time Complexity:

| Operation | Time |
|-----------|------|
| Push | O(1) |
| Pop | O(1) |
| Peek | O(1) |

### Real-World Uses:
- Browser back button
- Undo in text editor
- Function call stack
- Valid parentheses problem

### Example Problem: Valid Parentheses

```python
def is_valid(s):
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in s:
        if char in pairs:
            stack.append(char)
        else:
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0

print(is_valid("()[]{}"))  # True
print(is_valid("([)]"))    # False
```

---

## 🔹 3.3 QUEUE (FIFO - First In First Out)

### What is it?
First element added is first to come out (like a line at counter).

### Visual:

```
Enqueue 10:    [10]
Enqueue 20:    [10, 20]
Enqueue 30:    [10, 20, 30]
Dequeue:       [20, 30]  ← Removed 10
Dequeue:       [30]      ← Removed 20
```

### Python Code:

```python
from collections import deque

# Create
queue = deque()

# Enqueue (add to back)
queue.append(10)
queue.append(20)
queue.append(30)

# Dequeue (remove from front)
front = queue.popleft()  # 10
print(queue)             # deque([20, 30])

# Peek
print(queue[0])          # 20
```

### Time Complexity:

| Operation | Time |
|-----------|------|
| Enqueue | O(1) |
| Dequeue | O(1) |

### Real-World Uses:
- Task scheduling
- BFS (Breadth-First Search)
- Printer queue
- Customer service lines

---

## 🔹 3.4 LINKED LIST

### What is it?
Nodes connected by pointers (not continuous memory).

### Visual:

```
Singly: 10 → 20 → 30 → None

Doubly: None ← 10 ↔ 20 ↔ 30 → None

Circular: 10 → 20 → 30 → (back to 10)
```

### Node Structure:

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
```

### Complete Linked List Implementation:

```python
class LinkedList:
    def __init__(self):
        self.head = None
    
    # Insert at beginning
    def insert_at_beginning(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    # Insert at end
    def insert_at_end(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        temp = self.head
        while temp.next:
            temp = temp.next
        temp.next = new_node
    
    # Delete from beginning
    def delete_from_beginning(self):
        if self.head:
            self.head = self.head.next
    
    # Print list
    def display(self):
        temp = self.head
        while temp:
            print(temp.data, end=" → ")
            temp = temp.next
        print("None")
    
    # Reverse list
    def reverse(self):
        prev = None
        curr = self.head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        self.head = prev

# Usage
ll = LinkedList()
ll.insert_at_end(10)
ll.insert_at_end(20)
ll.insert_at_end(30)
ll.display()      # 10 → 20 → 30 → None
ll.reverse()
ll.display()      # 30 → 20 → 10 → None
```

### Time Complexity:

| Operation | Time |
|-----------|------|
| Access | O(n) |
| Search | O(n) |
| Insert | O(n) |
| Delete | O(n) |

### When to Use:
✅ Dynamic memory needed
✅ Frequent insertions/deletions
✅ Unknown size

### Important Concepts:

**Fast & Slow Pointer (Find Middle):**
```python
slow = fast = self.head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
return slow.data
```

**Detect Cycle (Floyd's Algorithm):**
```python
slow = fast = self.head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        return True  # Cycle detected
return False
```

---

## 🔹 3.5 HEAP (Priority Queue)

### What is it?
Always returns the smallest (min-heap) or largest (max-heap) element.

### Visual (Min-Heap):

```
        1
      /   \
     3     2
    / \   /
   7   4 6
```

### Python Code:

```python
import heapq

# Create min-heap
heap = []

# Insert (heappush)
heapq.heappush(heap, 30)
heapq.heappush(heap, 10)
heapq.heappush(heap, 20)

# Remove minimum
min_val = heapq.heappop(heap)  # 10
print(min_val)

# Peek
print(heap[0])  # Smallest

# Create heap from list
arr = [30, 10, 20]
heapq.heapify(arr)  # Convert to heap

# For max-heap (negate values)
max_heap = []
heapq.heappush(max_heap, -30)
heapq.heappush(max_heap, -10)
max_val = -heapq.heappop(max_heap)  # 30
```

### Time Complexity:

| Operation | Time |
|-----------|------|
| Insert | O(log n) |
| Remove min | O(log n) |
| Peek min | O(1) |

### Real-World Uses:
- Top K elements
- Dijkstra's shortest path
- Huffman coding
- Task scheduling

### Example: Top K Largest Elements

```python
def topKLargest(nums, k):
    heap = nums[:k]
    heapq.heapify(heap)
    
    for i in range(k, len(nums)):
        if nums[i] > heap[0]:
            heapq.heapreplace(heap, nums[i])
    
    return heap

print(topKLargest([3,2,1,5,6,4], 2))  # [5, 6]
```

---

## 🔹 3.6 TREE

### What is it?
Hierarchical data structure with parent-child relationships.

### Types:

**Binary Tree:** Each node has max 2 children
```
     1
    / \
   2   3
```

**Binary Search Tree (BST):** Left < Root < Right
```
     5
    / \
   3   7
  / \ / \
 1 4 6  8
```

### Tree Traversal:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# In-Order (Left → Root → Right) - Sorted for BST
def inorder(root):
    if not root:
        return
    inorder(root.left)
    print(root.val)
    inorder(root.right)

# Pre-Order (Root → Left → Right)
def preorder(root):
    if not root:
        return
    print(root.val)
    preorder(root.left)
    preorder(root.right)

# Post-Order (Left → Right → Root)
def postorder(root):
    if not root:
        return
    postorder(root.left)
    postorder(root.right)
    print(root.val)

# Level-Order (BFS)
from collections import deque

def levelorder(root):
    if not root:
        return
    queue = deque([root])
    while queue:
        node = queue.popleft()
        print(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
```

### Time Complexity:

| Operation | Time |
|-----------|------|
| Search (BST) | O(log n) average |
| Insert (BST) | O(log n) average |
| Delete (BST) | O(log n) average |
| Traverse | O(n) |

---

## 🔹 3.7 GRAPH

### What is it?
Nodes (vertices) connected by edges (relationships).

### Types:

**Undirected:** Edges go both ways
```
A --- B
|     |
C --- D
```

**Directed:** Edges have direction
```
A → B
↓   ↑
C → D
```

### Representation:

```python
# Adjacency List (Most used)
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A'],
    'D': ['B']
}

# Adjacency Matrix
import numpy as np
matrix = np.array([
    [0, 1, 1, 0],  # A connections
    [1, 0, 0, 1],  # B connections
    [1, 0, 0, 0],  # C connections
    [0, 1, 0, 0]   # D connections
])
```

### BFS (Breadth-First Search):

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        print(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Time: O(V + E), Space: O(V)
```

### DFS (Depth-First Search):

```python
def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(node)
    print(node)
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Time: O(V + E), Space: O(V)
```

### Use Cases:
- **BFS:** Shortest path, level-order
- **DFS:** Cycle detection, topological sort

---

## 🔹 3.8 TRIE (Prefix Tree)

### What is it?
Tree-like structure for efficient string/prefix searching.

### Visual:

```
        root
      / | | \
     a  c d  o
     |  |    |
     p  a    g
     |  |
     p  t
```

### Python Implementation:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Usage
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))     # True
print(trie.starts_with("app"))  # True
```

### Time Complexity:

| Operation | Time |
|-----------|------|
| Insert | O(m) - m = word length |
| Search | O(m) |
| Starts with | O(m) |

### Use Cases:
- Autocomplete
- Spell checking
- Dictionary
- IP routing

---

# 4. ALGORITHMS

## 🔹 4.1 SEARCHING ALGORITHMS

### Linear Search:

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Time: O(n), Space: O(1)
```

### Binary Search (For Sorted Array):

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Time: O(log n), Space: O(1)
```

---

## 🔹 4.2 SORTING ALGORITHMS

### Bubble Sort (Simple but Slow):

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Time: O(n²), Space: O(1)
```

### Merge Sort (Fast & Reliable):

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Time: O(n log n), Space: O(n)
```

### Quick Sort (Fastest Average):

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# Time: O(n log n) average, O(n²) worst, Space: O(log n)
```

---

## 🔹 4.3 RECURSION

### What is it?
Function calling itself until a base case is reached.

### Basic Structure:

```python
def recursive_function(n):
    # Base case (when to stop)
    if n == 0:
        return 1
    
    # Recursive case
    return n * recursive_function(n - 1)

print(recursive_function(5))  # 120 (5!)
```

### Important Concepts:

**Base Case:** When recursion stops
**Recursive Case:** When function calls itself

### Examples:

```python
# Factorial
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

# Fibonacci
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Binary Search
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1
    
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

### Time & Space:

| Concept | Time | Space |
|---------|------|-------|
| Factorial | O(n) | O(n) - call stack |
| Fibonacci | O(2ⁿ) | O(n) - call stack |

---

# 5. PROBLEM-SOLVING PATTERNS

## 🔹 5.1 TWO POINTERS

### What is it?
Use two indices to traverse data instead of nested loops.

### When to Use:

| Problem Type | Example |
|--------------|---------|
| Pair sum | Find two numbers that sum to target |
| Remove duplicates | Remove duplicates from sorted array |
| Reverse | Reverse array/string |
| Sorted array | Many sorted array problems |

### Techniques:

**Opposite Direction (left → right):**
```python
def reverse_string(s):
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    return s

# Time: O(n), Space: O(1)
```

**Same Direction (slow → fast):**
```python
def remove_duplicates(arr):
    if not arr:
        return 0
    
    slow = 0
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]
    
    return slow + 1

# Time: O(n), Space: O(1)
```

### Example Problem: Two Sum

```python
def two_sum(arr, target):
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [arr[left], arr[right]]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return None

print(two_sum([1, 2, 3, 7, 8], 10))  # [2, 8]
```

---

## 🔹 5.2 SLIDING WINDOW

### What is it?
Maintain a window (range) and slide it to avoid recalculating.

### When to Use:

| Problem | Example |
|---------|---------|
| Subarray | Max sum of subarray |
| Substring | Longest substring without repeating |
| Continuous | Any continuous elements problem |

### Example Problem: Maximum Sum of k Elements

```python
def max_sum_k_elements(arr, k):
    # Calculate sum of first k elements
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

print(max_sum_k_elements([1, 4, 2, 10, 2, 3, 1, 0, 20], 4))  # 24
# Time: O(n), Space: O(1)
```

### Example Problem: Longest Substring Without Repeating

```python
def longest_substring_without_repeating(s):
    char_map = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        if s[right] in char_map and char_map[s[right]] >= left:
            left = char_map[s[right]] + 1
        
        char_map[s[right]] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length

print(longest_substring_without_repeating("abcabcbb"))  # 3 ("abc")
# Time: O(n), Space: O(26) = O(1)
```

---

## 🔹 5.3 HASH MAP

### What is it?
Key → Value mapping for fast lookup.

### When to Use:

| Problem | Example |
|---------|---------|
| Count frequency | Count character/number frequency |
| Duplicates | Find duplicates |
| Lookup | Check if element exists |
| Pairs | Two sum, pair finding |

### Example Problems:

**Count Frequency:**
```python
def count_frequency(arr):
    freq = {}
    for num in arr:
        freq[num] = freq.get(num, 0) + 1
    return freq

print(count_frequency([1, 2, 2, 3, 3, 3]))  # {1: 1, 2: 2, 3: 3}
# Time: O(n), Space: O(n)
```

**Two Sum:**
```python
def two_sum(arr, target):
    seen = {}
    for i, num in enumerate(arr):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
    return []

print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
# Time: O(n), Space: O(n)
```

---

## 🔹 5.4 STACK

### When to Use:

| Problem | Pattern |
|---------|---------|
| Valid brackets | Check balanced parentheses |
| Undo/Redo | Browser history |
| Next greater | Next greater element |
| Monotonic | Maintain increasing/decreasing order |

### Example: Valid Parentheses (Already covered in section 3.2)

### Example: Next Greater Element

```python
def next_greater_element(arr):
    stack = []
    result = [-1] * len(arr)
    
    for i in range(len(arr) - 1, -1, -1):
        while stack and stack[-1] <= arr[i]:
            stack.pop()
        
        if stack:
            result[i] = stack[-1]
        
        stack.append(arr[i])
    
    return result

print(next_greater_element([1, 5, 0, 3, 4, 5]))
# [5, -1, 3, 4, 5, -1]
# Time: O(n), Space: O(n)
```

---

## 🔹 5.5 BINARY SEARCH

### When to Use:

| Problem | Example |
|---------|---------|
| Sorted data | Search in sorted array |
| First/Last | Find first/last position |
| Min/Max | Find minimum/maximum |
| Answer space | Search in answer space |

### Basic Template:

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Time: O(log n), Space: O(1)
```

### Example: Find First Position

```python
def find_first_position(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

print(find_first_position([5, 7, 7, 8, 8, 10], 8))  # 3
```

---

## 🔹 5.6 RECURSION (Already covered in section 4.3)

---

## 🔹 5.7 DYNAMIC PROGRAMMING (DP)

### What is it?
Save results of overlapping subproblems to avoid recomputation.

### When to Use:

| Problem | Pattern |
|---------|---------|
| Maximum/Minimum | Max profit, min cost |
| Fibonacci | Overlapping subproblems |
| Knapsack | Optimization |
| Grid paths | Count paths |

### Key Insight:
**Optimal Substructure:** Solution depends on solutions to subproblems

### Example: Fibonacci

```python
# Without DP (SLOW - O(2ⁿ))
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# With DP - Memoization (FAST - O(n))
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]

# With DP - Tabulation (FAST - O(n))
def fib_tab(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

print(fib_tab(10))  # 55
```

### Example: Coin Change

```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

print(coin_change([1, 2, 5], 5))  # 1 (one 5-coin)
# Time: O(amount × coins), Space: O(amount)
```

### DP States:
```python
# 1D DP: Linear problems (Fibonacci, Coin Change)
dp = [0] * n

# 2D DP: Grid/path problems
dp = [[0] * m for _ in range(n)]
```

---

## 🔹 5.8 GREEDY

### What is it?
Choose the best local option at each step hoping for global optimum.

### When to Use:

| Problem | Approach |
|---------|----------|
| Activity selection | Choose non-overlapping activities |
| Huffman coding | Build optimal tree |
| Coin change (some) | Greedy coins |
| Interval scheduling | Sort by end time |

### Example: Activity Selection

```python
def activity_selection(activities):
    # Sort by end time
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_end = activities[0][1]
    
    for start, end in activities[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end
    
    return selected

activities = [(1, 3), (2, 5), (4, 6), (6, 7)]
print(activity_selection(activities))
# [(1, 3), (4, 6), (6, 7)]
# Time: O(n log n), Space: O(1)
```

---

## 🔹 5.9 BACKTRACKING

### What is it?
Try → Check → Undo → Try next

### When to Use:

| Problem | Example |
|---------|---------|
| Permutations | All permutations |
| Combinations | All combinations |
| Puzzles | Sudoku, N-Queens |
| Subsets | All subsets |

### Example: Permutations

```python
def permute(nums):
    result = []
    
    def backtrack(path, remaining):
        if not remaining:
            result.append(path)
            return
        
        for i, num in enumerate(remaining):
            backtrack(path + [num], remaining[:i] + remaining[i+1:])
    
    backtrack([], nums)
    return result

print(permute([1, 2, 3]))
# [[1, 2, 3], [1, 3, 2], [2, 1, 3], ...]
# Time: O(n!), Space: O(n!)
```

### Example: Subsets

```python
def subsets(nums):
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

print(subsets([1, 2, 3]))
# [[], [1], [1, 2], [1, 2, 3], ...]
```

---

## 🔹 5.10 GRAPH - BFS & DFS

### Already covered in section 3.7

---

## 🔹 5.11 TRIE

### Already covered in section 3.8

---

## 🔹 5.12 PREFIX SUM

### What is it?
Pre-calculate cumulative sums for range queries.

### When to Use:

| Problem | Example |
|---------|---------|
| Range sum | Sum from index i to j |
| Subarray sum | Count subarrays with sum k |
| 2D range sum | Sum in 2D grid |

### Example: Range Sum Query

```python
class PrefixSum:
    def __init__(self, nums):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)
    
    def range_sum(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]

ps = PrefixSum([1, 3, 2, 7, 4])
print(ps.range_sum(1, 3))  # 3 + 2 + 7 = 12
# Setup: O(n), Query: O(1)
```

---

## 🔹 5.13 MONOTONIC STACK

### When to Use:

| Problem | Example |
|---------|---------|
| Next greater | Next greater element |
| Histogram | Largest rectangle |
| Stock span | Stock span problem |

### Example: Next Greater Element (Already covered in section 5.4)

---

## 🔹 5.14 UNION FIND (Disjoint Set)

### What is it?
Quickly check if two nodes are connected.

### Python Implementation:

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        
        if px == py:
            return False
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)

# Time: Nearly O(1) per operation
```

---

## 🔹 5.15 BIT MANIPULATION

### When to Use:

| Problem | Trick |
|---------|-------|
| Single number | XOR all: n ^ n = 0 |
| Power of 2 | n & (n-1) == 0 |
| Bit operations | Set, unset, toggle bits |

### Example: Single Number

```python
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num  # XOR: a ^ a = 0, a ^ 0 = a
    return result

print(single_number([4, 1, 2, 1, 2]))  # 4
# Time: O(n), Space: O(1)
```

---

# 6. GOLDEN METHOD TO SOLVE ANY PROBLEM

## Step 1️⃣: UNDERSTAND THE PROBLEM

Ask yourself:
```
❓ What is the input?
❓ What is the output?
❓ What are the constraints?
❓ Are there edge cases?
```

## Step 2️⃣: IDENTIFY THE PATTERN

Look for keywords in the problem:

| Keyword | Pattern |
|---------|---------|
| Sorted | Binary Search |
| Subarray/Substring | Sliding Window |
| Pair/Reverse | Two Pointers |
| Count/Frequency | Hash Map |
| Max/Min | Greedy, Heap |
| All choices/subsets | DP, Backtracking |
| Connected/Path | Graph, BFS/DFS |
| Prefix/Search | Trie |
| Range sum | Prefix Sum |
| Next greater | Monotonic Stack |

## Step 3️⃣: CHOOSE DATA STRUCTURE

Based on operation needed:

| Operation | Use |
|-----------|-----|
| Count elements | dict |
| Unique values | set |
| Ordered priority | heap |
| FIFO order | queue (deque) |
| LIFO order | stack (list) |
| Fast lookup | dict/set |

## Step 4️⃣: DRY RUN (VERY IMPORTANT ✏️)

Always simulate with small input:

```python
Example: arr = [1, 2, 3]

Step by step:
- i=0: process 1
- i=1: process 2
- i=2: process 3
Result: expected output ✓
```

## Step 5️⃣: CODE IN PYTHON

Keep it:
- Simple
- Readable
- Efficient

## Step 6️⃣: ANALYZE COMPLEXITY

Always state:
- **Time Complexity:** How fast
- **Space Complexity:** How much memory

---

# 7. PYTHON TOOLS FOR DSA

## Essential Imports

```python
from collections import defaultdict, Counter, deque
import heapq
import bisect
from functools import lru_cache
```

## Important Tools:

| Tool | Use | Example |
|------|-----|---------|
| `dict` | Frequency | `freq = {1: 2, 2: 1}` |
| `set` | Unique | `unique = {1, 2, 3}` |
| `deque` | Queue | `q = deque(); q.append(x); q.popleft()` |
| `heapq` | Priority | `heapq.heappush(h, x); heapq.heappop(h)` |
| `sorted()` | Sort | `sorted(arr); sorted(arr, key=lambda x: -x)` |
| `enumerate` | Index | `for i, val in enumerate(arr)` |
| `zip()` | Pair | `zip([1,2], ['a','b'])` → `[(1,'a'), (2,'b')]` |
| `Counter` | Frequency | `Counter([1, 1, 2])` → `Counter({1: 2, 2: 1})` |
| `defaultdict` | Default value | `dd = defaultdict(list)` |

---

# 8. PREPARATION ROADMAP

## 🟢 Phase 1: Basics (1–2 Weeks)

### Topics:
- ✅ Arrays/Lists
- ✅ Strings
- ✅ Hashing (dict, set)
- ✅ Recursion

### Practice:
- Simple array problems
- String manipulation
- Counting/grouping

---

## 🟡 Phase 2: Intermediate (2–3 Weeks)

### Topics:
- ✅ Stack & Queue
- ✅ Linked List
- ✅ Two Pointers
- ✅ Sliding Window

### Practice:
- Bracket matching
- Linked list operations
- Pair/range problems

---

## 🔴 Phase 3: Advanced (3–4 Weeks)

### Topics:
- ✅ Trees & BST
- ✅ Graphs (BFS/DFS)
- ✅ Dynamic Programming
- ✅ Backtracking
- ✅ Heap

### Practice:
- Tree traversal
- Shortest paths
- Optimization problems

---

## 📅 Total Timeline: 6–9 Weeks

## Daily Practice:

```
Monday: Learn 1 pattern + code 2 problems
Tuesday: Dry run 3 problems on paper
Wednesday: Optimize solutions
Thursday: Learn new concept
Friday: Review + interview questions
Weekend: Mix and match all patterns
```

---

# 9. INTERVIEW TIPS

## 🎤 How to Structure Your Answer

```
1. Clarify the problem
   "Do I need to handle empty arrays?"

2. State your approach
   "I'll use a hash map for O(1) lookup."

3. Time & Space
   "Time: O(n), Space: O(n)"

4. Code it
   "Let me write the code..."

5. Test with example
   "Let me trace through with [1, 2, 3]..."
```

## ❌ Don'ts in Interview

- ❌ Don't memorize code
- ❌ Don't code without thinking
- ❌ Don't forget edge cases
- ❌ Don't skip complexity analysis

## ✅ Do's in Interview

- ✅ Ask clarifying questions
- ✅ Think out loud
- ✅ Optimize if time permits
- ✅ Test edge cases

---

# 10. QUICK REFERENCE

## Complexity Cheat Sheet

| Complexity | Description | Example |
|-----------|-------------|---------|
| O(1) | Constant | Direct access, dict lookup |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Single loop |
| O(n log n) | Linearithmic | Merge sort, quick sort |
| O(n²) | Quadratic | Nested loop |
| O(2ⁿ) | Exponential | All subsets |
| O(n!) | Factorial | All permutations |

## Pattern Quick Lookup

```python
# Sorted array → Binary Search
# Subarray problem → Sliding Window
# Count something → Hash Map / Counter
# Multiple states → Dynamic Programming
# All combinations → Backtracking
# Graph problem → BFS / DFS
# String search → Trie
# Min/Max element → Heap
# Connected components → Union Find
# XOR problems → Bit Manipulation
```

## Common Code Templates

### Binary Search Template:
```python
left, right = 0, len(arr) - 1
while left <= right:
    mid = (left + right) // 2
    if condition:
        # Found or move
    elif arr[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
```

### DFS Template:
```python
def dfs(node, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, visited)
```

### BFS Template:
```python
from collections import deque
queue = deque([start])
visited = {start}
while queue:
    node = queue.popleft()
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)
```

### DP Template:
```python
dp = [0] * n  # or [[0]*m for _ in range(n)]
dp[0] = base_case
for i in range(1, n):
    dp[i] = recurrence_relation(dp[i-1])
return dp[n-1]
```

---

# FINAL TIPS FOR SUCCESS

## 🎯 Remember:

1. **DSA is about PATTERNS, not memorization**
2. **Master these 5 concepts:**
   - Arrays & Two Pointers
   - Hashing
   - Recursion & DP
   - Graphs (BFS/DFS)
   - Trees

3. **Practice daily:** 1-2 problems per day
4. **Focus on WHY, not HOW**
5. **Always analyze complexity**

## 🔥 You Can Do This!

If you master the patterns in this guide, you can solve 90% of DSA problems in interviews.

**Good luck! 🚀**

---

## Made with ❤️ for Interview Success
## Awais Amjad BSCS @ UET Python Developer
