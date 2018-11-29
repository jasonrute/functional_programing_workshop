# Functional Programming for Data Scientists Who Only Know Python

## Functional programming

### Brief history
In the 1920s and 1930s there were two  theories of computation.

  - **Turing machine** (Alan Turing, 1932) 
    - **step-by-step model** of computation based on a physical (albeit theoretical) machine.  
    - gave rise to **imperative programming languages**:
        - Fortran
        - C/C++
        - Java
        - Python
        - ...
  - **Lambda Calculus** (Alonzo Church, 1930s) and **Combinatorial Logic** (Haskell Curry, 1927)
    - **function based models** based on pure abstract mathematics.  
    - gave rise to **functional programming languages**: 
        - Lisp
        - Haskell
        - SML/OCaml
        - Clojure
        - Scala
        - ...     

### What is functional programming?
  1. A **category of programming languages** (see above)
  2. A **paradigm of programming** which also applies to imperative languages.

We will focus on **the functional programming paradigm** (i.e. patterns/principles) with examples in Python 3.  (However, I am secretly teaching you how to program in a purely functional language like Scala.)

### FP principles at a glance

 1. Use functions as values (which can be passed to other functions)
 2. Write pure functions (with no side effects)
 3. Program declaratively, not imperatively
     1. Composition not steps
     2. Avoid  `for`  and  `while`  loops
     3. Inline  `if then else`  (also  `cases`)
     4. Use pattern matching (deconstruction)
 4. Use immutable types
 5. Consider type safety
 6. Embrace type theory

***Warning** Python is not a purely functional language and following this advice exactly may cause you to write bad Python code.  Use common sense.*

### Why functional programming?

  - Mathematically elegant
      - *You probably don't care!*
  - Modular/Readable/Simple programming style
      - This subjective at best, but there is some truth to it 
  - Less prone to (certain) bugs
      - This is really important for some applications
  - Naturally suited for parallel and vectorized code
      - This is a big deal, especially in (big) data science

## Functions and functional programming

### FP Principle 1:  Use functions as values (which can be passed to other functions)

```python
# function
def plus_one(x):
    return x + 1

# higher order function (takes a function f as an argument)
def repeat_twice(f):
    # use input function f to construct a new function g = f o f
    def g(x):
        return f(f(x))
    # return this function
    return g

plus_two = repeat_twice(plus_one) # plus_two is a function
plus_two(0)                       # == plus_one(plus_one(0)) == 2
```

### λ notation

For simple functions, can use `lambda`.
```python
# lambda functions (not how they are typically used)
plus_one = lambda x: x + 1
plus_two = lambda x: plus_one(plus_one(x))
plus_two(0)  # == plus_one(plus_one(0)) == 2
```

### Map functions
```python
l    # list object
s    # pd.Series object
df   # pd.DataFrame object

# apply function to every element of the list (rare to see list.map in Python)
l.map(lambda x: x + 1)      # a copy of l with every item incremented

# apply function to every series element
s.map(lambda x: x * x)      # a copy of s with every item squared

# apply function to every dataframe column
df.apply(lambda x: x.sum()) # a series containing the sum of each column
```

### Aside: Object oriented programming
  Languages like Python, Java, and C++ are **object oriented**.  

In Python **everything is an object.**  Objects have attached attributes and methods which are attached to the object via the `.` notation.

```python
my_array = np.array([[1, 2, 3], [4, 5, 6])
my_list = [1, 2, 3]
my_float = 7.5
my_string = "HI THERE"
my_function = len

# methods
my_list.append(7)
my_number.hex()       # == 0x1.e000000000000p+2
my_string.lower()     # == "hi there"

# attributes
my_array.shape      # == (2, 3)
my_function.__doc__ # == 'Return the number of items in a container.'
```

### Functional and object oriented programming are compatable 
Many purely functional programming languages are also object oriented, including
- OCaml
- Scala

### FP Principle 2: Write pure functions (with no side effects)

```python
# NOT PURE!!!!
def third_largest_item(list_of_nums):
    global global_variable
    
    list_of_nums.sort()   # <-- Side effect: list_of_nums has been is modified
    global_variable += 1  # <-- Side effect: changing a global variable
    print("Hello")        # <-- Side effect: input/output operations

    return list_of_nums[-3] 

# Pure
def third_largest_item(list_of_nums):
    new_list = list(sorted(list_of_nums))  # list_of_nums is not changed
    return new_list[-3]
```

Pure functions perform the same way every time. 

## Coding Style and Functional Programming

### FP Principle 3: Program declaratively, not imperatively

  - **Imperative:** 
    - Focused on **steps** (the "how").  
    - Closer to the actual machine steps.
    - Think "recipes"
  - **Declarative:** 
    - Focused on **results** (the "what").  
    - Closer to the desired results.
    - Think "formulas"

```python
"""
find the minimum word (alphabetically) in a string
"""

# imperative (focused on steps)
def min_word(s):
    # make s lower case
    lower_s = ""
    for c in s:
        lower_s += c.lower()

    # split new_s into words
    words = []
    w = ""
    for c in new_s:
        if c == " ":
            words.append(w)
            w == ""
        else:
            w += c
    w += c

    # find lowest word alphabetically
    min_word = ""
    for w in words:
        if w < min_word:
            min_word = w

    return min_word

# declaritive (focused on results)
def min_words(s):
    return min(s.lower().split())
```

SQL code is purely declaritive (no steps, only results).
```sql 
SELECT gender, SUM(income)
FROM income_list
WHERE age > 20
GROUP BY gender
ORDER BY SUM(income);
```

Excel formulas are another example of purely declaritive code.
```  
= SUM(INDEX(C5:F11,0,MATCH(I6,C4:F4,0)))
```
Functional programming prefers declaritive code over imperative code.

### Declaritive principle: Composition not steps

```python
# imperative (steps)
def my_function(x):
    x = f(x) # step 1
    x = g(x) # step 2
    x = h(x) # step 3
    return w

# declaritive (function composition)
def my_function(x):
    return h(g(f(x)))
```

With object oriented programming, it is possible to write code declaritively, but still preserve the "natural" order of the operations:
```python
# imperative (steps)
x = "This is a string"
x = x.lower()           # make lower case
x = x.split()           # split into a list of words
x = x.index("is")       # find the index of the first occurrence of the word "is"

# declarative (function composition)
x = ("This is a string"
        .lower()        # make lower case
        .split()        # split into a list of words
        .index("is"))   # find the index of the first occurrence of the word "is"
```

***Python Pro Tip:** Use parentheses on both sides of an expression to break it into multiple lines.  You can line things up and insert comments as desired.*   

Data science example of imperative/declarative style.  Here we are plotting the number of rows per week.
```python
# imperative
def plot_number_of_rows_per_week(df):
    df1 = df.copy()                    # we don't want to change the original df
    
    weeks = df1['date_time'].dt.week() # get week of the year
    df1['week'] = weeks                # add column for the week of the year
    
    df2 = df1.groupby('week').size()   # count number of rows for each week
    df2.plot()                         # plot

# declarative
def plot_number_of_rows_per_week(df):
    (df.groupby(df.date_time.dt.week())  # group by week of the year
       .size()                           # count rows for each week
       .plot())                          # plot
```

### Declaritive principle: Avoid `for` and `while` loops (but "`for` comprehensions" are ok)

#### Recursion
Many loop constructs can be replaced with recursion.  This is especially true of algorithms for trees.
```python
class Tree:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

"""
Check if a certain value is in a tree
"""

# imperative (while loop + a stack)
def contains(tree, value):
    to_visit = [tree]  # list of nodes to visit
    while to_visit:
        node = to_visit.pop() # visit node
        
        if node is None:
            continue
        elif node.val == value:
            return True
        else:
            # explore child nodes
            to_visit.append(node.left)
            to_visit.append(node.right)
    
    return False

# declaritive (recursion)
def contains(tree, value):
    # base case
    if tree is None:
        return False
    
    # recursion
    return (tree.val == value 
              or contains(tree.left, val) 
              or contains(tree.right, val))
```

#### List comprehension
Simple loops (especially those which construct lists or strings) can be accomplished via list comprehension
```python
"""
Copy list and add one to every item.
"""

# imperative (for loop)
def add_one_to_each_item(old_list):
    new_list = []
    for n in old_list:
        new_list.append(n+1)
       
    return old_list
    
# declaritive (list comprehension)
def add_one_to_each_item(old_list):
    return [n + 1 for n in old_list] # list comprehension
```

Note: `[n + 1 for n in old_list]` is equivalent to `old_list.map(lambda n: n + 1)`, but more common in Python.

```python
"""
Divide all even numbers by 2.
"""

# imperative (for loop)
def divide_by_two_if_even(old_list):
    new_list = []
    for n in old_list:
        if n % 2 == 0: # even
            new_list.append(n // 2)
           
  return old_list
    
# declaritive (list comprehension)
def add_one_to_each_item(old_list):
    return [n // 2 for n in old_list if n % 2 == 0]
```

```python
"""
Reverse each word in a string.
e.g. "This is a sentence." -> "sihT si a .ecnetnes"
"""

# imperative (for loop)
def reverse_each_word(sentence):
    words = sentence.split() # split sentences
    s = ""
    for w in words:
        s += " " + w[::-1]  # add space and reversed word  
    return s[1:] # remove leading space
    
# declaritive (list comprehension and join)
def reverse_each_word(sentence):
    return " ".join([w[::-1]                       # reverse word 
                     for w in sentence.split()])   # for all words in the sentence
```

There are other types of comprehension.
```python
# list comprehension
[n for n in range(4) if n != 2]         # == [0, 1, 3]

# set comprehension
{n for n in range(4) if n != 2}         # == {0, 1, 3} 

# dictionary comprehension
{n:n+1 for n in range(4) if n != 2}     # == {0:1, 1:2, 3:4} 

# tuple comprehension
tuple(n for n in range(4) if n != 2)    # == (0, 1, 4)

# iterator comprehension
(n for n in range(4) if n != 2)         # an iterator (advanced concept)
```

Can also have nested loops:
```python
# list comprehension with multiple loops
[a + b for a in "AB" for b in "YZ"]   # == ['AY', 'AZ', 'BY', 'BZ']
```

#### Reduce (aggregation) operations
```python
sum(list_of_nums)              # sum numbers
min(list_of_nums)              # find smallest number
max(list_of_strings, key=len)  # find longest word
"".join(list_of_strings)       # concatenate strings
reversed(my_list)              # reverse the list (returns an iterator)
all(list_of_bools)             # True if all conditions are True
any(list_of_bools)             # True if at least one condition is True
```

### Declaritive principle: Inline `if then else` logic (also `cases`)
```python
# imperative
def sign(n):
    if n < 0:
        return -1
    elif n == 0:
        return 0
    else:
        return 1

# declaritive
def sign(n):
    return (-1 if n < 0 
               else (0 if n == 0 else 1))
```

The inline `... if ... else ...` of Python is a bit awkward to use.  Most other programming languages have a more natural inline `if ... then ... else ...` structure.  SQL has `CASES` which has a similar functionality.

```sql
SELECT 
    CASES 
        WHEN age < 16 THEN 'child age'
        WHEN age < 65 THEN 'working age'
        ELSE 'retirement age'
    END AS age_category
FROM income_list; 
``` 

Nonetheless, Python's `if else` logic is useful for some one-liners.
```python
def replace_negative_numbers_with_0(number_list):
    return [0 if x < 0 else x for x in number_list]
```

### Declaritive principle: Use pattern matching (deconstruction)

```python
# pattern matching
x, _, y = [1, 2, 3]            # set x = 1 and y = 3 (ignore 2)
a, (b, c), d = (1, (2, 3), 4)  # set a = 1, b = 2, c = 3, d = 4

# swapping variables
a, b = b, a                    # swaps a and b
n_old, n = n, n + 1            # simaltaneously set n_old = n and n = n + 1

# cool iterators which loop over pairs
for index, item in enumerate(my_list):  # loop over index/items pairs
    pass 
    
for a, b in zip(list_a, list_b):        # loop over list_a and list_b in parallel
    pass 

for key, value in my_dict.items():      # loop over dictionary key/value pairs
    pass 

for i, (a, b) in enumerate(zip(list_a, list_b)):
    pass
    
```

Purely functional languages (e.g. Haskell) have really powerful pattern matching abilities.

### <strike>Declaritive</strike> *Universal* principle: Use built in functions and tools

The following Python packages are really useful
- `collections`: useful types such as `defaultdict` and `deque`
- `itertools`: useful tools for working with iterators
- `functools`: useful tools for working with functions

### <strike>Declaritive</strike> *Universal* principle: Break up your code into managable pieces.  Reuse code.
You are are writing the same code often:
 - Check if that functionality is already in a standard Python package.
 - Turn your code into a function (or class or decorator).
 - Write tests for your function.
 - Replace your old code with that function.

### Discussion

Python is not a functional language and does not have the tools to write in a purely declaritive style all the time.

- Functional programming must be written declaritively.
- Imperative programming may be written declaritively (when appropriate).
 
**The goal is to write working, readable, maintainable code.  Use common sense.**

## Parallelization and Vectorization and Functional Programming

### Vectorization
Numpy and Pandas (as well as R, Matlab, Tensorflow, and probabilistic programming languages) are based on vector calculations.  Vectorized operations are faster since they are done in C, Fortran, or the GPU.  Vectorized programming is also more declarative.
```python
# fast vectorized operations (declarative)
y = np.matmul(w, x) + b
``` 

### Parallelization
Computer science is moving to massively parallel programming.  Writing declarative code with pure functions which have no side effects goes a long way to making code which is naturally parallelizable.

### MapReduce
Google's MapReduce is an example of how functional/declarative programming has influenced data science.  Suppose one wants to count the instances of every word on the Internet.
- **Map** (In parallel for each website) Count the words on a website, creating a dictionary of word counts.
- **Group** (In parallel for each website) For each word in the dictionary, send that word count to a process which handles only that word.
- **Reduce** (In parallel for each word) Sum all the counts for that word.

Google noticed many of their data gathering tasks followed this format, and therefore they could reduce the task to a simple function:
```
map_reduce(map_function, reduce_function)
```
Hadoop is based on MapReduce applied to databases.

## Types and Functional Programming

### FP Principle 4: Use immutable types

- **Immutable Types** can't be changed after they are created.  These include
  - `int`, `float`, `string`, `tuple`, `bool`, `frozenset`, `bytes`, `complex`
- **Mutable Types** can be changed.  These include
  - `list`, `dict`, `set`, `np.array`, `pd.DataFrame`, and almost all other types

```python
def try_to_change_input(x):
    x += x  # will this change x?
    return x

# immutable
a = 1
try_to_change_input(a)   # == 2
a                        # == 1

b = "hello"
try_to_change_input(b)   # == "hellohello"
b                        # == "hello"

c = (1, 2, 3)
try_to_change_input(c)   # == (1, 2, 3, 1, 2, 3)
c                        # == (1, 2, 3)

# mutable
d = [1, 2, 3]
try_to_change_input(d)   # == [1, 2, 3, 1, 2, 3]
c                        # == [1, 2, 3, 1, 2, 3]   changed!

e = np.array([1, 2, 3])
try_to_change_input(e)   # == np.array([2, 4, 6])
e                        # == np.array([2, 4, 6])  changed!
```

_**Python pro tip:**_
```python
# tuples of various lengths
(1, 2, 3)
(1, 2)
(1,)  # note the trailing comma; in Python: (1) == 1
()    # could also use tuple()
```
Functional programmers are really good at building immutable equivalents of mutable objects.

```python
# An immutable linked list (built with 2-tuples)
my_list = (1, (2, (3, (4, (5, (6, None))

# even though it is immutable I can "change" the first element
my_list2 = (-1, my_list[1])  # == (-1, (2, (3, (4, (5, (6, None))

# but this doesn't change the original list
my_list                      # == (1, (2, (3, (4, (5, (6, None))

# an immutable tree (built with 3-tuples)
#     a
#    / \
#   b   e
#  / \   \ 
# c   d   f

t = ("a",                       # first element is the value
        ("b",                   # second element is the left child
            ("c", None, None), 
            ("d", None, None)), 
        ("e",                   # third element is the right child
            None, 
            ("f", None, None))
```

_**Python pro tip:**
Use [`collections.NamedTuple`](https://docs.python.org/3/library/collections.html#collections.namedtuple) to build immutable types with attributes._

_**Python hack:**
`np.arrays` are not immutable, but they can hacked to be immutable by setting a flag._
```python
a = np.array([1, 2, 3])
a.setflags(write=False)  # makes immutable
a += 1                   # this will raise an error
```

### Immutable types and pure functions
Immutable types help ensure that functions stay pure (no side effects).  This is really important for parallel programming where the most common bugs are when two different processes try to change the same (non-thread-safe) object.

### Immutable types and dictionary keys
Immutable types can be used as dictionary keys.  The keys of a `dict`ionary (and the items in a `set`) must be immutable.

More formally, keys need to be **hashable**, but in Python the hashable objects are exactly the immutable ones:
- `int`, `float`, `complex`, `bool`, `string`, `bytes`, 
- `tuple` (if the items of the tuple are hashable!)
- `frozenset`

_**Python pro tip:**_
```python
d = {}

# tuples make good dictionary keys
d[1, 2] = 3  # key = (1,2) , value = 3
(1, 2) in d  # == True

# if you really need to use an immutable object as a key:

# convert lists to tuples
d[tuple(my_list)] = 3 

# convert sets and dictionaries to frozensets
d[frozenset(my_set)] = 3
d[frozenset(my_dict.items())] = 3

# convert numpy arrays to bytes
d[my_array.tobytes())] = 3
```

### Memoization (Caching)
Pure functions can be memoized/cached.  This is a significant part of **dynamic programming.**
```python
cache = {} # dict storing the results
def fib(n):
    """
    Computes nth Fibonacci number 
    0, 1, 1, 2, 3, 5, 8, ...
    """
    # check cache:
    if n in cache: 
        return cache[n]
    
    # otherwise calculate result (and cache it)
    elif n == 0: 
        result = cache[n] = 0
    elif n == 1: 
        result = cache[n] = 1
    else: 
        result = cache[n] = fib(n - 1) + fib(n - 2)
    
    return result
```

_**Python pro tip:**
Use [`@functools.lru_cache`](https://docs.python.org/3/library/functools.html#functools.lru_cache) to memoize automatically._
```python
from functools import lru_cache

@lru_cache  # automatically memoizes this function
def fib(n):
    """
    Computes nth Fibonacci number 
    0, 1, 1, 2, 3, 5, 8, ...
    """
    if n == 0: 
        return 0
    elif n == 1: 
        return 1
    else: 
        return fib(n - 1) + fib(n - 2)
```

### FP Principle 5: Consider type safety
- **Strongly typed language.** Every object has a fixed type.  Python is strongly typed, as are C++, Java, and most functional languages.
    ```python
    3                  # type: int
    "abc"              # type: string
    np.array([1,2,3])  # type: np.array
    ```
- **Dynamically typed (a.k.a. duck typed) language.** Variables and function arguments don't have a specified type.  It is sometimes even possible to add attributes to objects.  Python is dynamically typed.  This makes it great for a scripting language, but can cause errors.
    ```python
    def f(x): return x + x   # no type specification of x
    
    f(1)                # == 2
    f("abc")            # == abcabc
    f([1,2])            # == [1,2,1,2]
    f(np.array([1,2]))  # == np.array([2,4])
    f(None)             # TypeError!  (+ doesn't work on None)
    ```
- **Statically typed language.** Every variable and every function has a specified type.  C++, Java, and most functional languages are statically typed.

   _**Python pro tip:** Python has type hints, so it can pretend to be statically typed._
   ```python
   from typing import List

   def my_function(l: List[int], i: int) -> int:  
      # input types ─────┴───────────┘       │
      # return type ─────────────────────────┘    
      return l[i]
   
   # This is only a suggestion.  The compiler ignores type hints!
   my_function("abc", 2)   # == "c"
   ```
    Many newer functional languages have **type inference** where the compiler automatically infers the types for you so you don't have to always put them in the code.
   
- **Type safe language.**  This roughly means you can't do inappropriate things with types. A common type safety issue is Null values:
    ```python
    def divide(i: int, j: int) -> int:
        if j == 0: 
            return None   # to signify dividing by zero
       else: 
         return i // j
  
  divide(1,0)   # == None (which is not an int)
                # Is this what I want?  Does the use of divide
                # in my code take this into account?
    ```
    Some functional languages have advanced ways to:
    - avoid null types (which we have to remember to handle)
    - avoid raising errors (which we have to remember to handle)
    - make sure one handles all possible cases (even rare ones)
    - ensure that objects are only used in the way they were intended

### FP Principle 6: Embrace type theory
**Holy Trinity of functional programming theory**
- Type theory
- Symbolic logic
- Category theory

Don't believe anyone who says you have to learn category theory to program in a functional language.  It is enough to understand basic type theory.

#### Product type
If `A` and `B` are types, then so is the product `A x B`.  An element of this type is a pair `(a, b)`.  This is basically the same thing as a tuple in Python.
```python
a      # type: A
b      # type: B
(a, b) # type: A x B
```

#### Function type
If `A` and `B` are types then `A -> B` is the type of functions with inputs from `A` and outputs from `B`.
```python
def add(a: int, b: int) -> int:  # type: (int x int) -> int
    return a + b

def add_curried(a: int):         # type: int -> (int -> int)
    return lambda b: a + b

add(2, 3)            # 5           type: int

add_curried(2)       # function    type: int -> int
add_curried(2)(3)    # 5           type: int
```
This is an example of **currying**.  In most purely functional languages, functions default to curried form.  This makes it easy to partially apply the function.

```python 
plus_one = add_curried(1)  # function (type: int -> int)
```

In a functional language like OCaml, one doesn't have to write parentheses:
```ocaml
add;;                   (* Type: int -> int -> int       *)
add 1;;                 (* Type: int -> int              *)  
add 1 3;;               (* Type: int                     *)

add mult 1 2 add 3 4;;  (* Type: int                     *)
```

#### Sum type
If `A` and `B` are types then `A | B` is the type of objects `x`, where `x` is of type either  `A` or `B`.  There is not (a natural) way to do this in Python.  Sum types are great because they allow more than one type as input but still have strict control over the types of input.

OCaml, for example, has a generalization of sum types:
```ocaml
type PrimaryColor =  (* A type with only three values *)
| cyan
| yellow
| magenta;;

type Option =  
| None              (* None value represents an error *)
| Some of int;;     (* All other values are of the form Some x *)

cyan;;          (* Type: PrimaryColor  *)
None;;          (* Type: Option        *)
Some 5;;        (* Type: Option        *)
```
This allows pattern matching.
```ocaml
let other_colors c =
  match c with
  | cyan -> (yellow, magenta)
  | yellow -> (magenta, cyan)
  | magenta -> (cyan, yellow);;

let option_add x y =
  match x with
  | None -> None
  | Some x_ -> Some (x_ + y);;

other_colors;;            (* Type: PrimaryColor -> PrimaryColor x PrimaryColor *)

option_add;;              (*           Type: Option -> int -> Option  *)             
option_add (Some 5) 4;;   (* Some 9    Type: Option                   *)
option_add None 4;;       (* None      Type: Option                   *)
```

#### Recursive types
Recursive types are self referential types.  They naturally go with recursive functions.
```ocaml
type linked_list =
  | Nil                    (* The end of the list *)
  | Cons of linked_list;;  (* A node in the linked list *)
                           (* "Cons" is short for constructor *)
type tree =
  | Leaf                        (* Dead end *)
  | Node of int * tree * tree;; (* Node containing: 
                                   - node value
                                   - left subtree
                                   - right subtree *)
                                   
(* recursive function which sums all the values in a tree *)
let sum_values t =
  match t with
  | Leaf -> 0
  | Node v t1 t2 -> v + (sum_values t1) + (sum_values t2);;
```

# The end
