#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# minimal requirements
# python >= 3.6
# pandas >= 1

import numpy as np
import pandas as pd
from typing import Dict
from functools import lru_cache 
from operator import itemgetter
from collections import Counter


# You have
# * a ribbon of length 'LENGTH' meters (the problem is 1D).
# * defects that lies on the ribbon. Each defect has 
#     * a position `x`
#     * and a class [`a`, `b`, `c`, ...]
# * set of available products. Each product can be produced an infinite number of times, and has:
#     * a size (along the same dimension as the ribbon)
#     * a value (see next paragraph)
#     * and a threshold for the maximum number of defects of each size it can contains
# 
# A solution is an affectation of products on the ribbon. An affectation, to be valid, need to be:
# * at __integer positions__
# * have no intersection between its products placement. If you place a product P1 of size 3, at position x=2, you cant have any product affected at positions `x=3` nor `x=4`.
# * each product placed on the ribbon needs to contain less (or equal) defects of each class from the ribbon than authorized by its thresholds. A product P1 of size 3 placed at `x=2`, contains all defects of the ribbon which `x` is in `[2, 5]`. If in this interval, we have 3 defects of class `a`, and that threshold of P1 authorized maximum 2 defects of class `a`, the affectation is invalid
# 
# The value of the solution is the sum of the value of the individual products placed on it. Part of the ribbon with no product affected amount to 0
# 
# 
# Benchmark:
# * this notebook generates random instances.
# * if you run this cells in this order after the seeding (and without calling other randoms/re-executing cells), you can find a solution of value 358.

# # Setup
# ## we define some fixed parameters and generate some random instances

# In[ ]:


# seed random for reproductibility of benchmark instance
np.random.seed(5)


# In[ ]:


# length of the glass ribbon to cut
LENGTH = 500


# ## defects

# In[ ]:


n_defects = 500

# classe 1, classe 2 etc
n_defect_classes = 4
assert n_defect_classes <= 26, "too much defects classes"

# generates defects position
defects_x = np.random.uniform(0, LENGTH, (n_defects))

# generates their classes
defects_class = np.random.choice(
    list('abcdefghijklmnopqrstuvwxyz'[:n_defect_classes]),
    (n_defects)
)

# summarize
defects = pd.DataFrame(
    columns=['x', 'class'],
    data = np.array([defects_x, defects_class]).T
)
defects['x'] = defects['x'].astype(float)
defects.head(3).style.set_caption('extract of defects of the ribbon').format('{:,.2f}', subset='x')


# ## products

# In[ ]:


n_products = 4

class Product:

    def __init__(self):
        self.length: int = np.random.randint(4,10)
        self.value: int = np.random.randint(1,10)
        self.max_defects: Dict[int, int] = {
            key: np.random.randint(1,5) for key in defects_class
        }

    def __repr__(self):
        return f'Product of size {self.length}, value {self.value} and max_defects {self.max_defects}'


# generate n_products random product (ie random size, value and threshold)
Products = [Product() for _ in range(n_products)]
print('\n * '.join([f'The {n_products} products are'] + [str(p) for p in Products]))


# ## Solution structure
# 
# The Solution class needs to be instantiated with the current "defects" you are using.
# ```python
# current_solution = Solution(defects)
# ```
# 
# You can create the Solution iteratively, by placing product `p` at position `position`, with the method 
# ```python
# current_solution.add_product(p, position)
# ```
# 
# You can compute your current solution score with
# ```python
# current_solution.compute_value()
# ```
# 
# You can check if you dont have any invalidities with
# ```python
# current_solution.checker()
# ```

# In[ ]:


class PlacedProduct:
    """
    helper class representing a product placed on a position of the ribbon
    """
    def __init__(self, product: Product, position: int):
        self.product = product
        self.position = position

class Solution:
    def __init__(self, defects:pd.DataFrame):
        self.placedProducts = []
        self.defects = defects

    def add_product(self, product: Product, position: int=0)->None:
        self.placedProducts.append(PlacedProduct(product, position))

    def compute_value(self):
        return sum((pp.product.value for pp in self.placedProducts))

    def checker(self):
        bob = pd.DataFrame(
            np.array([[pp.position for pp in self.placedProducts], [pp.product.length for pp in self.placedProducts]]).T,
            columns=['pos', 'length']
        )
        bob = bob.sort_values('pos')

        a0 = bob[np.floor(bob['pos']) < bob['pos']]
        assert len(a0) == 0, f'placedProducts at non integer positions {*a0.to_list(), }'

        a1 = bob.loc[bob.sum(axis=1)>LENGTH, 'pos']
        assert len(a1) == 0, f'placedProducts exceed LENGTH {*a1.to_list(), }'

        a2 = bob.sort_values('pos')
        a2 = bob.loc[bob['pos'] + bob['length'] > bob['pos'].shift(-1), 'pos']
        assert len(a2)==0, f'overlapping placedProducts at positions {*a2.to_list(), }'


        # check max defects OK
        for pp in self.placedProducts:
            defects_in_plate = self.defects.loc[(self.defects['x'] >= pp.position) & (self.defects['x'] <= pp.position + pp.product.length), "class"].to_list()
            a3 = Counter(defects_in_plate)-Counter(pp.product.max_defects)
            assert not a3, f"plate at position {pp.position} contains too much defects of classes {*a3.keys(), }"


        print("solution valid")


# In[ ]:


## demo OK with initial seeding
sol = Solution(defects)
sol.add_product(
    product=Products[0],
    position=10
)
print('value', sol.compute_value())
sol.checker()


# In[ ]:


## demo not OK cause overlap with previous plate
sol = Solution(defects)
sol.add_product(
    product=Products[3],
    position=0
)

sol.add_product(
    product=Products[3],
    position=2
)
print('value', sol.compute_value())
sol.checker()


# In[ ]:


## demo not OK because too much defects
sol = Solution(defects)
sol.add_product(
    product=Products[2],
    position=5
)
print('value', sol.compute_value())
sol.checker()


# ## HELPERS
# 
# Some functions that can be used build a solution

# In[ ]:


from functools import lru_cache 
from operator import itemgetter
from collections import Counter


# In[ ]:


# helpers
def defects_counts(x: float, length: float) -> Counter:
    """
    Count number of defects of each class on the ribbon [x, x+length]
    return {defect class: number of defects}
    """
    # filter index of defects within range
    res = defects.loc[(defects["x"] >= x) & (defects["x"] <= x+length), "class"].to_list()

    return Counter(res)
    
# if we start at x, a plate of length 7 will have quality: number         
defects_counts(20, 7)


# In[ ]:


def contains(container: Counter, content: Counter) -> bool:
    """
    check if all values of container are bigger than those of content.
    Can compare <defects_counts> to products thresholds
    """
    return not content - container

print(contains(Counter([1,1]), Counter([1])))
print(contains(Counter([1,1]), Counter([1,2])))


# In[ ]:


def possible(x: float, p: Product) -> bool:
    """
    return True product p is compatible with position x
    """
    defects_present = defects_counts(x, p.length)
    return contains(Counter(p.max_defects), defects_present)


possible(1, Products[0])


# In[ ]:


# example of solution creation from a list res = [(position first element, product first element), (position second element, product second element), ...]
res = [(,), (,), ...]
sol = Solution(defects)
for item in res:
    sol.add_product(item[1], item[0])
print(sol.compute_value())
sol.checker()


# In[ ]:




