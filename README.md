# Support Vector Machines

**Files:**
- [**Incomplete**] From Scratch
- Using scikit-learn, on [breast cancer data](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)

**Notes**
- There are two branches in the repo:
    1. master
    2. model
- On `master`:
```python
# With test_size=0.2
type(X)         # object of shape (699, 9)
type(X_test)    # object of shape (140, 9)
type(X_train)   # object of shape (559, 9)

type(y)         # int64 of shape (699,)
type(y_test)    # int64 of shape (140,)
type(y_train)   # int64 of shape (559,)
```
- On `model`:
    - `df.replace('?', -99999, inplace=True)` probably not a good idea. Better to replace with `0`
    

**Implementation tasks/ideas**

1. Generate graphs of acc, recall, sensitivity for each classification
2. Find the difference (in terms of graphs) between scores in each classification and each iteration