# NumPy: A Technical Overview

## Introduction

NumPy, short for Numerical Python, is a fundamental library in Python for numerical computing, widely used in scientific computing, data analysis, and machine learning applications. Its primary data structure, `numpy.ndarray`, is an n-dimensional array that efficiently stores and manipulates numerical data. In this technical overview, we'll delve into the key features and capabilities of NumPy that make it such a powerful tool for numerical computation.

## Key Features

### Arrays and Memory Management

NumPy's `ndarray` is the cornerstone of its functionality. Unlike Python lists, `ndarray` is a contiguous block of memory containing elements of the same data type, allowing for efficient storage and manipulation of large datasets. This layout facilitates fast element-wise operations and eliminates the overhead associated with dynamic typing.

### Vectorized Operations and Performance

NumPy's vectorized operations enable efficient element-wise operations on arrays, leveraging optimized, low-level implementations for improved performance. By avoiding explicit iteration over array elements, vectorized operations take advantage of hardware-level optimizations and parallelism, resulting in significant speedups compared to traditional Python loops.

### Broadcasting

Broadcasting is a powerful mechanism in NumPy that allows arrays with different shapes to be combined in arithmetic operations. NumPy automatically aligns dimensions to perform operations, simplifying code and improving readability. This feature is particularly useful when working with arrays of different shapes or when performing operations between arrays and scalars.

### Mathematical Functions and Numerical Computations

NumPy provides a comprehensive suite of mathematical functions for numerical computations. These functions cover a wide range of mathematical operations, including basic arithmetic, trigonometry, logarithms, exponentials, and more. NumPy's ability to apply these functions element-wise to entire arrays enables efficient numerical computations on large datasets.

### Random Number Generation

The `numpy.random` module offers robust tools for generating random numbers from various probability distributions. Random number generation is essential for tasks such as model initialization, data simulation, and statistical analysis. NumPy's random number generation capabilities provide researchers and practitioners with tools to simulate stochastic processes and analyze probabilistic outcomes effectively.

### Integration with Low-Level Languages and Existing Libraries

NumPy seamlessly integrates with code written in low-level languages such as C and Fortran, allowing Python developers to leverage existing optimized numerical routines. This integration enhances computational efficiency and enables access to a vast ecosystem of pre-existing algorithms and numerical methods. NumPy's interoperability with other libraries, such as SciPy, Matplotlib, and scikit-learn, further extends its capabilities for scientific computing and machine learning applications.


## Importing NumPy
Once you've installed NumPy you can import it as a library:
```python
import numpy as np
import numpy as np
```
## NumPy Arrays
 NumPy arrays essentially come in two flavors: vectors and matrices. Vectors are strictly 1-dimensional (1D) arrays and matrices are 2D (but you should note a matrix can still have only one row or one column).

### Why use Numpy array? Why not just a list?
There are lot's of reasons to use a Numpy array instead of a "standard" python list object. 
Our main reasons are:

> Memory Efficiency of Numpy Array vs list
> Easily expands to N-dimensional objects
> Speed of calculations of numpy array
> Broadcasting operations and functions with numpy
> All the data science and machine learning libraries we use are built with Numpy

### Simple Example of what numpy array can do
```python
my_list = [1,2,3]
my_array = np.array([1,2,3])
type(my_list)
```
Output list
​
Let's begin our introduction by exploring how to create NumPy arrays.

### Creating NumPy Arrays from Objects
From a Python List
We can create an array by directly converting a list or list of lists:
```python
my_list = [1,2,3]
my_list
```
Output [1, 2, 3]
```python
np.array(my_list)
```
Output array([1, 2, 3])
```python
my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
my_matrix
```
Output [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
np.array(my_matrix)
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
### Built-in Methods to create arrays
There are lots of built-in ways to generate arrays.

#### arange
Return evenly spaced values within a given interval. [reference]
```python
np.arange(0,10)
```
Output array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```python
np.arange(0,11,2)
```
Output array([ 0,  2,  4,  6,  8, 10])
#### zeros and ones
Generate arrays of zeros or ones. [reference]
```python
np.zeros(3)
```
Output array([0., 0., 0.])
```python
np.zeros((5,5))
```
output
array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])
```python
np.ones(3)
```
output array([1., 1., 1.])
```python
np.ones((3,3))
```
output 
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
#### linspace
Return evenly spaced numbers over a specified interval. [reference]
```python
np.linspace(0,10,3)
```
Output array([ 0.,  5., 10.])
``` python
np.linspace(0,5,20)
```
Output
array([0.        , 0.26315789, 0.52631579, 0.78947368, 1.05263158,
       1.31578947, 1.57894737, 1.84210526, 2.10526316, 2.36842105,
       2.63157895, 2.89473684, 3.15789474, 3.42105263, 3.68421053,
       3.94736842, 4.21052632, 4.47368421, 4.73684211, 5.        ])
Note that .linspace() includes the stop value. To obtain an array of common fractions, increase the number of items:
```python
np.linspace(0,5,21)
```
Output 
array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  , 2.25, 2.5 ,
       2.75, 3.  , 3.25, 3.5 , 3.75, 4.  , 4.25, 4.5 , 4.75, 5.  ])
#### eye
Creates an identity matrix [reference]
```python
np.eye(4)
```
Output
array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])
#### Random
Numpy also has lots of ways to create random number arrays:

##### rand
Creates an array of the given shape and populates it with random samples from a uniform distribution over [0, 1). [reference]
```python
np.random.rand(2)
```
Output array([0.37065108, 0.89813878])
``` python
np.random.rand(5,5)
```
Output
array([[0.03932992, 0.80719137, 0.50145497, 0.68816102, 0.1216304 ],
       [0.44966851, 0.92572848, 0.70802042, 0.10461719, 0.53768331],
       [0.12201904, 0.5940684 , 0.89979774, 0.3424078 , 0.77421593],
       [0.53191409, 0.0112285 , 0.3989947 , 0.8946967 , 0.2497392 ],
       [0.5814085 , 0.37563686, 0.15266028, 0.42948309, 0.26434141]])
##### randn
Returns a sample (or samples) from the "standard normal" distribution [σ = 1]. Unlike rand which is uniform, values closer to zero are more likely to appear. [reference]
```python
np.random.randn(2)
```
Output array([-0.36633217, -1.40298731])
``` python
np.random.randn(5,5)
```
Output
array([[-0.45241033,  1.07491082,  1.95698188,  0.40660223, -1.50445807],
       [ 0.31434506, -2.16912609, -0.51237235,  0.78663583, -0.61824678],
       [-0.17569928, -2.39139828,  0.30905559,  0.1616695 ,  0.33783857],
       [-0.2206597 , -0.05768918,  0.74882883, -1.01241629, -1.81729966],
       [-0.74891671,  0.88934796,  1.32275912, -0.71605188,  0.0450718 ]])
##### randint
Returns random integers from low (inclusive) to high (exclusive). [reference]
```python
np.random.randint(1,100)
```
Output 61
``` python
np.random.randint(1,100,10)
```
Output array([39, 50, 72, 18, 27, 59, 15, 97, 11, 14])
##### seed
Can be used to set the random state, so that the same "random" results can be reproduced. [reference]
``` python
np.random.seed(42)
np.random.rand(4)
```
Output array([0.37454012, 0.95071431, 0.73199394, 0.59865848])
``` python
np.random.seed(42)
np.random.rand(4)
```
Output array([0.37454012, 0.95071431, 0.73199394, 0.59865848])

### Array Attributes and Methods
Let's discuss some useful attributes and methods for an array:
```python
arr = np.arange(25)
ranarr = np.random.randint(0,50,10)
arr
```
Output
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24])
ranarr
array([38, 18, 22, 10, 10, 23, 35, 39, 23,  2])

#### Reshape
Returns an array containing the same data with a new shape. [reference]
``` python
arr.reshape(5,5)
```
Output 
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
### max, min, argmax, argmin
These are useful methods for finding max or min values. Or to find their index locations using argmin or argmax

#### ranarr
``` python
array([38, 18, 22, 10, 10, 23, 35, 39, 23,  2])
ranarr.max()
```
39
``` python
ranarr.argmax()
```
7
``` python
ranarr.min()
```
2
``` python
ranarr.argmin()
```
9
#### Shape
Shape is an attribute that arrays have (not a method): [reference]
``` python
# Vector
arr.shape
```
(25,)
Notice the two sets of brackets
``` python
arr.reshape(1,25)
```
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24]])
 ```python
arr.reshape(1,25).shape
```
(1, 25)
``` python
arr.reshape(25,1)
```
array([[ 0],
       [ 1],
       [ 2],
       [ 3],
       [ 4],
       [ 5],
       [ 6],
       [ 7],
       [ 8],
       [ 9],
       [10],
       [11],
       [12],
       [13],
       [14],
       [15],
       [16],
       [17],
       [18],
       [19],
       [20],
       [21],
       [22],
       [23],
       [24]])
 ```python
arr.reshape(25,1).shape
```
(25, 1)
#### dtype
You can also grab the data type of the object in the array: [reference]
``` python
arr.dtype
```
dtype('int32')
```python
arr2 = np.array([1.2, 3.4, 5.6])
arr2.dtype
```
dtype('float64')
​
## NumPy Indexing and Selection
```python
import numpy as np
#Creating sample array
arr = np.arange(0,11)
#Show
arr
```
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
### Bracket Indexing and Selection
The simplest way to pick one or some elements of an array looks very similar to python lists:
```python
#Get a value at an index
arr[8]
```
8
```python
#Get values in a range
arr[1:5]
```
array([1, 2, 3, 4])
```python
#Get values in a range
arr[0:5]
```
array([0, 1, 2, 3, 4])
### Broadcasting
NumPy arrays differ from normal Python lists because of their ability to broadcast. With lists, you can only reassign parts of a list with new parts of the same size and shape. That is, if you wanted to replace the first 5 elements in a list with a new value, you would have to pass in a new 5 element list. With NumPy arrays, you can broadcast a single value across a larger set of values:
```python
#Setting a value with index range (Broadcasting)
arr[0:5]=100​
#Show
arr
```
array([100, 100, 100, 100, 100,   5,   6,   7,   8,   9,  10])
```python
#Reset array, we'll see why I had to reset in  a moment
arr = np.arange(0,11)
​#Show
arr
```
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```python
#Important notes on Slices
slice_of_arr = arr[0:6]
​#Show slice
slice_of_arr
```
array([0, 1, 2, 3, 4, 5])
```python
#Change Slice
slice_of_arr[:]=99
​#Show Slice again
slice_of_arr
```
array([99, 99, 99, 99, 99, 99])
Now note the changes also occur in our original array!
```python
arr
```
array([99, 99, 99, 99, 99, 99,  6,  7,  8,  9, 10])

Data is not copied, it's a view of the original array! This avoids memory problems!
#To get a copy, need to be explicit
```python
arr_copy = arr.copy()
arr_copy
```
array([99, 99, 99, 99, 99, 99,  6,  7,  8,  9, 10])

### Indexing a 2D array (matrices)
The general format is arr_2d[row][col] or arr_2d[row,col]. I recommend using the comma notation for clarity.
``` python
arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))
​#Show
arr_2d
```
array([[ 5, 10, 15],
       [20, 25, 30],
       [35, 40, 45]])
``` python
#Indexing row
arr_2d[1]
```
array([20, 25, 30])
``` python
#Format is arr_2d[row][col] or arr_2d[row,col]
​#Getting individual element value
arr_2d[1][0]
```
20
``` python
#Getting individual element value
arr_2d[1,0]
```
20
``` python
#2D array slicing
​#Shape (2,2) from top right corner
arr_2d[:2,+]
```
array([[10, 15],
       [25, 30]])
``` python
#Shape bottom row
arr_2d[2]
```
array([35, 40, 45])
``` python
#Shape bottom row
arr_2d[2,:]
```
array([35, 40, 45])

### More Indexing Help
Indexing a 2D matrix can be a bit confusing at first, especially when you start to add in step size. Try google image searching NumPy indexing to find useful images, like this one:

Image Image source: http://www.scipy-lectures.org/intro/numpy/numpy.html

### Conditional Selection
This is a very fundamental concept that will directly translate to pandas later on, make sure you understand this part!

Let's briefly go over how to use brackets for selection based off of comparison operators.
``` python
arr = np.arange(1,11)
arr
```
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
``` python
arr > 4
```
array([False, False, False, False,  True,  True,  True,  True,  True,
        True])
``` python
bool_arr = arr>4
bool_arr
```
array([False, False, False, False,  True,  True,  True,  True,  True,
        True])
``` python
arr[bool_arr]
```
array([ 5,  6,  7,  8,  9, 10])
``` python
arr[arr>2]
```
array([ 3,  4,  5,  6,  7,  8,  9, 10])
``` python
x = 2
arr[arr>x]
```
array([ 3,  4,  5,  6,  7,  8,  9, 10])

## NumPy Operations
### Arithmetic
You can easily perform array with array arithmetic, or scalar with array arithmetic. Let's see some examples:
``` python
import numpy as np
arr = np.arange(0,10)
arr
```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```python
arr + arr
```
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])
``` python
arr * arr
```
array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])
``` python
arr - arr
```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
``` python
#This will raise a Warning on division by zero, but not an error!
#It just fills the spot with nan
arr/arr
```
``` python
C:\Anaconda3\envs\tsa_course\lib\site-packages\ipykernel_launcher.py:3: RuntimeWarning: invalid value encountered in true_divide
  This is separate from the ipykernel package so we can avoid doing imports until
```
array([nan,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
``` python
# Also a warning (but not an error) relating to infinity
1/arr
C:\Anaconda3\envs\tsa_course\lib\site-packages\ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in true_divide
  ```
array([       inf, 1.        , 0.5       , 0.33333333, 0.25      ,
       0.2       , 0.16666667, 0.14285714, 0.125     , 0.11111111])
 ```python
arr**3
```
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729], dtype=int32)
### Universal Array Functions
NumPy comes with many universal array functions, or ufuncs, which are essentially just mathematical operations that can be applied across the array.
Let's show some common ones:
``` python
#Taking Square Roots
np.sqrt(arr)
```
array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,
       2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ])
       ``` python
#Calculating exponential (e^)
np.exp(arr)
```
array([1.00000000e+00, 2.71828183e+00, 7.38905610e+00, 2.00855369e+01,
       5.45981500e+01, 1.48413159e+02, 4.03428793e+02, 1.09663316e+03,
       2.98095799e+03, 8.10308393e+03])
``` python
#Trigonometric Functions like sine
np.sin(arr)
```
array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ,
       -0.95892427, -0.2794155 ,  0.6569866 ,  0.98935825,  0.41211849])
       ``` python
#Taking the Natural Logarithm
np.log(arr)
```
C:\Anaconda3\envs\tsa_course\lib\site-packages\ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in log
  
array([      -inf, 0.        , 0.69314718, 1.09861229, 1.38629436,
       1.60943791, 1.79175947, 1.94591015, 2.07944154, 2.19722458])

### Summary Statistics on Arrays
NumPy also offers common summary statistics like sum, mean and max. You would call these as methods on an array.
``` python
arr = np.arange(0,10)
arr
```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
``` python
arr.sum()
```
45
``` python
arr.mean()
```
4.5
``` python
arr.max()
```
9
#### Other summary statistics include:

arr.min() returns 0                   minimum
arr.var() returns 8.25                variance
arr.std() returns 2.8722813232690143  standard deviation
### Axis Logic
When working with 2-dimensional arrays (matrices) we have to consider rows and columns. This becomes very important when we get to the section on pandas. In array terms, axis 0 (zero) is the vertical axis (rows), and axis 1 is the horizonal axis (columns). These values (0,1) correspond to the order in which arr.shape values are returned.

Let's see how this affects our summary statistic calculations from above.
``` python
arr_2d = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
arr_2d
```
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
       ``` python
arr_2d.sum(axis=0)
```
array([15, 18, 21, 24])
By passing in axis=0, we're returning an array of sums along the vertical axis, essentially [(1+5+9), (2+6+10), (3+7+11), (4+8+12)]

### Image
``` python
arr_2d.shape


#  2.matplotlib.py LIBRRARY
# Matplotlib: A Comprehensive Data Visualization Library

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It is widely used for producing high-quality plots, charts, and figures for various scientific, engineering, and data analysis applications. Matplotlib provides a MATLAB-like interface and is highly customizable, allowing users to create a wide range of visualizations with ease.

## Key Features and Components of Matplotlib:

### Pyplot Interface:
The `matplotlib.pyplot` module provides a MATLAB-like interface for creating plots. It simplifies the process of creating common types of plots such as line plots, scatter plots, histograms, bar charts, etc.

### Object-Oriented Interface:
Matplotlib also offers an object-oriented interface, which gives more control and flexibility over the plots. Users can directly work with figure and axis objects to create and customize plots in a more granular manner.

### Supported Plot Types:
Matplotlib supports a wide range of plot types, including:
- Line plots
- Scatter plots
- Histograms
- Bar charts
- Pie charts
- Box plots
- Contour plots
- Heatmaps
- 3D plots (with mplot3d toolkit)

### Customization:
Matplotlib allows extensive customization of plots. Users can control aspects such as colors, line styles, markers, labels, titles, axis limits, grid lines, legends, annotations, and more.

### Multiple Output Formats:
Matplotlib can save plots in various formats such as PNG, PDF, SVG, and EPS, making it easy to integrate plots into documents, reports, or presentations.

### Integration with NumPy:
Matplotlib seamlessly integrates with NumPy, a fundamental library for numerical computing in Python. This integration allows users to plot NumPy arrays directly and perform mathematical operations on data before plotting.

### Support for LaTeX:
Matplotlib supports LaTeX for mathematical expressions in labels, titles, annotations, and other text elements, enabling high-quality typesetting of mathematical symbols and equations.

### Interactive Features:
Matplotlib can be used in interactive environments such as Jupyter notebooks, allowing users to dynamically explore and manipulate plots.

### Extensibility:
Matplotlib is highly extensible and customizable. Users can create custom plot types, styles, and functionalities by extending Matplotlib's functionality or by utilizing additional toolkits and libraries built on top of Matplotlib, such as Seaborn, pandas plotting, and mpld3 for interactive web-based visualizations.

Matplotlib is a powerful and versatile library for data visualization in Python, making it a popular choice among data scientists, researchers, engineers, and analysts for exploring and communicating data insights.

 # Matplotlib: A Comprehensive Data Visualization Library
## Introduction
Matplotlib is the "grandfather" library of data visualization with Python. It was created by John Hunter. He created it to try to replicate MatLab's (another programming language) plotting capabilities in Python. 
Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It is widely used for producing high-quality plots, charts, and figures for various scientific, engineering, and data analysis applications. Matplotlib provides a MATLAB-like interface and is highly customizable, allowing users to create a wide range of visualizations with ease.

It is an excellent 2D and 3D graphics library for generating scientific figures.

Some of the major Pros of Matplotlib are:

-Generally easy to get started for simple plots
-Support for custom labels and texts
-Great control of every element in a figure
-High-quality output in many formats
-Very customizable in general
-Matplotlib allows you to create reproducible figures programmatically.

## Key Features and Components of Matplotlib:

### Pyplot Interface:
The `matplotlib.pyplot` module provides a MATLAB-like interface for creating plots. It simplifies the process of creating common types of plots such as line plots, scatter plots, histograms, bar charts, etc.

### Object-Oriented Interface:
Matplotlib also offers an object-oriented interface, which gives more control and flexibility over the plots. Users can directly work with figure and axis objects to create and customize plots in a more granular manner.

### Supported Plot Types:
Matplotlib supports a wide range of plot types, including:
- Line plots
- Scatter plots
- Histograms
- Bar charts
- Pie charts
- Box plots
- Contour plots
- Heatmaps
- 3D plots (with mplot3d toolkit)

### Customization:
Matplotlib allows extensive customization of plots. Users can control aspects such as colors, line styles, markers, labels, titles, axis limits, grid lines, legends, annotations, and more.

### Multiple Output Formats:
Matplotlib can save plots in various formats such as PNG, PDF, SVG, and EPS, making it easy to integrate plots into documents, reports, or presentations.

### Integration with NumPy:
Matplotlib seamlessly integrates with NumPy, a fundamental library for numerical computing in Python. This integration allows users to plot NumPy arrays directly and perform mathematical operations on data before plotting.

### Support for LaTeX:
Matplotlib supports LaTeX for mathematical expressions in labels, titles, annotations, and other text elements, enabling high-quality typesetting of mathematical symbols and equations.

### Interactive Features:
Matplotlib can be used in interactive environments such as Jupyter notebooks, allowing users to dynamically explore and manipulate plots.

### Extensibility:
Matplotlib is highly extensible and customizable. Users can create custom plot types, styles, and functionalities by extending Matplotlib's functionality or by utilizing additional toolkits and libraries built on top of Matplotlib, such as Seaborn, pandas plotting, and mpld3 for interactive web-based visualizations.

## Installation
If you are using our environment, its already installed for you. If you are not using our environment (not recommended), you'll need to install matplotlib first with either:

conda install matplotlib
or

pip install matplotlib

## Importing
Import the matplotlib.pyplot module under the name plt (the tidy way):
```python
#COMMON MISTAKE!
#DON'T FORGET THE .PYPLOT part
​
import matplotlib.pyplot as plt
```
NOTE: If you are using an older version of jupyter, you need to run a "magic" command to see the plots inline with the notebook. Users of jupyter notebook 1.0 and above, don't need to run the cell below:
```python
%matplotlib inline
```
NOTE: For users running .py scripts in an IDE like PyCharm or Sublime Text Editor. You will not see the plots in a notebook, instead if you are using another editor, you'll use: plt.show() at the end of all your plotting commands to have the figure pop up in another window.

## Basic Example
Let's walk through a very simple example using two numpy arrays:

### Basic Array Plot
Let's walk through a very simple example using two numpy arrays. You can also use lists, but most likely you'll be passing numpy arrays or pandas columns (which essentially also behave like arrays).

#### The data we want to plot:
```python
import numpy as np
x = np.arange(0,10)
y = 2*x
x
```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```python
y
```
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])

## Using Matplotlib with plt.plot() function calls
### Basic Matplotlib Commands
We can create a very simple line plot using the following ( I encourage you to pause and use Shift+Tab along the way to check out the document strings for the functions we are using).
```python
plt.plot(x, y) 
plt.xlabel('X Axis Title Here')
plt.ylabel('Y Axis Title Here')
plt.title('String Title Here')
plt.show() # Required for non-jupyter users , but also removes Out[] info
```
![1](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/ad2190c7-9838-49be-a34d-6a834c709d13)

### Editing more figure parameters
```python
plt.plot(x, y) 
plt.xlabel('X Axis Title Here')
plt.ylabel('Y Axis Title Here')
plt.title('String Title Here')
plt.xlim(0,6) # Lower Limit, Upper Limit
plt.ylim(0,12) # Lower Limit, Upper Limit
plt.show() # Required for non-jupyter users , but also removes Out[] info
```
![2](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/21c2709a-1451-4a34-8595-eca03f58ff56)

## Exporting a plot
```python
help(plt.savefig)
```
Help on function savefig in module matplotlib.pyplot:

savefig(*args, **kwargs)
    Save the current figure.
    
    Call signature::
    
      savefig(fname, dpi=None, facecolor='w', edgecolor='w',
              orientation='portrait', papertype=None, format=None,
              transparent=False, bbox_inches=None, pad_inches=0.1,
              frameon=None, metadata=None)
    
    The output formats available depend on the backend being used.
    
    Parameters
    ----------
    
    fname : str or PathLike or file-like object
        A path, or a Python file-like object, or
        possibly some backend-dependent object such as
        `matplotlib.backends.backend_pdf.PdfPages`.
    
        If *format* is not set, then the output format is inferred from
        the extension of *fname*, if any, and from :rc:`savefig.format`
        otherwise.  If *format* is set, it determines the output format.
    
        Hence, if *fname* is not a path or has no extension, remember to
        specify *format* to ensure that the correct backend is used.
    
    Other Parameters
    ----------------
    
    dpi : [ *None* | scalar > 0 | 'figure' ]
        The resolution in dots per inch.  If *None*, defaults to
        :rc:`savefig.dpi`.  If 'figure', uses the figure's dpi value.
    
    quality : [ *None* | 1 <= scalar <= 100 ]
        The image quality, on a scale from 1 (worst) to 95 (best).
        Applicable only if *format* is jpg or jpeg, ignored otherwise.
        If *None*, defaults to :rc:`savefig.jpeg_quality` (95 by default).
        Values above 95 should be avoided; 100 completely disables the
        JPEG quantization stage.
    
    optimize : bool
        If *True*, indicates that the JPEG encoder should make an extra
        pass over the image in order to select optimal encoder settings.
        Applicable only if *format* is jpg or jpeg, ignored otherwise.
        Is *False* by default.
    
    progressive : bool
        If *True*, indicates that this image should be stored as a
        progressive JPEG file. Applicable only if *format* is jpg or
        jpeg, ignored otherwise. Is *False* by default.
    
    facecolor : color spec or None, optional
        The facecolor of the figure; if *None*, defaults to
        :rc:`savefig.facecolor`.
    
    edgecolor : color spec or None, optional
        The edgecolor of the figure; if *None*, defaults to
        :rc:`savefig.edgecolor`
    
    orientation : {'landscape', 'portrait'}
        Currently only supported by the postscript backend.
    
    papertype : str
        One of 'letter', 'legal', 'executive', 'ledger', 'a0' through
        'a10', 'b0' through 'b10'. Only supported for postscript
        output.
    
    format : str
        The file format, e.g. 'png', 'pdf', 'svg', ... The behavior when
        this is unset is documented under *fname*.
    
    transparent : bool
        If *True*, the axes patches will all be transparent; the
        figure patch will also be transparent unless facecolor
        and/or edgecolor are specified via kwargs.
        This is useful, for example, for displaying
        a plot on top of a colored background on a web page.  The
        transparency of these patches will be restored to their
        original values upon exit of this function.
    
    bbox_inches : str or `~matplotlib.transforms.Bbox`, optional
        Bbox in inches. Only the given portion of the figure is
        saved. If 'tight', try to figure out the tight bbox of
        the figure. If None, use savefig.bbox
    
    pad_inches : scalar, optional
        Amount of padding around the figure when bbox_inches is
        'tight'. If None, use savefig.pad_inches
    
    bbox_extra_artists : list of `~matplotlib.artist.Artist`, optional
        A list of extra artists that will be considered when the
        tight bbox is calculated.
    
    metadata : dict, optional
        Key/value pairs to store in the image metadata. The supported keys
        and defaults depend on the image format and backend:
    
        - 'png' with Agg backend: See the parameter ``metadata`` of
          `~.FigureCanvasAgg.print_png`.
        - 'pdf' with pdf backend: See the parameter ``metadata`` of
          `~.backend_pdf.PdfPages`.
        - 'eps' and 'ps' with PS backend: Only 'Creator' is supported.
    
    pil_kwargs : dict, optional
        Additional keyword arguments that are passed to `PIL.Image.save`
        when saving the figure.  Only applicable for formats that are saved
        using Pillow, i.e. JPEG, TIFF, and (if the keyword is set to a
        non-None value) PNG.
```python
plt.plot(x,y)
plt.savefig('example.png')
```

![3](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/ab3aac25-e70a-4a7c-b028-78b95330797a)

Import the matplotlib.pyplot module under the name plt (the tidy way):
```python
#COMMON MISTAKE!
#DON'T FORGET THE .PYPLOT part
​
import matplotlib.pyplot as plt
```
NOTE: For users running .py scripts in an IDE like PyCharm or Sublime Text Editor. You will not see the plots in a notebook, instead if you are using another editor, you'll use: plt.show() at the end of all your plotting commands to have the figure pop up in another window.

## Matplotlib Object Oriented Method
Now that we've seen the basics, let's break it all down with a more formal introduction of Matplotlib's Object Oriented API. This means we will instantiate figure objects and then call methods or attributes from that object.

### The Data
```python
import numpy as np
a = np.linspace(0,10,11)
b = a ** 4
a
```
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
```python
b
```
array([0.000e+00, 1.000e+00, 1.600e+01, 8.100e+01, 2.560e+02, 6.250e+02,
       1.296e+03, 2.401e+03, 4.096e+03, 6.561e+03, 1.000e+04])
```python
x = np.arange(0,10)
y = 2 * x
x
```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```python
y
```
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])
## Creating a Figure
The main idea in using the more formal Object Oriented method is to create figure objects and then just call methods or attributes off of that object. This approach is nicer when dealing with a canvas that has multiple plots on it.

#Creates blank canvas
fig = plt.figure()
<Figure size 432x288 with 0 Axes>
### NOTE: ALL THE COMMANDS NEED TO GO IN THE SAME CELL!

To begin we create a figure instance. Then we can add axes to that figure:
```python
#Create Figure (empty canvas)

fig = plt.figure()
​
#Add set of axes to figure
axes = fig.add_axes([0, 0, 1, 1]) # left, bottom, width, height (range 0 to 1)
​
#Plot on that set of axes
axes.plot(x, y)
​
plt.show()
```
![4](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/68556cd5-1e1a-4f81-9ad0-b7bbf4dc4aa4)
```python
#Create Figure (empty canvas)
fig = plt.figure()
​
#Add set of axes to figure
axes = fig.add_axes([0, 0, 1, 1]) # left, bottom, width, height (range 0 to 1)
​
#Plot on that set of axes
axes.plot(a, b)
​
plt.show()
```
![5](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/666961e7-a699-49a0-95be-9c89a09741b1)

### Adding another set of axes to the Figure
So far we've only seen one set of axes on this figure object, but we can keep adding new axes on to it at any location and size we want. We can then plot on that new set of axes.
```python
type(fig)
```
matplotlib.figure.Figure
Code is a little more complicated, but the advantage is that we now have full control of where the plot axes are placed, and we can easily add more than one axis to the figure. Note how we're plotting a,b twice here
```python
#Creates blank canvas
fig = plt.figure()
​
axes1 = fig.add_axes([0, 0, 1, 1]) # Large figure
axes2 = fig.add_axes([0.2, 0.2, 0.5, 0.5]) # Smaller figure
​
#Larger Figure Axes 1
axes1.plot(a, b)
​
#Use set_ to add to the axes figure
axes1.set_xlabel('X Label')
axes1.set_ylabel('Y Label')
axes1.set_title('Big Figure')
​
#Insert Figure Axes 2
axes2.plot(a,b)
axes2.set_title('Small Figure');
```
![6](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/3aa5e0f1-bca3-43ed-a0d0-9c7a32be71bd)

Let's move the small figure and edit its parameters.
```python
#Creates blank canvas
fig = plt.figure()
​
axes1 = fig.add_axes([0, 0, 1, 1]) # Large figure
axes2 = fig.add_axes([0.2, 0.5, 0.25, 0.25]) # Smaller figure
​
#Larger Figure Axes 1
axes1.plot(a, b)
​
#Use set_ to add to the axes figure
axes1.set_xlabel('X Label')
axes1.set_ylabel('Y Label')
axes1.set_title('Big Figure')
​
#Insert Figure Axes 2
axes2.plot(a,b)
axes2.set_xlim(8,10)
axes2.set_ylim(4000,10000)
axes2.set_xlabel('X')
axes2.set_ylabel('Y')
axes2.set_title('Zoomed In');
```
![7](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/d665b98c-0d6c-4653-8248-4789cb69ee74)

You can add as many axes on to the same figure as you want, even outside of the main figure if the length and width correspond to this.
```python
#Creates blank canvas
fig = plt.figure()
​
axes1 = fig.add_axes([0, 0, 1, 1]) # Full figure
axes2 = fig.add_axes([0.2, 0.5, 0.25, 0.25]) # Smaller figure
axes3 = fig.add_axes([1, 1, 0.25, 0.25]) # Starts at top right corner!
​
#Larger Figure Axes 1
axes1.plot(a, b)
​
#Use set_ to add to the axes figure
axes1.set_xlabel('X Label')
axes1.set_ylabel('Y Label')
axes1.set_title('Big Figure')
​
#Insert Figure Axes 2
axes2.plot(a,b)
axes2.set_xlim(8,10)
axes2.set_ylim(4000,10000)
axes2.set_xlabel('X')
axes2.set_ylabel('Y')
axes2.set_title('Zoomed In');
​
#Insert Figure Axes 3
axes3.plot(a,b)
​```
[<matplotlib.lines.Line2D at 0x1cd42ad2888>]
![8](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/90f0bedb-940e-4ec4-b323-0b0fd73246fb)

### Figure Parameters
```python
#Creates blank canvas
fig = plt.figure(figsize=(12,8),dpi=100)
​
axes1 = fig.add_axes([0, 0, 1, 1])
​
axes1.plot(a,b)
```
[<matplotlib.lines.Line2D at 0x1cd42d53848>]
![9](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/d342bfbb-9ab2-4d9a-ae33-cf1bfd8dba18)

### Exporting a Figure
```python
fig = plt.figure()
​
axes1 = fig.add_axes([0, 0, 1, 1])
​
axes1.plot(a,b)
axes1.set_xlabel('X')
​
# bbox_inches ='tight' automatically makes sure the bounding box is correct
fig.savefig('figure.png',bbox_inches='tight')
```
![10](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/787a105f-b632-4c59-b817-9cbcfc270f55)
```python
# Creates blank canvas
fig = plt.figure(figsize=(12,8))
​
axes1 = fig.add_axes([0, 0, 1, 1]) # Full figure
axes2 = fig.add_axes([1, 1, 0.25, 0.25]) # Starts at top right corner!
​
# Larger Figure Axes 1
axes1.plot(x,y)
​
# Insert Figure Axes 2
axes2.plot(x,y)
​
fig.savefig('test.png',bbox_inches='tight')
```
![11](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/784c883b-77f5-490c-86e4-e46ed77001fb)
## Matplotlib subplots
import the matplotlib.pyplot module under the name plt (the tidy way):
![12](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/2694643f-7fdb-447d-a07b-67daac5cafd9)
```python
# COMMON MISTAKE!
# DON'T FORGET THE .PYPLOT part
​
import matplotlib.pyplot as plt
```
NOTE: For users running .py scripts in an IDE like PyCharm or Sublime Text Editor. You will not see the plots in a notebook, instead if you are using another editor, you'll use: plt.show() at the end of all your plotting commands to have the figure pop up in another window.

### The Data
``` python
import numpy as np
a = np.linspace(0,10,11)
b = a ** 4
a
```
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
```python
b
```
array([0.000e+00, 1.000e+00, 1.600e+01, 8.100e+01, 2.560e+02, 6.250e+02,
       1.296e+03, 2.401e+03, 4.096e+03, 6.561e+03, 1.000e+04]) 
```python
x = np.arange(0,10)
y = 2 * x
x
```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```python
y
```
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])
## plt.subplots()
NOTE: Make sure you put the commands all together in the same cell as we do in this notebook and video!

The plt.subplots() object will act as a more automatic axis manager. This makes it much easier to show multiple plots side by side.

Note how we use tuple unpacking to grba both the Figure object and a numpy array of axes:
```python
#Use similar to plt.figure() except use tuple unpacking to grab fig and axes
fig, axes = plt.subplots()
​
#Now use the axes object to add stuff to plot
axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title'); #; hides Out[]
```
![13](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/24759ebe-fcae-4c1a-be4f-d26e3bcd9576)

### Adding rows and columns
Then you can specify the number of rows and columns when creating the subplots() object:
```python
#Empty canvas of 1 by 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2)
```
![14](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/3e5cad56-9704-42cb-a04f-844b504f9d74)
```python
#Axes is an array of axes to plot on
axes
```
array([<matplotlib.axes._subplots.AxesSubplot object at 0x0000023521E20588>,
       <matplotlib.axes._subplots.AxesSubplot object at 0x0000023521E5D8C8>],
      dtype=object)
```python
axes.shape
```
(2,)
```python
#Empty canvas of 2 by 2 subplots
fig, axes = plt.subplots(nrows=2, ncols=2)
```
![15](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/153abcc0-6d10-412a-b93d-e926ea08394c)
```python
axes
```
array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000023521ED5E48>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x0000023521F09D88>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x0000023521F45308>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x0000023521F79D88>]],
      dtype=object)
```python
axes.shape
```
(2, 2)
### Plotting on axes objects
Just as before, we simple .plot() on the axes objects, and we can also use the .set_ methods on each axes.

Let's explore this, make sure this is all in the same cell:
``` python
fig,axes = plt.subplots(nrows=1,ncols=2)
​
for axe in axes:
    axe.plot(x,y)
```
![16](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/915f0ca2-b6de-48f3-93c3-204545484152)
``` python
fig,axes = plt.subplots(nrows=1,ncols=2)
axes[0].plot(a,b)
axes[1].plot(x,y)
```
[<matplotlib.lines.Line2D at 0x2352216ce88>]
![17](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/1262fcb0-0a93-4339-ada3-91019996d354)
``` python
###NOTE! This returns 2 dimensional array
fig,axes = plt.subplots(nrows=2,ncols=2)
​
axes[0][0].plot(a,b)
axes[1][1].plot(x,y)
```
[<matplotlib.lines.Line2D at 0x2352229c648>]

![18](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/b5093c20-e317-453d-beb9-926701461f40)

A common issue with matplolib is overlapping subplots or figures. We ca use fig.tight_layout() or plt.tight_layout() method, which automatically adjusts the positions of the axes on the figure canvas so that there is no overlapping content:
``` python
#NOTE! This returns 2 dimensional array
fig,axes = plt.subplots(nrows=2,ncols=2)
​
axes[0][0].plot(a,b)
axes[1][1].plot(x,y)  
​
plt.tight_layout()
```
![19](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/aad3ff80-c795-4bc9-9da8-e12a690296a8)

### Parameters on subplots()
Recall we have both the Figure object and the axes. Meaning we can edit properties at both levels.
``` python
fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
​
# SET YOUR AXES PARAMETERS FIRST
​
# Parameters at the axes level
axes[0][0].plot(a,b)
axes[0][0].set_title('0 0 Title')
​
​
axes[1][1].plot(x,y)
axes[1][1].set_title('1 1 Title')
axes[1][1].set_xlabel('1 1 X Label')
​
axes[0][1].plot(y,x)
axes[1][0].plot(b,a)
​
# THEN SET OVERALL FIGURE PARAMETERS
​
# Parameters at the Figure level
fig.suptitle("Figure Level",fontsize=16)
​
​
plt.show()
```
![20](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/917a9dff-c903-4010-8af4-f7fa1ea73cce)

### Manual spacing on subplots()
Use .subplots_adjust to adjust spacing manually.
Full Details Here: https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
Example 

left = 0.125 # the left side of the subplots of the figure
right = 0.9 # the right side of the subplots of the figure
bottom = 0.1 # the bottom of the subplots of the figure
top = 0.9 # the top of the subplots of the figure
wspace = 0.2 # the amount of width reserved for space between subplots,
        # expressed as a fraction of the average axis width
hspace = 0.2 # the amount of height reserved for space between subplots,
        # expressed as a fraction of the average axis height
``` python
fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
​
# Parameters at the axes level
axes[0][0].plot(a,b)
axes[1][1].plot(x,y)
axes[0][1].plot(y,x)
axes[1][0].plot(b,a)
​
# Use left,right,top, bottom to stretch subplots
# Use wspace,hspace to add spacing between subplots
fig.subplots_adjust(left=None,
    bottom=None,
    right=None,
    top=None,
    wspace=0.9,
    hspace=0.1,)
​
plt.show()
```
![21](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/1d2f4e74-732a-4f01-a53d-cf83bebc8264)

## Exporting plt.subplots()
``` python
# NOTE! This returns 2 dimensional array
fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
​
axes[0][0].plot(a,b)
axes[1][1].plot(x,y)
axes[0][1].plot(y,x)
axes[1][0].plot(b,a)
​
fig.savefig('subplots.png',bbox_inches='tight')
​
plt.show()
```
![22](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/1f0786fe-3794-4a32-833b-bf7624fe9c7d)
## Matplotlib-Styling-Plots
Import the matplotlib.pyplot module under the name plt (the tidy way):
```python
# COMMON MISTAKE!
# DON'T FORGET THE .PYPLOT part
​
import matplotlib.pyplot as plt
```
NOTE: For users running .py scripts in an IDE like PyCharm or Sublime Text Editor. You will not see the plots in a notebook, instead if you are using another editor, you'll use: plt.show() at the end of all your plotting commands to have the figure pop up in another window.

### The Data
```python
import numpy as np
x = np.arange(0,10)
y = 2 * x
```
### Legends
You can use the label="label text" keyword argument when plots or other objects are added to the figure, and then using the legend method without arguments to add the legend to the figure:
```python
fig = plt.figure()
​
ax = fig.add_axes([0,0,1,1])
​
ax.plot(x, x**2, label="x**2")
ax.plot(x, x**3, label="x**3")
ax.legend()
```
<matplotlib.legend.Legend at 0x151b84e0c08>
![23](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/8da4555d-5c5f-4eca-9da2-7fcf09a031d7)

Notice how legend could potentially overlap some of the actual plot!

The legend function takes an optional keyword argument loc that can be used to specify where in the figure the legend is to be drawn. The allowed values of loc are numerical codes for the various places the legend can be drawn. See the documentation page for details. Some of the most common loc values are:
```python
# Lots of options....
​
ax.legend(loc=1) # upper right corner
ax.legend(loc=2) # upper left corner
ax.legend(loc=3) # lower left corner
ax.legend(loc=4) # lower right corner
​
# .. many more options are available
​
# Most common to choose
ax.legend(loc=0) # let matplotlib decide the optimal location
fig
```
![24](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/7f9d62bb-4217-42b7-872e-68a689b6f41b)
```python
ax.legend(loc=(1.1,0.5)) # manually set location
fig
```
![25](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/d0c8dad7-d883-4674-bfca-8380908d3c1a)

### Setting colors, linewidths, linetypes
Matplotlib gives you a lot of options for customizing colors, linewidths, and linetypes.

There is the basic MATLAB like syntax (which I would suggest you avoid using unless you already feel really comfortable with MATLAB). Instead let's focus on the keyword parameters.

#### Quick View:
##### Colors with MatLab like syntax
With matplotlib, we can define the colors of lines and other graphical elements in a number of ways. First of all, we can use the MATLAB-like syntax where 'b' means blue, 'g' means green, etc. The MATLAB API for selecting line styles are also supported: where, for example, 'b.-' means a blue line with dots:
```python
# MATLAB style line color and style 
fig, ax = plt.subplots()
ax.plot(x, x**2, 'b.-') # blue line with dots
ax.plot(x, x**3, 'g--') # green dashed line
```
[<matplotlib.lines.Line2D at 0x151b8c263c8>]
![26](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/2ce7d13a-46df-47f1-a3fc-d254b07039fc)

### Suggested Approach: Use keyword arguments
#### Colors with the color parameter
We can also define colors by their names or RGB hex codes and optionally provide an alpha value using the color and alpha keyword arguments. Alpha indicates opacity.
```python
fig, ax = plt.subplots()
​
ax.plot(x, x+1, color="blue", alpha=0.5) # half-transparant
ax.plot(x, x+2, color="#8B008B")        # RGB hex code
ax.plot(x, x+3, color="#FF8C00")        # RGB hex code
```
[<matplotlib.lines.Line2D at 0x151b8c7fa08>]
![27](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/5f0cfc9f-dd45-40e4-936f-4668579ea318)

### Line and marker styles
#### Linewidth
To change the line width, we can use the linewidth or lw keyword argument.
```python
fig, ax = plt.subplots(figsize=(12,6))
​
# Use linewidth or lw
ax.plot(x, x-1, color="red", linewidth=0.25)
ax.plot(x, x-2, color="red", lw=0.50)
ax.plot(x, x-3, color="red", lw=1)
ax.plot(x, x-4, color="red", lw=10)
```
[<matplotlib.lines.Line2D at 0x151b6dda608>]
![28](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/dd2145f0-e06f-422a-8b52-a6ed7f9987eb)


#### Linestyles
There are many linestyles to choose from, here is the selection:
```python
# possible linestype options ‘--‘, ‘–’, ‘-.’, ‘:’, ‘steps’
fig, ax = plt.subplots(figsize=(12,6))
​
ax.plot(x, x-1, color="green", lw=3, linestyle='-') # solid
ax.plot(x, x-2, color="green", lw=3, ls='-.') # dash and dot
ax.plot(x, x-3, color="green", lw=3, ls=':') # dots
ax.plot(x, x-4, color="green", lw=3, ls='--') # dashes
```
[<matplotlib.lines.Line2D at 0x151b6df5d48>]
![29](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/3eea3843-4970-49b1-ae56-7f3d7103a7bd)

#### Custom linestyle dash
The dash sequence is a sequence of floats of even length describing the length of dashes and spaces in points.

For example, (5, 2, 1, 2) describes a sequence of 5 point and 1 point dashes separated by 2 point spaces.

First, we see we can actually "grab" the line from the .plot() command
```python
fig, ax = plt.subplots(figsize=(12,6))
​
lines = ax.plot(x,x)
​
print(type(lines))
```
<class 'list'>
![30](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/d05ce2b5-8f77-43d2-891c-65c63c6e9de4)
```python
fig, ax = plt.subplots(figsize=(12,6))
# custom dash
lines = ax.plot(x, x+8, color="black", lw=5)
lines[0].set_dashes([10, 10]) # format: line length, space length
```
![31](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/f90bc48a-3747-48c9-9bc9-15692d4c9609)
```python
fig, ax = plt.subplots(figsize=(12,6))
# custom dash
lines = ax.plot(x, x+8, color="black", lw=5)
lines[0].set_dashes([1, 1,1,1,10,10]) # format: line length, space length
```
### Markers
We've technically always been plotting points, and matplotlib has been automatically drawing a line between these points for us. Let's explore how to place markers at each of these points.

#### Markers Style
Huge list of marker types can be found here: https://matplotlib.org/3.2.2/api/markers_api.html
```python
fig, ax = plt.subplots(figsize=(12,6))
​
# Use marker for string code
# Use markersize or ms for size
​
ax.plot(x, x-1,marker='+',markersize=20)
ax.plot(x, x-2,marker='o',ms=20) #ms can be used for markersize
ax.plot(x, x-3,marker='s',ms=20,lw=0) # make linewidth zero to see only markers
ax.plot(x, x-4,marker='1',ms=20)
```
[<matplotlib.lines.Line2D at 0x151b89eed08>]
![32](https://github.com/RiziaPrabin/Python-for-ML/assets/160464556/549cafcc-3244-4dcd-9dfc-83538e422d7a)

#### Custom marker edges, thickness,size,and style
```python
fig, ax = plt.subplots(figsize=(12,6))
# marker size and color
ax.plot(x, x, color="black", lw=1, ls='-', marker='s', markersize=20, 
        markerfacecolor="red", markeredgewidth=8, markeredgecolor="blue");
```
