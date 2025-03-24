Learn how to summarize the columns available in an R data frame. 
  You will also learn how to chain operations together with the
  pipe operator, and how to compute grouped summaries using.

## Welcome!

Hey there! Ready for the first lesson?

The dfply package makes it possible to do R's dplyr-style data manipulation with pipes in python on pandas DataFrames.

[dfply website here](https://github.com/kieferk/dfply)

[![](https://www.rforecology.com/pipes_image0.png "https://github.com/kieferk/dfply"){width="600"}](https://github.com/kieferk/dfply)


```python
import pandas as pd
import seaborn as sns
cars = sns.load_dataset('mpg')
from dfply import *
cars >> head(30)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>usa</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>usa</td>
      <td>ford torino</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15.0</td>
      <td>8</td>
      <td>429.0</td>
      <td>198.0</td>
      <td>4341</td>
      <td>10.0</td>
      <td>70</td>
      <td>usa</td>
      <td>ford galaxie 500</td>
    </tr>
    <tr>
      <th>6</th>
      <td>14.0</td>
      <td>8</td>
      <td>454.0</td>
      <td>220.0</td>
      <td>4354</td>
      <td>9.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet impala</td>
    </tr>
    <tr>
      <th>7</th>
      <td>14.0</td>
      <td>8</td>
      <td>440.0</td>
      <td>215.0</td>
      <td>4312</td>
      <td>8.5</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth fury iii</td>
    </tr>
    <tr>
      <th>8</th>
      <td>14.0</td>
      <td>8</td>
      <td>455.0</td>
      <td>225.0</td>
      <td>4425</td>
      <td>10.0</td>
      <td>70</td>
      <td>usa</td>
      <td>pontiac catalina</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15.0</td>
      <td>8</td>
      <td>390.0</td>
      <td>190.0</td>
      <td>3850</td>
      <td>8.5</td>
      <td>70</td>
      <td>usa</td>
      <td>amc ambassador dpl</td>
    </tr>
    <tr>
      <th>10</th>
      <td>15.0</td>
      <td>8</td>
      <td>383.0</td>
      <td>170.0</td>
      <td>3563</td>
      <td>10.0</td>
      <td>70</td>
      <td>usa</td>
      <td>dodge challenger se</td>
    </tr>
    <tr>
      <th>11</th>
      <td>14.0</td>
      <td>8</td>
      <td>340.0</td>
      <td>160.0</td>
      <td>3609</td>
      <td>8.0</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth 'cuda 340</td>
    </tr>
    <tr>
      <th>12</th>
      <td>15.0</td>
      <td>8</td>
      <td>400.0</td>
      <td>150.0</td>
      <td>3761</td>
      <td>9.5</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet monte carlo</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14.0</td>
      <td>8</td>
      <td>455.0</td>
      <td>225.0</td>
      <td>3086</td>
      <td>10.0</td>
      <td>70</td>
      <td>usa</td>
      <td>buick estate wagon (sw)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>24.0</td>
      <td>4</td>
      <td>113.0</td>
      <td>95.0</td>
      <td>2372</td>
      <td>15.0</td>
      <td>70</td>
      <td>japan</td>
      <td>toyota corona mark ii</td>
    </tr>
    <tr>
      <th>15</th>
      <td>22.0</td>
      <td>6</td>
      <td>198.0</td>
      <td>95.0</td>
      <td>2833</td>
      <td>15.5</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth duster</td>
    </tr>
    <tr>
      <th>16</th>
      <td>18.0</td>
      <td>6</td>
      <td>199.0</td>
      <td>97.0</td>
      <td>2774</td>
      <td>15.5</td>
      <td>70</td>
      <td>usa</td>
      <td>amc hornet</td>
    </tr>
    <tr>
      <th>17</th>
      <td>21.0</td>
      <td>6</td>
      <td>200.0</td>
      <td>85.0</td>
      <td>2587</td>
      <td>16.0</td>
      <td>70</td>
      <td>usa</td>
      <td>ford maverick</td>
    </tr>
    <tr>
      <th>18</th>
      <td>27.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>88.0</td>
      <td>2130</td>
      <td>14.5</td>
      <td>70</td>
      <td>japan</td>
      <td>datsun pl510</td>
    </tr>
    <tr>
      <th>19</th>
      <td>26.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>46.0</td>
      <td>1835</td>
      <td>20.5</td>
      <td>70</td>
      <td>europe</td>
      <td>volkswagen 1131 deluxe sedan</td>
    </tr>
    <tr>
      <th>20</th>
      <td>25.0</td>
      <td>4</td>
      <td>110.0</td>
      <td>87.0</td>
      <td>2672</td>
      <td>17.5</td>
      <td>70</td>
      <td>europe</td>
      <td>peugeot 504</td>
    </tr>
    <tr>
      <th>21</th>
      <td>24.0</td>
      <td>4</td>
      <td>107.0</td>
      <td>90.0</td>
      <td>2430</td>
      <td>14.5</td>
      <td>70</td>
      <td>europe</td>
      <td>audi 100 ls</td>
    </tr>
    <tr>
      <th>22</th>
      <td>25.0</td>
      <td>4</td>
      <td>104.0</td>
      <td>95.0</td>
      <td>2375</td>
      <td>17.5</td>
      <td>70</td>
      <td>europe</td>
      <td>saab 99e</td>
    </tr>
    <tr>
      <th>23</th>
      <td>26.0</td>
      <td>4</td>
      <td>121.0</td>
      <td>113.0</td>
      <td>2234</td>
      <td>12.5</td>
      <td>70</td>
      <td>europe</td>
      <td>bmw 2002</td>
    </tr>
    <tr>
      <th>24</th>
      <td>21.0</td>
      <td>6</td>
      <td>199.0</td>
      <td>90.0</td>
      <td>2648</td>
      <td>15.0</td>
      <td>70</td>
      <td>usa</td>
      <td>amc gremlin</td>
    </tr>
    <tr>
      <th>25</th>
      <td>10.0</td>
      <td>8</td>
      <td>360.0</td>
      <td>215.0</td>
      <td>4615</td>
      <td>14.0</td>
      <td>70</td>
      <td>usa</td>
      <td>ford f250</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>200.0</td>
      <td>4376</td>
      <td>15.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevy c20</td>
    </tr>
    <tr>
      <th>27</th>
      <td>11.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>210.0</td>
      <td>4382</td>
      <td>13.5</td>
      <td>70</td>
      <td>usa</td>
      <td>dodge d200</td>
    </tr>
    <tr>
      <th>28</th>
      <td>9.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>193.0</td>
      <td>4732</td>
      <td>18.5</td>
      <td>70</td>
      <td>usa</td>
      <td>hi 1200d</td>
    </tr>
    <tr>
      <th>29</th>
      <td>27.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>88.0</td>
      <td>2130</td>
      <td>14.5</td>
      <td>71</td>
      <td>japan</td>
      <td>datsun pl510</td>
    </tr>
  </tbody>
</table>
</div>



## The \>\> and \>\>=

dfply works directly on pandas DataFrames, chaining operations on the data with the >> operator, or alternatively starting with >>= for inplace operations.

*The X DataFrame symbol*

The DataFrame as it is passed through the piping operations is represented by the symbol X. It records the actions you want to take (represented by the Intention class), but does not evaluate them until the appropriate time. Operations on the DataFrame are deferred. Selecting two of the columns, for example, can be done using the symbolic X DataFrame during the piping operations.

### Exercise 1.

Select the columns 'mpg' and 'horsepower' from the cars DataFrame.


```python
cars >> select(X.mpg, X.horsepower) >> head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>horsepower</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>130.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>150.0</td>
    </tr>
  </tbody>
</table>
</div>



## Selecting and dropping

There are two functions for selection, inverse of each other: select and drop. The select and drop functions accept string labels, integer positions, and/or symbolically represented column names (X.column). They also accept symbolic "selection filter" functions, which will be covered shortly.

### Exercise 2.

Select the columns 'mpg' and 'horsepower' from the cars DataFrame using the drop function.


```python
cars >> drop(X.weight, X.origin, X.cylinders, X.displacement, X.acceleration,  X.model_year, X.name) >> head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>horsepower</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>130.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>150.0</td>
    </tr>
  </tbody>
</table>
</div>



## Selection using \~

One particularly nice thing about dplyr's selection functions is that you can drop columns inside of a select statement by putting a subtraction sign in front, like so: ... %>% select(-col). The same can be done in dfply, but instead of the subtraction operator you use the tilde ~.

### Exercise 3.

Select all columns except 'model_year', and 'name' from the cars DataFrame.


```python
cars >> select(~X.model_year, ~X.name) >> head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>usa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>usa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>usa</td>
    </tr>
  </tbody>
</table>
</div>



## Filtering columns

The vanilla select and drop functions are useful, but there are a variety of selection functions inspired by dplyr available to make selecting and dropping columns a breeze. These functions are intended to be put inside of the select and drop functions, and can be paired with the ~ inverter.

First, a quick rundown of the available functions:

-   starts_with(prefix): find columns that start with a string prefix.
-   ends_with(suffix): find columns that end with a string suffix.
-   contains(substr): find columns that contain a substring in their name.
-   everything(): all columns.
-   columns_between(start_col, end_col, inclusive=True): find columns between a specified start and end column. The inclusive boolean keyword argument indicates whether the end column should be included or not.
-   columns_to(end_col, inclusive=True): get columns up to a specified end column. The inclusive argument indicates whether the ending column should be included or not.
-   columns_from(start_col): get the columns starting at a specified column.

### Exercise 4.

The selection filter functions are best explained by example. Let's say I wanted to select only the columns that started with a "c":


```python
cars >> select(starts_with('c')) >> head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cylinders</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



### Exercise 5.

Select the columns that contain the substring "e" from the cars DataFrame.


```python
cars >> select(contains('e')) >> head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>plymouth satellite</td>
    </tr>
  </tbody>
</table>
</div>



### Exercise 6.

Select the columns that are between 'mpg' and 'origin' from the cars DataFrame.


```python
cars >> select(columns_between('mpg', 'origin')) >> head(3)
```

## Subsetting and filtering

### row_slice()

Slices of rows can be selected with the row_slice() function. You can pass single integer indices or a list of indices to select rows as with. This is going to be the same as using pandas' .iloc.

#### Exercise 7.

Select the first three rows from the cars DataFrame.


```python
cars >> row_slice([0, 1, 2])
```

### distinct()

Selection of unique rows is done with distinct(), which similarly passes arguments and keyword arguments through to the DataFrame's .drop_duplicates() method.

#### Exercise 8.

Select the unique rows from the 'origin' column in the cars DataFrame.


```python
cars >> distinct(X.origin)
```

## mask()

Filtering rows with logical criteria is done with mask(), which accepts boolean arrays "masking out" False labeled rows and keeping True labeled rows. These are best created with logical statements on symbolic Series objects as shown below. Multiple criteria can be supplied as arguments and their intersection will be used as the mask.

### Exercise 9.

Filter the cars DataFrame to only include rows where the 'mpg' is greater than 20, origin Japan, and display the first three rows:


```python
cars >> mask(X.mpg>20, X.origin=="japan" ) >> head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>24.0</td>
      <td>4</td>
      <td>113.0</td>
      <td>95.0</td>
      <td>2372</td>
      <td>15.0</td>
      <td>70</td>
      <td>japan</td>
      <td>toyota corona mark ii</td>
    </tr>
    <tr>
      <th>18</th>
      <td>27.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>88.0</td>
      <td>2130</td>
      <td>14.5</td>
      <td>70</td>
      <td>japan</td>
      <td>datsun pl510</td>
    </tr>
    <tr>
      <th>29</th>
      <td>27.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>88.0</td>
      <td>2130</td>
      <td>14.5</td>
      <td>71</td>
      <td>japan</td>
      <td>datsun pl510</td>
    </tr>
  </tbody>
</table>
</div>



## pull()

The pull() function is used to extract a single column from a DataFrame as a pandas Series. This is useful for passing a single column to a function or for further manipulation.

### Exercise 10.

Extract the 'mpg' column from the cars DataFrame, japanese origin, model year 70s, and display the first three rows.


```python
cars >> mask((X.model_year >= 70) & (X.model_year<80) & (X.origin == "japan")) >> select(X.mpg) >> head(3)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>24.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>27.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
</div>



## DataFrame transformation

*mutate()*

The mutate() function is used to create new columns or modify existing columns. It accepts keyword arguments of the form new_column_name = new_column_value, where new_column_value is a symbolic Series object.

### Exercise 11.

Create a new column 'mpg_per_cylinder' in the cars DataFrame that is the result of dividing the 'mpg' column by the 'cylinders' column.


```python
cars >> mutate(mpg_per_cylinder = X.mpg / X.cylinders) >> head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
      <th>mpg_per_cylinder</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
      <td>2.250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
      <td>1.875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth satellite</td>
      <td>2.250</td>
    </tr>
  </tbody>
</table>
</div>




*transmute()*

The transmute() function is a combination of a mutate and a selection of the created variables.

### Exercise 12.

Create a new column 'mpg_per_cylinder' in the cars DataFrame that is the result of dividing the 'mpg' column by the 'cylinders' column, and display only the new column.


```python
cars >> transmute(mpg_per_cylinder = X.mpg / X.cylinders) >> head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg_per_cylinder</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.250</td>
    </tr>
  </tbody>
</table>
</div>



## Grouping

*group_by() and ungroup()*

The group_by() function is used to group the DataFrame by one or more columns. This is useful for creating groups of rows that can be summarized or transformed together. The ungroup() function is used to remove the grouping.

### Exercise 13.

Group the cars DataFrame by the 'origin' column and calculate the lead of the 'mpg' column.


```python
cars >> group_by(X.origin)>> mutate(lead_mpg=lead(X.mpg,1)) >> select(X.lead_mpg) >> head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin</th>
      <th>lead_mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>europe</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>europe</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>europe</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>japan</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>japan</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>japan</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>usa</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>usa</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>usa</td>
      <td>16.0</td>
    </tr>
  </tbody>
</table>
</div>



## Reshaping

*arrange()*

The arrange() function is used to sort the DataFrame by one or more columns. This is useful for reordering the rows of the DataFrame.

### Exercise 14.

Sort the cars DataFrame by the 'mpg' column in descending order.


```python
cars >> arrange(X.mpg, ascending=True) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>9.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>193.0</td>
      <td>4732</td>
      <td>18.5</td>
      <td>70</td>
      <td>usa</td>
      <td>hi 1200d</td>
    </tr>
    <tr>
      <th>25</th>
      <td>10.0</td>
      <td>8</td>
      <td>360.0</td>
      <td>215.0</td>
      <td>4615</td>
      <td>14.0</td>
      <td>70</td>
      <td>usa</td>
      <td>ford f250</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>200.0</td>
      <td>4376</td>
      <td>15.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevy c20</td>
    </tr>
    <tr>
      <th>103</th>
      <td>11.0</td>
      <td>8</td>
      <td>400.0</td>
      <td>150.0</td>
      <td>4997</td>
      <td>14.0</td>
      <td>73</td>
      <td>usa</td>
      <td>chevrolet impala</td>
    </tr>
    <tr>
      <th>124</th>
      <td>11.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>180.0</td>
      <td>3664</td>
      <td>11.0</td>
      <td>73</td>
      <td>usa</td>
      <td>oldsmobile omega</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>326</th>
      <td>43.4</td>
      <td>4</td>
      <td>90.0</td>
      <td>48.0</td>
      <td>2335</td>
      <td>23.7</td>
      <td>80</td>
      <td>europe</td>
      <td>vw dasher (diesel)</td>
    </tr>
    <tr>
      <th>394</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52.0</td>
      <td>2130</td>
      <td>24.6</td>
      <td>82</td>
      <td>europe</td>
      <td>vw pickup</td>
    </tr>
    <tr>
      <th>325</th>
      <td>44.3</td>
      <td>4</td>
      <td>90.0</td>
      <td>48.0</td>
      <td>2085</td>
      <td>21.7</td>
      <td>80</td>
      <td>europe</td>
      <td>vw rabbit c (diesel)</td>
    </tr>
    <tr>
      <th>329</th>
      <td>44.6</td>
      <td>4</td>
      <td>91.0</td>
      <td>67.0</td>
      <td>1850</td>
      <td>13.8</td>
      <td>80</td>
      <td>japan</td>
      <td>honda civic 1500 gl</td>
    </tr>
    <tr>
      <th>322</th>
      <td>46.6</td>
      <td>4</td>
      <td>86.0</td>
      <td>65.0</td>
      <td>2110</td>
      <td>17.9</td>
      <td>80</td>
      <td>japan</td>
      <td>mazda glc</td>
    </tr>
  </tbody>
</table>
<p>398 rows × 9 columns</p>
</div>




*rename()*

The rename() function is used to rename columns in the DataFrame. It accepts keyword arguments of the form new_column_name = old_column_name.

### Exercise 15.

Rename the 'mpg' column to 'miles_per_gallon' in the cars DataFrame.


```python
cars>> rename(miles_per_gallon=X.mpg)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>miles_per_gallon</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>usa</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>usa</td>
      <td>ford torino</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>393</th>
      <td>27.0</td>
      <td>4</td>
      <td>140.0</td>
      <td>86.0</td>
      <td>2790</td>
      <td>15.6</td>
      <td>82</td>
      <td>usa</td>
      <td>ford mustang gl</td>
    </tr>
    <tr>
      <th>394</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52.0</td>
      <td>2130</td>
      <td>24.6</td>
      <td>82</td>
      <td>europe</td>
      <td>vw pickup</td>
    </tr>
    <tr>
      <th>395</th>
      <td>32.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84.0</td>
      <td>2295</td>
      <td>11.6</td>
      <td>82</td>
      <td>usa</td>
      <td>dodge rampage</td>
    </tr>
    <tr>
      <th>396</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79.0</td>
      <td>2625</td>
      <td>18.6</td>
      <td>82</td>
      <td>usa</td>
      <td>ford ranger</td>
    </tr>
    <tr>
      <th>397</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82.0</td>
      <td>2720</td>
      <td>19.4</td>
      <td>82</td>
      <td>usa</td>
      <td>chevy s-10</td>
    </tr>
  </tbody>
</table>
<p>398 rows × 9 columns</p>
</div>




*gather()*

The gather() function is used to reshape the DataFrame from wide to long format. It accepts keyword arguments of the form new_column_name = new_column_value, where new_column_value is a symbolic Series object.

### Exercise 16.

Reshape the cars DataFrame from wide to long format by gathering the columns 'mpg', 'horsepower', 'weight', 'acceleration', and 'displacement' into a new column 'variable' and their values into a new column 'value'.


```python
long = cars>> gather("variable", "value",[X.mpg, X.horsepower, X.weight, X.acceleration],add_id=True) 
long >> head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
      <th>_ID</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>307.0</td>
      <td>70</td>
      <td>usa</td>
      <td>chevrolet chevelle malibu</td>
      <td>0</td>
      <td>mpg</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>350.0</td>
      <td>70</td>
      <td>usa</td>
      <td>buick skylark 320</td>
      <td>1</td>
      <td>mpg</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>318.0</td>
      <td>70</td>
      <td>usa</td>
      <td>plymouth satellite</td>
      <td>2</td>
      <td>mpg</td>
      <td>18.0</td>
    </tr>
  </tbody>
</table>
</div>




*spread()*

Likewise, you can transform a "long" DataFrame into a "wide" format with the spread(key, values) function. Converting the previously created elongated DataFrame for example would be done like so.

### Exercise 17.

Reshape the cars DataFrame from long to wide format by spreading the 'variable' column into columns and their values into the 'value' column.


```python
long >> spread(X.variable, X.value) >> head(3)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>model_year</th>
      <th>origin</th>
      <th>name</th>
      <th>_ID</th>
      <th>acceleration</th>
      <th>horsepower</th>
      <th>mpg</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>70.0</td>
      <td>72</td>
      <td>japan</td>
      <td>mazda rx2 coupe</td>
      <td>71</td>
      <td>13.5</td>
      <td>97.0</td>
      <td>19.0</td>
      <td>2330.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>70.0</td>
      <td>73</td>
      <td>japan</td>
      <td>maxda rx3</td>
      <td>111</td>
      <td>13.5</td>
      <td>90.0</td>
      <td>18.0</td>
      <td>2124.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>70.0</td>
      <td>80</td>
      <td>japan</td>
      <td>mazda rx-7 gs</td>
      <td>334</td>
      <td>12.5</td>
      <td>100.0</td>
      <td>23.7</td>
      <td>2420.0</td>
    </tr>
  </tbody>
</table>
</div>




## Summarization

*summarize()*

The summarize() function is used to calculate summary statistics for groups of rows. It accepts keyword arguments of the form new_column_name = new_column_value, where new_column_value is a symbolic Series object.

### Exercise 18.

Calculate the mean 'mpg' for each group of 'origin' in the cars DataFrame.


```python
cars >> group_by(X.origin)>>summarize(mean_mpg=X.mpg.mean(
))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin</th>
      <th>mean_mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>europe</td>
      <td>27.891429</td>
    </tr>
    <tr>
      <th>1</th>
      <td>japan</td>
      <td>30.450633</td>
    </tr>
    <tr>
      <th>2</th>
      <td>usa</td>
      <td>20.083534</td>
    </tr>
  </tbody>
</table>
</div>




*summarize_each()*

The summarize_each() function is used to calculate summary statistics for groups of rows. It accepts keyword arguments of the form new_column_name = new_column_value, where new_column_value is a symbolic Series object.

### Exercise 19.

Calculate the mean 'mpg' and 'horsepower' for each group of 'origin' in the cars DataFrame.


```python
cars>> group_by(X.origin) >> summarize_each([np.mean], X.mpg, X.horsepower)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin</th>
      <th>mpg_mean</th>
      <th>horsepower_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>europe</td>
      <td>27.891429</td>
      <td>80.558824</td>
    </tr>
    <tr>
      <th>1</th>
      <td>japan</td>
      <td>30.450633</td>
      <td>79.835443</td>
    </tr>
    <tr>
      <th>2</th>
      <td>usa</td>
      <td>20.083534</td>
      <td>119.048980</td>
    </tr>
  </tbody>
</table>
</div>




*summarize() can of course be used with groupings as well.*

### Exercise 20.

Calculate the mean 'mpg' for each group of 'origin' and 'model_year' in the cars DataFrame.


```python
cars>> group_by(X.origin, X.model_year) >> summarize_each([np.mean], X.mpg)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model_year</th>
      <th>origin</th>
      <th>mpg_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>70</td>
      <td>europe</td>
      <td>25.200000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71</td>
      <td>europe</td>
      <td>28.750000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72</td>
      <td>europe</td>
      <td>22.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>73</td>
      <td>europe</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74</td>
      <td>europe</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75</td>
      <td>europe</td>
      <td>24.500000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>76</td>
      <td>europe</td>
      <td>24.250000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>77</td>
      <td>europe</td>
      <td>29.250000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>78</td>
      <td>europe</td>
      <td>24.950000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>79</td>
      <td>europe</td>
      <td>30.450000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>80</td>
      <td>europe</td>
      <td>37.288889</td>
    </tr>
    <tr>
      <th>11</th>
      <td>81</td>
      <td>europe</td>
      <td>31.575000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>82</td>
      <td>europe</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>70</td>
      <td>japan</td>
      <td>25.500000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>71</td>
      <td>japan</td>
      <td>29.500000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>72</td>
      <td>japan</td>
      <td>24.200000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>73</td>
      <td>japan</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>74</td>
      <td>japan</td>
      <td>29.333333</td>
    </tr>
    <tr>
      <th>18</th>
      <td>75</td>
      <td>japan</td>
      <td>27.500000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>76</td>
      <td>japan</td>
      <td>28.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>77</td>
      <td>japan</td>
      <td>27.416667</td>
    </tr>
    <tr>
      <th>21</th>
      <td>78</td>
      <td>japan</td>
      <td>29.687500</td>
    </tr>
    <tr>
      <th>22</th>
      <td>79</td>
      <td>japan</td>
      <td>32.950000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>80</td>
      <td>japan</td>
      <td>35.400000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>81</td>
      <td>japan</td>
      <td>32.958333</td>
    </tr>
    <tr>
      <th>25</th>
      <td>82</td>
      <td>japan</td>
      <td>34.888889</td>
    </tr>
    <tr>
      <th>26</th>
      <td>70</td>
      <td>usa</td>
      <td>15.272727</td>
    </tr>
    <tr>
      <th>27</th>
      <td>71</td>
      <td>usa</td>
      <td>18.100000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>72</td>
      <td>usa</td>
      <td>16.277778</td>
    </tr>
    <tr>
      <th>29</th>
      <td>73</td>
      <td>usa</td>
      <td>15.034483</td>
    </tr>
    <tr>
      <th>30</th>
      <td>74</td>
      <td>usa</td>
      <td>18.333333</td>
    </tr>
    <tr>
      <th>31</th>
      <td>75</td>
      <td>usa</td>
      <td>17.550000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>76</td>
      <td>usa</td>
      <td>19.431818</td>
    </tr>
    <tr>
      <th>33</th>
      <td>77</td>
      <td>usa</td>
      <td>20.722222</td>
    </tr>
    <tr>
      <th>34</th>
      <td>78</td>
      <td>usa</td>
      <td>21.772727</td>
    </tr>
    <tr>
      <th>35</th>
      <td>79</td>
      <td>usa</td>
      <td>23.478261</td>
    </tr>
    <tr>
      <th>36</th>
      <td>80</td>
      <td>usa</td>
      <td>25.914286</td>
    </tr>
    <tr>
      <th>37</th>
      <td>81</td>
      <td>usa</td>
      <td>27.530769</td>
    </tr>
    <tr>
      <th>38</th>
      <td>82</td>
      <td>usa</td>
      <td>29.450000</td>
    </tr>
  </tbody>
</table>
</div>


