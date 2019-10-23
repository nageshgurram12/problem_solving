
# coding: utf-8

# # Reading Data

# To do any data science, the first thing you're going to need is data. Let's start to look at a few common ways of reading data in Python. Along the way, we'll take a look at some language constructs that we'll use throughout the day and really start to get familiar with coding in Python.

# ### File handling in Python

# First, let's look at file objects in Python. We get a file handle by using the **open** function.

# In[1]:


csv_file = open("data/health_inspection_chi_sample.csv")


# File objects are lazy **iterators** (here, *stream objects*). Lazy means that they only do things, in this case read data, when you ask them to. You can call **next** on iterator objects to explicitly get the next item.

# In[ ]:


line = next(csv_file)


# In[ ]:


print(line)


# Here, the next item is a string which is the line in a file. In this case, it's the headers of the `csv` file.
# 
# The nice thing about **iterators** is that you can iterate through them with a for loop.

# In[ ]:


for line in csv_file:
    pass


# In[ ]:


print(line)


# After consuming the file, we need to close the open file-handle.

# In[ ]:


csv_file.close()


# We can avoid that by using what's called a **context manager** in Python. This is a really nice abstraction. You use context managers via the **with** statement.

# In[ ]:


with open("data/health_inspection_chi_sample.csv") as csv_file:
    for line in csv_file:
        pass


# By using the `open` function as a context manager, we get an automatic call to close the open file when we exit the context (determined by the indentation level). When working with files non-interactively, you'll almost always want to use open as a context manager.

# ## Exercise
# 
# Write some code that iterates through the file `data/health_inspection_chi_sample.json` twice. Only call `open` once, however, then close the file. Can you find out, programatically, how many characters are in the file (hint: have a look at the `tell` method of the file object)?

# In[ ]:


# Type your solution here


# In[ ]:


get_ipython().run_line_magic('load', 'solutions/read_json_twice.py')


# ### Reading Data: CSVs

# We peaked at this file above. We might already see some things we're going to have to clean up. Let's look at how we may do this. As always, Python has *batteries included*. The `csv` built-in module is great for basic data-munging needs.

# In[ ]:


import csv
from pprint import pprint


# Let's open a csv file and use csv's `reader` function to read in the data.

# In[ ]:


csv_file = open("data/health_inspection_chi_sample.csv")

reader = csv.reader(csv_file)


# `csv.reader` also returns a lazy iterator.

# In[ ]:


headers = next(reader)


# In[ ]:


pprint(headers)


# In[ ]:


line = next(reader)


# In[ ]:


pprint(line)


# The biggest difference in using `csv.reader` vs. iterating through the file is that it automatically splits the csv on commas and returns the line of the file split into a list.
# 
# You can control this behavior through a `Dialect` object. By default, `csv.reader` uses a Dialect object called "excel." Here let's look at the attributes of the excel dialect. Don't worry too much about the code used to achieve this. We'll look more at this later.

# In[ ]:


items = csv.excel.__dict__.items()

pprint({key: value for key, value in items if not key.startswith("_")})


# You might also find `DictReader` to be useful. 
# 
# First, let's back up in the file to the beginning.

# In[ ]:


csv_file.seek(0)


# In[ ]:


reader = csv.DictReader(csv_file)


# In[ ]:


pprint(next(reader))


# In[ ]:


csv_file.close()


# One last handy trick with the `csv` module is to use `Sniffer` to determine the csv dialect for you.

# In[ ]:


file_name = "data/health_inspection_chi_sample.csv"

with open(file_name) as csv_file:
    
    try:
        dialect = csv.Sniffer().sniff(csv_file.read(1024))
    except csv.Error as err:
        # log that this file format couldn't be deduced
        print(f"The format of {file_name} could not be detected.")
    else:
        csv_file.seek(0)
    
        dta = csv.reader(csv_file, dialect=dialect)


# A big part of increased productivity is saving yourself some work later. The first rule of dealing with data is probably something like "The data is against you. Act accordingly."
# 
# We set this block of code up to be pretty paranoid. Software engineers call this **defensive programming**. It's a good habit to get into when you're doing any data science work in Python. 
# 
# There are three things to note here. First, is the use of `Sniffer` at all. If `file_name` is a standard csv format, we'll be able to read it.
# 
# The second is using a `try/except/else` block. In a `try/except` block, any exception that is raised will trigger the code in the except block. If no exception is raised, the *optional* else block will run.
# 
# Let's take a look at a toy example to fix ideas.

# In[ ]:


try:
    1/0
except:
    print("Something went wrong.")


# In[ ]:


try:
    1/0
except ZeroDivisionError as err:
    print("Something went wrong.")
    raise(err)


# In[ ]:


try:
    1/0
except FileNotFoundError:
    print("This error isn't raised, so we're not here.")


# The final thing to note in the block above is the use of `print` to provide some information about what went wrong. Logging is another really good habit to get into, and print statements are the dead simplest way to log the behavior of your code.
# 
# In practice, you probably don't want to use `print`. You want to use the [logging](https://docs.python.org/3/library/logging.html) module, but we're not going to talk about best practices in logging anymore today.

# ### Reading Data: json

# Perhaps the second most common file format after CSVs is the `JSON` format. JSON stands for JavaScript Object Notation. When reading data from an API, for example, you will often encounter json files.

# Each line in the file `data/health_inspection_chi_sample.json` is a single json object that represents the same data above. 

# In[ ]:


get_ipython().system('head -n 1 data/health_inspection_chi_sample.json')


# We can use the `json` module to read and manipulate json data.

# In[ ]:


import json


# Since each line is a json object here, we need to iterate over the file and parse each line. We use the `json.loads` function here for "load string." The similar function `json.load` takes a file-like object.

# In[ ]:


dta = []

with open("data/health_inspection_chi_sample.json") as json_file:
    for line in json_file:
        line = json.loads(line)
        dta.append(line)

pprint(dta[0])


# `json.loads` places each json object into a Python dictionary, helpfully filling in `None` for `null` for missing values and otherwise preserving types. It also, works recursively as we see in the `location` field.

# ## Aside: List Comprehensions

# Let's take a look at another Pythonic concept, introduced a bit above, called a **list comprehension**. This is what's called *syntactic sugar*. It's a concise way to create a list.

# In[ ]:


[i for i in range(1, 6)]


# Alternatively, we could have made this list by writing

# In[ ]:


result_list = []

for i in range(1, 6):
    result_list.append(i)

result_list


# List comprehensions can contain logic.

# In[ ]:


x = ['a', 'b', 'c', 'd', '_e', '_f']


# In[ ]:


[i for i in x if not i.startswith('_')]


# You can also use a an else clause. Notice the slightly different syntax.

# In[ ]:


[i if not i.startswith('_') else 'skipped' for i in x]


# List comprehensions can be nested, though it's usually best practices not to go overboard. They can quickly become difficult to read.

# In[ ]:


matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
]


# Notice that we write this from outside to in.

# In[ ]:


[element for row in matrix for element in row]


# You can also create **dictionary comprehensions**.

# In[ ]:


pairs = [
    ("first_initial", "J"), 
    ("last_name", "Doe"), 
    ("address", "1234 Main Street")
]


# In[ ]:


{key: value for key, value in pairs}


# ## Exercise
# 
# Returning to the code that we introduced above, we can take further control over how a file with json objects is read in by using the `object_hook` argument. Say we wanted to remove the `location` field above. We don't need the `geoJSON` formatted information. We could do so with the `object_hook`. Write a function called `remove_entry` that removes the `'location'` field from each record in the `'data/health_inspection_chi_sample.json'` file.
# 
# Pass this function to the `object_hook` argument of `json.loads`. Be careful, `object_hook` will get called recursively on nested json objects.

# In[ ]:


# Type your solution here


# In[ ]:


get_ipython().run_line_magic('load', 'solutions/object_hook_json.py')


# ### Reading Data: pandas

# Now let's take a look at using [pandas]() to read data.

# #### Introducing Pandas
# 
# First, a few words of introduction for **pandas**. Pandas is a Python package providing fast, flexible, and expressive data structures designed to work with relational or labeled data. It is a high-level tool for doing practical, real world data analysis in Python.
# 
# You reach for pandas when you have:
# 
# * Tabular data with heterogeneously-typed columns
# * Ordered and unordered (not necessarily fixed-frequency) time series data.
# * Arbitrary matrix data with row and column labels
# 
# Almost any dataset can be converted to a pandas data structure for cleaning, transformation, and analysis.
# 
# Pandas is built on top of **numpy**, which we'll look a bit more at later.

# First, let's import the pandas.

# In[ ]:


import pandas as pd


# Let's set some display options.

# In[ ]:


pd.set_option("max.rows", 6)


# Pandas has facilities for reading csv files and files containing JSON records (and other formats). We can use `read_csv` for csv files.

# In[ ]:


pd.read_csv("data/health_inspection_chi_sample.csv")


# `read_csv` is one of the best/worst functions in all of Python. It's great because it does just about everything. It's terrible, because it does just about everything. Chances are if you have a special case that pandas `read_csv` will accomodate your needs. Go ahead and have a look at the `docstring`.
# 
# 
# ```python
# pd.read_csv?
# ```

# `read_csv` returns a pandas DataFrame. We'll take a deeper dive into DataFrames next when we start to clean this data set.

# The JSON counterpart to `read_csv` is `read_json`.

# ## Exercise
# 
# Use `pd.read_json` to read in the Chicago health inspections json sample in the `data` folder.

# In[ ]:


# Type your solution Here


# In[ ]:


get_ipython().run_line_magic('load', 'solutions/read_json.py')


# ### Reading Data: web data

# So far, we've seen some ways that we can read data from disk. As Data Scientists, we often need to go out and grab data from the Internet.
# 
# Generally Python is "batteries included" and reading data from the Internet is no exception, but there are some *great* packages out there. [requests](http://docs.python-requests.org/en/master/) is one of them for making HTTP requests.
# 
# Let's look at how we can use the [Chicago Data Portal](https://data.cityofchicago.org/) API to get this data in the first place. (I originally used San Francisco for this, but the data was just too clean to be terribly interesting.)

# In[ ]:


import requests


# We use the requests library to perform a GET request to the API, passing an optional query string via `params` to limit the returned number of records. The parameters are documented as part of the [Socrata Open Data API](https://dev.socrata.com/consumers/getting-started.html) (SODA).

# In[ ]:


response = requests.get(
    "https://data.cityofchicago.org/resource/cwig-ma7x.json", 
    params="$limit=10"
)


# Requests returns a [Reponse](http://docs.python-requests.org/en/master/api/#requests.Response) object with many helpful methods and attributes.

# In[ ]:


response


# In[ ]:


response.ok


# In[ ]:


dta = pd.read_json(response.content, orient='records')


# We can use the `head` method to peak at the first 5 rows of data.

# In[ ]:


dta.head()


# Of course, pandas can also load data directly from a URL, but I encourage you to reach for `requests` as often as you need it.

# ## Exercise
# 
# Try passing the URL above to `pd.read_json`. What happens?

# In[ ]:


# Type your solution here


# In[ ]:


get_ipython().run_line_magic('load', 'solutions/read_url_json.py')


# Notice how you can split a string across lines. This can be a very handy tip for improving readability, by splitting a string and putting it in parentheses, we preserve a single string.

# In[ ]:


("super "
 "long "
 "string "
 "split "
 "across "
 "lines")


# #### Pandas DataReader

# In addition to the core I/O functionality in pandas, there is also the [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/) project. This package provides programmatic access to data sets from
# 
# * Yahoo! Finance (deprecated)
# * Google Finance
# * Enigma
# * Quandl
# * FRED
# * Fama/French
# * World Bank
# * OECD
# * Eurostat
# * EDGAR Index (deprecated)
# * TSP Fund Data
# * Nasdaq Trader Symbol Definitions
# * Morningstar
# * Etc.

# #### Further Resources

# Sometimes we need to be resourceful in order to get data. Knowing how to scrape the web can really come in handy.
# 
# We're not going to go into details today, but you'll likely find libraries like [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/), [lxml](http://lxml.de/), and [mechanize](https://mechanize.readthedocs.io/en/latest/) to be helpful. 
# 
# There's also a `read_html` function in pandas that will quickly scrape HTML tables for you and put them into a DataFrame. 
