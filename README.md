# CartogrAPI

Easy inspection of Python APIs

# Usage

CartogrAPI works with Python 2.7 and 3.3+. It can be invoked from
the commandline to generate a JSON description of an API (see
[cartograpi.json](cartograpi.json) for an example):

```bash
$ ./cartograpi.py module > module.json
```

It can also be imported and used directly. Import the `Api` class
which defines many static methods:

```python
from cartograpi import Api

import module
class_list = Api.classes(module)
```
