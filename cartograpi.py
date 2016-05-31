#! /usr/bin/env python

"""
Easy inspection of Python APIs

CartogrAPI wraps Python's standard introspection functions in an
easy-to-use programmatic interface, plus adds a few extra features
enabled by default (but can be disabled):

 * definitions are sorted by source-order, when possible
 * only "public" definitions are iterated, where public means:
   - identifiers don't start with `_`
   - defined locally (not just imported)
 * differences between Python 2 and 3 are smoothed over:
   - function signatures
   - unbound methods
   - qualnames

A useful feature of CartogrAPI is the `Api.build_index()` function,
which generates a dictionary description of the API. This can be
dumped to JSON from the commandline:

    $ cartograpi.py module > module-api.json

This uses the module name (which must be importable), not the filename.
For instance, CartogrAPI's API description can be generated as follows:

    $ cartograpi.py cartograpi
    {
      "cartograpi.Api.iscallable": {
        "docstring": "Return `True` if the object is callable.",
        "module": "cartograpi",
        "name": "iscallable",
        "qualname": "Api.iscallable",
        "signature": [
          "obj"
        ],
        "methodtype": "staticmethod",
        "fullname": "cartograpi.Api.iscallable",
        "type": "method",
        "islocal": true
      },
      ...
    }

Note that, being a dictionary, the generated index may not appear in
source order, but sub-definitions (e.g. the value of the "methods" key
on a class description) will be.

"""

try:
    from builtins import filter as _filter
except ImportError:
    from future_builtins import filter as _filter

from types import (
    FunctionType,
    MethodType,
    BuiltinFunctionType,
)

from inspect import (
    getmembers,
    getdoc,
    getmodule,
    getsourcelines,
    getmro,
    ismodule,
    isclass,
    isroutine,
    ismethod,
    isfunction,
    cleandoc,
)
try:
    from inspect import signature as _signature
    def _paramlist(obj):
        try:
            sig = _signature(obj)
            params = []
            for p in sig.parameters.values():
                if p.default is not p.empty:
                    params.append('{}={}'.format(
                        p.name,
                        getattr(p.default, '__name__', repr(p.default))
                    ))
                else:
                    params.append(p.name)
            return params
        except (ValueError, TypeError):
            return ['?']
except ImportError:
    from inspect import getargspec
    def _paramlist(obj):
        try:
            if not isfunction(obj) and hasattr(obj, '__func__'):
                obj = obj.__func__
            elif isclass(obj):
                obj = obj.__init__
            a, vargs, kwargs, d = getargspec(obj)
            if d:
                params = a[:-len(d)]
                params.extend('%s=%s' % (a, getattr(d, '__name__', repr(d)))
                              for a, d in zip(a[-len(d):], d))
            else:
                params = a
            if vargs:
                params.append('*' + vargs)
            if kwargs:
                params.append('**' + kwargs)
            return params
        except (ValueError, TypeError):
            pass
        return ['?']

import pkgutil
import importlib
import re
import warnings

# Python2 doesn't have list.clear()
def clearlist(lst):
    del lst[:]


global_exclude = ['<lambda>']

_ignore_docstrings = set([
    object.__init__.__doc__,
    Exception.__doc__,
    Warning.__doc__,
])


def sourceline_sort(obj):
    try:
        return (getsourcelines(obj)[1], Api.name(obj))
    except (OSError, TypeError):
        return (-1, Api.name(obj))

class Api(object):

    @staticmethod
    def build_index(mod, exclude=None):
        index = {}
        old_global_exclude = list(global_exclude)
        if exclude:
            global_exclude.extend(exclude)

        def _build_index(data, prefix):
            if data['type'] == 'module':
                prefix = ''
                qualname = ''
                fullname = data['module']
            else:
                qualname = prefix + data['name']
                prefix = qualname + '.'
                fullname = '%s.%s' % (data['module'], qualname)
            data.update([
                ('qualname', qualname),
                ('fullname', fullname),
            ])
            children = {
                'modules': Api.modules(data['obj']),
                'classes': Api.classes(data['obj']),
                'methods': Api.methods(data['obj']),
                'functions': Api.functions(data['obj']),
            }
            for key, datalist in children.items():
                if datalist:
                    data[key] = []
                    for child_data in datalist:
                        child_fullname = _build_index(child_data, prefix)
                        data[key].append(child_fullname)
            index[fullname] = data
            return fullname

        _build_index(Api.getmoduleinfo(mod), '')
        # reset global exclude (without reassignment)
        clearlist(global_exclude)
        global_exclude.extend(old_global_exclude)
        return index

    @staticmethod
    def name_cache(obj):
        cache = {}

        def _name_cache(_obj, module, prefix):
            name = Api.name(_obj)
            if ismodule(_obj):
                prefix = ''
                module = name
                qualname = ''
                fullname = module
            else:
                qualname = prefix + name
                prefix = qualname + '.'
                fullname = '%s.%s' % (module, qualname)
            cache[id(_obj)] = {
                'module': module,
                'qualname': qualname,
                'fullname': fullname,
            }
            for mod in Api.modules(_obj):
                _name_cache(mod['obj'], module, prefix)
            for cls in Api.classes(_obj):
                _name_cache(cls['obj'], module, prefix)
            for mth in Api.methods(_obj):
                _name_cache(mth['obj'], module, prefix)
            for fnc in Api.functions(_obj):
                _name_cache(fnc['obj'], module, prefix)

        _name_cache(obj, '', '')
        return cache


    ################
    # TEST FUNCTIONS


    @staticmethod
    def isspecial(obj):
        """Return `True` if *obj*'s name begins and ends with `__`."""
        name = Api.name(obj)
        return name[:2]=='__' and name[-2:]=='__'

    @staticmethod
    def isprivate(obj):
        """
        Return `True` if *obj*'s name begins with `_` but doesn't match
        isspecial(*obj*).
        """
        name = Api.name(obj)
        return name[:1] == '_' and not Api.isspecial(obj)

    @staticmethod
    def context(obj):
        """
        Return the containing context of *obj* (e.g. a module's package,
        a class's module, etc.).
        """
        return getattr(obj, '__package__', getattr(obj, '__module__', None))

    @staticmethod
    def ispackage(obj):
        """Return `True` if the object is a package."""
        return ismodule(obj) and hasattr(obj, '__path__')

    @staticmethod
    def ismodule(obj):
        """Return `True` if the object is a module."""
        return ismodule(obj)

    @staticmethod
    def isclass(obj):
        """Return `True` if the object is a class."""
        return isclass(obj)

    @staticmethod
    def isroutine(obj):
        """Return `True` if the object is a routine."""
        return isroutine(obj)

    @staticmethod
    def ismethod(obj):
        """Return `True` if the object is a method."""
        return ismethod(obj)

    @staticmethod
    def isfunction(obj):
        """Return `True` if the object is a function."""
        return isfunction(obj)

    @staticmethod
    def iscallable(obj):
        """Return `True` if the object is callable."""
        return hasattr(obj, '__call__')


    ################
    # INFO FUNCTIONS


    @staticmethod
    def getinfo(obj, prefix='', **kwargs):
        if ismodule(obj):
            return Api.getmoduleinfo(obj)
        elif isclass(obj):
            return Api.getclassinfo(obj)
        elif ismethod(obj):
            return Api.getmethodinfo(obj)
        elif isfunction(obj):
            return Api.getfunctioninfo(obj)

    @staticmethod
    def getbasicinfo(obj):
        info = {
            'obj': obj,
            'name': Api.name(obj),
            'module': Api.modulename(obj),
            'docstring': Api.docstring(obj),
        }
        if Api.iscallable(obj):
            info['signature'] = Api.signature(obj)
        return info

    @staticmethod
    def getmoduleinfo(mod, pkg=None):
        info = Api.getbasicinfo(mod)
        info.update({
            'type': 'module',
            'ispackage': Api.ispackage(mod),
        })
        return info

    @staticmethod
    def getclassinfo(cls):
        if issubclass(cls, type):
            classtype = 'metaclass'
        elif issubclass(cls, Exception):
            classtype = 'exception'
        elif issubclass(cls, Warning):
            classtype = 'warning'
        else:
            classtype = 'class'
        info = Api.getbasicinfo(cls)
        info.update({
            'type': 'class',
            'classtype': classtype,
            # 'bases': cls.__bases__,
        })
        return info


    @staticmethod
    def getmethodinfo(method, cls=None):
        if not (type(method) is MethodType or
                (type(method) in (FunctionType, BuiltinFunctionType)
                 and cls is not None)):
            raise ValueError(
                'First argument must be a method, or be a function '
                'with the second argument its binding class.'
            )
        if cls is None:
            cls = method.__self__
        name = Api.name(method)
        im_class = method_type = islocal = None
        for c in getmro(cls):
            if name in c.__dict__:
                im_class = c
                method_type = c.__dict__[name].__class__.__name__
                islocal = c is cls
                break
        info = Api.getbasicinfo(method)
        # remove implicit arg of non-static methods
        if method_type != 'staticmethod':
            # this is very hacky... no way to introspect?
            info['signature'] = info['signature'][1:]
        info.update({
            'type': 'method',
            # 'definingclass': im_class,
            'methodtype': method_type,
            'islocal': islocal,
        })
        return info

    @staticmethod
    def getfunctioninfo(func, module=None):
        info = Api.getbasicinfo(func)
        if module is not None:
            info['module'] = module
        info['type'] = 'function'
        return info


    #############################
    # CONTENT RETRIEVAL FUNCTIONS


    @staticmethod
    def members(obj, predicate=None, filter=None):
        """
        Return all member objects in *obj* that match *predicate* and
        are not filtered by *filter*. This is a wrapper for Python's
        `inspect.getmembers()` with the addition of the filter function.
        """
        return sorted(
            _filter(
                filter,
                [x for _, x in getmembers(obj, predicate)]
            ),
            key=sourceline_sort
        )

    @staticmethod
    def make_filter(
            context=None,
            exclude=None,
            isprivate=False,
            isspecial=False):
        """
        Make a function for filtering results from `members()`.
        Below, *x* is the object to be filtered.

        Args:
            context: if not None, test that the value of
                     `context(x)` is equal to *context*
            exclude: if not None, test that *x* is in *exclude*;
                     *exclude* must be a collection
            isprivate: if not None, test that the value of
                       `isprivate(x)` is equal to *isprivate*
            isspecial: if not None, test that the value of
                       `isspecial(x)` is equal to *isspecial*
        Returns:
            A function that returns `True` if all conditions are
            satisfied, or `False` if any condition fails.
        """
        def filterfunc(x):
            excludes = global_exclude + ([] if exclude is None else exclude)
            if Api.name(x) in excludes:
                return False
            if context is not None and Api.context(x) != context:
                return False
            priv = None if isprivate is None else Api.isprivate(x)
            spec = None if isspecial is None else Api.isspecial(x)
            if not ((priv==isprivate and spec==isspecial)
                    or priv==isprivate==True
                    or spec==isspecial==True):
                return False
            return True
        return filterfunc

    @staticmethod
    def packages(obj, **kwargs):
        """
        Return the non-module packages in *obj*.
        Optional keyword arguments are used to define a filter (see
        Api.make_filter()).
        """
        return Api.modules(obj, ispackage=True, **kwargs)

    @staticmethod
    def modules(obj, ispackage=None, prefix='', **kwargs):
        """
        Return the modules in the package *obj*. This returns both
        regular modules and packages by default. If *ispackage* is
        `False`, only regular modules are returned, and similarly only
        packages are returned if *ispackage* is `True`.
        Optional keyword arguments are used to define a filter (see
        Api.make_filter()).
        """
        # f = Api.make_filter(obj.__name__, **kwargs)
        # return Api.members(obj, ismodule, f)
        pkg = Api.name(obj)
        f = Api.make_filter(pkg, **kwargs)
        mods = []
        if Api.ispackage(obj):
            for _, name, ispkg in pkgutil.iter_modules(obj.__path__):
                mod = importlib.import_module('.' + name, package=pkg)
                if ispackage is None or Api.ispackage(mod)==ispkg:
                    mod.__package__ = pkg
                    mods.append(mod)
        return [
            Api.getmoduleinfo(m, pkg)
            for m in list(filter(f, filter(ismodule, mods)))
        ]

    @staticmethod
    def classes(obj, **kwargs):
        """
        Return the classes defined in *obj*.
        Optional keyword arguments are used to define a filter (see
        Api.make_filter()).
        """
        f = Api.make_filter(Api.name(obj), **kwargs)
        return [
            Api.getclassinfo(c)
            for c in Api.members(obj, isclass, f)
        ]

    @staticmethod
    def methods(cls,
                islocal=None, isclassmethod=None, isstaticmethod=None,
                **kwargs):
        """
        Return the methods defined in class *cls*. By default, all
        methods (regular, static, class, local, inherited) are returned.
        Optional keyword arguments are used to define a filter (see
        Api.make_filter()).
        """
        if not isinstance(cls, type):
            return []
            # raise ValueError("First argument must be a class.")
        f = Api.make_filter(cls.__module__, **kwargs)
        ms = []
        for m in Api.members(cls, isroutine, f):
            try:
                info = Api.getmethodinfo(m, cls=cls)
            except ValueError:
                warnings.warn(
                    'Skipping invalid method: {!r}'.format(m),
                    Warning
                )
                continue
            if (islocal in (None, info['islocal']==islocal) and
                isclassmethod in (None, info['methodtype']=='classmethod') and
                isstaticmethod in (None, info['methodtype']=='staticmethod')):
                    ms.append(info)
        return ms

    @staticmethod
    def staticmethods(obj, **kwargs):
        """
        Return the static methods defined in *obj*.
        Optional keyword arguments are used to define a filter (see
        Api.make_filter()).
        """
        return Api.methods(
            obj, isstaticmethod=True, isclassmethod=False, **kwargs
        )

    @staticmethod
    def classmethods(obj, **kwargs):
        """
        Return the class methods defined in *obj*.
        Optional keyword arguments are used to define a filter (see
        Api.make_filter()).
        """
        return Api.methods(
            obj, isclassmethod=True, isstaticmethod=False, **kwargs
        )

    @staticmethod
    def regularmethods(obj, **kwargs):
        """
        Return the regular (non-static, non-class) methods defined in
        *obj*.
        Optional keyword arguments are used to define a filter (see
        Api.make_filter()).
        """
        return Api.methods(
            obj, isclassmethod=False, isstaticmethod=False, **kwargs
        )

    @staticmethod
    def functions(obj, **kwargs):
        """
        Return the functions defined in *obj*.
        Optional keyword arguments are used to define a filter (see
        Api.make_filter()).
        """
        f = Api.make_filter(Api.name(obj), **kwargs)
        return [
            Api.getfunctioninfo(func)
            for func in Api.members(obj, isroutine, f)
        ]


    ######################
    # BASIC DATA FUNCTIONS


    @staticmethod
    def name(obj):
        """Return the name of *obj*."""
        return getattr(obj, '__name__', '')

    @staticmethod
    def modulename(obj):
        """Return the module name of *obj*."""
        if ismodule(obj):
            return Api.name(obj)
        else:
            return getattr(obj, '__module__', '')

    @staticmethod
    def qualname(obj):
        """
        Return the qualified name of *obj*.

        Unlike __qualname__ in Python 3.3+, this function returns '' for
        modules. This is so the following always returns the fullname:

        >>> (Api.modulename(obj) + '.' + Api.qualname(obj)).rstrip('.')
        """
        if ismodule(obj):
            return ''
        elif hasattr(obj, '__qualname__'):
            return obj.__qualname__
        # still don't got it; try to figure it out
        elif hasattr(obj, '__module__'):
            idx = Api.name_cache(getmodule(obj))
            if id(obj) in idx:
                return idx[id(obj)]['qualname']
        # give up
        return Api.name(obj)  # raise error instead?

    @staticmethod
    def docstring(obj, ignorecommon=True):
        """Return the docstring of *obj*."""
        ds = getdoc(obj) or ''
        if ignorecommon and ds in _ignore_docstrings: ds = ''
        return ds

    @staticmethod
    def signature(obj):
        """
        Return the Signature object for callable *obj*.
        """
        try:
            return _paramlist(obj)
        except (TypeError, ValueError):
            warnings.warn('No signature found for {!r}'.format(obj), Warning)
            return ['?']


###################
# DOCSTRING PARSING


def _Docstring_rejoin(lines):
    return cleandoc('\n'.join(lines))

def _Docstring_block(lines, pattern):
    block = []
    while lines and re.search(pattern, lines[-1]):
        block.append(lines.pop())
    return block

def _Docstring_remaining_lines(lines, indent):
    while lines and lines[-1].strip() == '':
        lines.pop()
    if lines and lines[-1].startswith(' ' * indent):
        return True
    return False

_GoogleDocstring_section_re = re.compile(
    r'(?P<indent> *)'
    r'(?P<name>'
    r'(?:keyword +)?arg(?:ument)?s?'
    r'|attributes?'
    r'|examples?'
    r'|methods?'
    r'|notes?'
    r'|(?:other +)?param(?:eter)?s?'
    r'|returns?'
    r'|raises?'
    r'|references?'
    r'|see +also'
    r'|todos?'
    r'|warn(?:ing)?s?'
    r'|yields?'
    r')'
    r' *:\s*$',  # line-ending colon
    re.U|re.I
)

_GoogleDocstring_item_re = re.compile(
    # items have a name and a colon at least
    r'(?P<indent> {2,})'  # two or more spaces (from indent level)
    r'(?P<name>\w+)\s*'
    r'(?:\((?P<meta>[^)]+)\))?\s*'
    r':'
    r'(?P<desc>.*)$',
    re.U
)

_GoogleDocstring_code_re = re.compile(
    r'(?P<indent> *)'
    '('
    r'(?P<python>>>> )'
    '|'
    r'(?P<fenced>```)\s*(?P<fenced_language>\w+)\s*$'
    ')',
    re.U
)


def GoogleDocstring(s):
    """
    Parse the Google-style docstring and return the analyzed sections.
    Sections (taken from Napoleon):
        Args (alias of Parameters)
        Arguments (alias of Parameters)
        Attributes
        Example
        Examples
        Keyword Args (alias of Keyword Arguments)
        Keyword Arguments
        Methods
        Note
        Notes
        Other Parameters
        Parameters
        Return (alias of Returns)
        Returns
        Raises
        References
        See Also
        Todo
        Warning
        Warnings (alias of Warning)
        Warns
        Yield (alias of Yields)
        Yields

    Returns:
        A list of sections, where each section is a dictionary with
        the following information:

        * `name` - section title
        * `contents` - items within a section

        Each item in `contents` is then a dictionary with the following
        information:

        * `name` - item (e.g. argument) name (possibly None)
        * `metadata` - metadata (e.g. type info) that may follow `name`
          (possibly None)
        * `text` - the textual content of the item, processed with
          Python's `inspect.cleandoc()`
    """
    lines = s.splitlines()
    lines = list(reversed(lines))  # for stack order
    return list(_GoogleDocstring_sections(lines))

def _GoogleDocstring_sections(lines):
    while lines:
        match = _GoogleDocstring_section_re.match(lines[-1])
        if match:
            yield _GoogleDocstring_named_section(lines, match)
        else:
            contents = _GoogleDocstring_contents(
                lines, _GoogleDocstring_section_re, 0
            )
            yield {'contents': list(contents)}
    assert len(lines) == 0

def _GoogleDocstring_named_section(lines, match):
    lines.pop()  # no more use for starting line
    name = match.group('name').strip()
    indent = len(match.group('indent'))
    items = []
    while _Docstring_remaining_lines(lines, indent + 1):
        match = _GoogleDocstring_item_re.match(lines[-1])
        if match:
            items.append(_GoogleDocstring_item(lines, match))
        else:
            items.extend(
                _GoogleDocstring_contents(
                    lines, _GoogleDocstring_item_re, indent + 1
                )
            )
    return {'name': name, 'contents': items}

def _GoogleDocstring_item(lines, match):
    lines.pop()  # no more use for starting line
    item = {'name': match.group('name')}
    if match.group('meta'):
        item['meta'] = match.group('meta')
    indent = len(match.group('indent'))
    desc = match.group('desc')
    desc = [desc] if desc else []
    while _Docstring_remaining_lines(lines, indent + 1):
        desc.append(lines.pop())

    item['text'] = _Docstring_rejoin(desc)
    return item


def _GoogleDocstring_contents(lines, matcher, indent):
    while _Docstring_remaining_lines(lines, indent) and \
            not matcher.match(lines[-1]):
        item = {}
        match = _GoogleDocstring_code_re.match(lines[-1])
        if match:
            ind = ' ' * len(match.group('indent'))
            content_type = 'code'
            if match.group('python'):
                item['language'] = 'python'
                block = _Docstring_block(lines, r'^%s[^ ]' % (ind,))
            elif match.group('fenced'):
                lines.pop()  # remove fenced opening
                if match.group('fenced_language'):
                    item['language'] = match.group('fenced_language')
                block = _Docstring_block(lines, r'^(%s(?!```)| *$)' % (ind,))
                if lines[-1].strip() == '```':
                    lines.pop()  # remove fenced closing
        elif lines[-1].startswith(' ' * (indent + 4)):
            ind = len(re.search(r'^\s*', lines[-1]).group(0))
            content_type = 'code'
            block = _Docstring_block(lines, r'^( {%d,}| *$)' % (ind,))
        else:
            content_type = 'text'
            block = _Docstring_block(
                lines, r'^ {%d,}[^ ]' % (indent,)
            )
        item[content_type] = _Docstring_rejoin(block)
        yield item


docstring_parsers = {
    'google': GoogleDocstring,
}

#####################
# COMMAND LINE ACCESS


def index(args):
    if args.format == 'json':
        import json
        dump = json.dumps
    elif args.format == 'yaml':
        import yaml
        dump = yaml.dump
    else:
        raise ValueError(args.format)

    mod = importlib.import_module(args.module)

    exclude = None
    if args.exclude:
        exclude = list(map(str.strip, args.exclude.split(',')))

    idx = Api.build_index(mod, exclude=exclude)

    # postprocessing

    for fullname in idx:
        if 'obj' in idx[fullname]:
            del idx[fullname]['obj']  # remove object references for json

    docstring_parse = docstring_parsers.get(args.docstring_parse)
    if docstring_parse:
        for fullname, obj in idx.items():
            if 'docstring' in obj:
                obj['doc'] = docstring_parse(obj['docstring'])

    print(dump(idx, indent=2))

def diff(args):
    import json
    idx1 = json.load(open(args.index1))
    idx2 = json.load(open(args.index2))
    allnames = set(list(idx1) + list(idx2))
    for name in sorted(allnames):
        if name in idx1 and name not in idx2:
            print('-\t' + name)
        elif name not in idx1 and name in idx2:
            print('+\t' + name)
        else:
            sig1 = idx1.get(name).get('signature')
            sig2 = idx2.get(name).get('signature')
            if sig1 != sig2:
                print('~\t' + name)
                print(' .. 1: {}({})'.format(name, ', '.join(sig1)))
                print(' .. 2: {}({})'.format(name, ', '.join(sig2)))

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Inspect the API for a Python module or package'
    )
    cmds = parser.add_subparsers()

    idx = cmds.add_parser('index')
    idx.add_argument('module', help='the module or package to inspect')
    idx.add_argument(
        '-x', '--exclude',
        help='comma-separated list of members to exclude (these are '
             'full names, so module + qualname)'
    )
    idx.add_argument(
        '-d', '--docstring-parse',
        metavar='STYLE', choices=('none', 'google'), default='none',
        help='parse docstrings in the given STYLE'
    )
    idx.add_argument(
        '-f', '--format',
        choices=('json', 'yaml'), default='json',
        help='the output file format'
    )
    idx.set_defaults(func=index)

    dif = cmds.add_parser('diff')
    dif.add_argument('index1')
    dif.add_argument('index2')
    dif.set_defaults(func=diff)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
