#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, datetime
import base64, cStringIO

path_to_scripts = '/home/singlecell/scripts/Tamas/scTCRpy/'

def fig2hex(f, tag=None, dpi=200):
    if tag == None:
        tag = ''
    my_stringIObytes = cStringIO.StringIO()
    f.savefig(my_stringIObytes, format='png', dpi=dpi)
    my_stringIObytes.seek(0)
    s = base64.b64encode(my_stringIObytes.read())
    return '<img '+tag+' src="data:image/png;base64, ' + s + '" />\n'

def stampTime(text, before=None):
    parttime = datetime.datetime.now()
    text += ' ' + parttime.strftime("%H:%M:%S.%f")
    if before != None:
        text += ' (elapsed time: ' + str(parttime-before) + ')'
    print text
    return parttime

def readORrun(fn, forced, ifread, ifrun, iffail, parentDir=None, kwargs={}):
    if parentDir != None:
        if len(parentDir) > 0:
            if parentDir[-1] == '/':
                parentDir = parentDir[:-1]
            fn = parentDir + '/' + fn
    if fn != None:
        kwargs['fn'] = fn
        if os.path.isfile(fn):
            if forced:
                return ifrun(**kwargs)
            else:
                return ifread(**kwargs)
        else:
            if os.access(os.path.dirname(fn), os.W_OK):
                return ifrun(**kwargs)
            else:
                return iffail
    else:
        return iffail

def tabWriter(data, fn, sep='\t', linebr='\n', nan='', rowname='Rowname', rownames=None, colnames=None, header=None, addheader=True, lineformatter=None, tabformatter=None, transposed=False):
    def basic_str(l):
        return [str(x) for x in l]
    data_valid = False
    if isinstance(data, dict):
        data_valid = True
        mainitem = 'dict'
        subitem = 'singles'
        heads = data.keys()
        firstitem = data[heads[0]]
        if transposed:
            if colnames == None:
                colnames = heads
        else:
            if rownames == None:
                rownames = heads
        if isinstance(firstitem, dict):
            subitem = 'dict'
            if transposed:
                if rownames == None:
                    rownames = list(firstitem.keys())
            else:
                if colnames == None:
                    colnames = list(firstitem.keys())
        else:
            if isinstance(firstitem, list):
                subitem = 'list'
    else:
        if isinstance(data, list):
            data_valid = True
            mainitem = 'list'
            subitem = 'singles'
            firstitem = data[0]
            if isinstance(firstitem, dict):
                subitem = 'dict'
                if transposed:
                    if rownames == None:
                        rownames = list(firstitem.keys())
                else:
                    if colnames == None:
                        colnames = list(firstitem.keys())
            else:
                if isinstance(firstitem, list):
                    subitem = 'list'
    if lineformatter == None:
        lineformatter = basic_str
    def dict_of_dict():
        o = ''
        if addheader:
            o += sep.join([rowname]+colnames)+linebr
        for row in rownames:
            items = []
            for col in colnames:
                items.append(data[row][col])
            o += sep.join(lineformatter(items))+linebr
        return o
    def dict_of_list():
        o = ''
        if transposed:
            if addheader:
                o += sep.join([rowname]+colnames)+linebr
            for i in range(0, len(firstitem)):
                items = []
                for col in colnames:
                    items.append(data[col][i])
                o += sep.join(lineformatter(items))+linebr
        else:
            if addheader:
                if colnames != None:
                    if rowname != None:
                        o += sep.join([rowname]+colnames)+linebr
                    else:
                        o += sep.join(colnames)+linebr
            for row in rownames:
                o += sep.join(lineformatter([row]+data[row]))+linebr
        return o
    def dict_of_singles(rownames=rownames):
        o = ''
        if rownames == None:
            rownames = list(data.keys())
        for row in rownames:
            o += row + sep + str(data[row]) + linebr
        return o
    def list_of_dict():
        o = ''
        if transposed:
            for row in rownames:
                items = []
                for col in data:
                    if row in col:
                        items.append(col[row])
                    else:
                        items.append(nan)
                o += sep.join(lineformatter([row]+items))+linebr
        else:
            if addheader:
                o += sep.join([rowname]+colnames)+linebr
            for row in data:
                items = []
                for col in colnames:
                    if col in row:
                        items.append(row[col])
                    else:
                        items.append(nan)
                o += sep.join(lineformatter(items))+linebr
        return o
    def list_of_list():
        o = ''
        if transposed:
            N, M = 0, 0
            if colnames != None:
                M = len(colnames)
                if addheader:
                    o += sep.join([rowname]+colnames)+linebr
            else:
                M = len(data)
            if rownames != None:
                N = len(rownames)
                for i in range(0, N):
                    items = []
                    for j in range(0, M):
                        items.append(data[j][i])
                    o += sep.join(lineformatter([rownames[i]]+items))+linebr
            else:
                N = len(data[0])
                for i in range(0, N):
                    items = []
                    for j in range(0, M):
                        items.append(data[j][i])
                    o += sep.join(lineformatter(items))+linebr
        else:
            if colnames != None:
                if addheader:
                    o += sep.join([rowname]+colnames)+linebr
            if rownames != None:
                for i in range(0, len(rownames)):
                    o += sep.join(lineformatter([rownames[i]]+list(data[i])))+linebr
            else:
                for row in data:
                    o += sep.join(lineformatter(row))+linebr
        return o
    def list_of_singles():
        o = ''
        for e in data:
            o += str(e) + linebr
        return o
    method_dict = {'dict_of_dict': dict_of_dict, 'dict_of_list': dict_of_list, 'dict_of_singles': dict_of_singles, 'list_of_dict': list_of_dict, 'list_of_list': list_of_list, 'list_of_singles': list_of_singles}
    if tabformatter == None:
        if data_valid:
            tabformatter = method_dict[mainitem+'_of_'+subitem]
        else:
            stampTime('Failed saving file '+fn)
            return
    else:
        if isinstance(tabformatter, str):
            tabformatter = method_dict[tabformatter]
    f = open(fn, 'w')
    f.write(tabformatter())
    f.close()
    return

def tabReader(fn, data=None, sep=None, linebr='\n', nan='', add_nan=True, rowname='Rowname', rownames=None, colnames=None, header=None, hasheader=True, lineformatter=None, tabformatter=None, tabformat=None, transposed=False, includelist=[], evenflatter=False, floating=[], namedcols=True, uniquekeys=True, keycol=0):
    if sep == None:
        ext = fn[-4:]
        if ext == '.csv':
            sep = ','
        else:
            sep = '\t'
    def argsetter(locarg, pararg):
        ardi = {}
        positional = ['data', 'line', 'lineNum', 'firstline', 'includend', 'test_only']
        for k, v in locarg.iteritems():
            if k in pararg:
                if k not in positional:
                    ardi[k] = pararg[k]
        return ardi
    def dict_of_obj_init(data):
        if data == None:
            data = namedMatrix()
        else:
            if isinstance(data, dict):
                data = namedMatrix(aggr=data)
        return data
    def dict_of_obj_parse(data, line, lineNum, firstline, includend, test_only=False, uniquekeys=True, keycol=0, includelist=[], evenflatter=False, floating=[], nan='DROP', add_nan=True, header=None, hasheader=True):
        if test_only:
            return locals()
        if hasheader and firstline:
            firstline = False
            if header == None:
                header = []
                if len(includelist) > 0:
                    for i in includelist:
                        header.append(line[i])
                else:
                    header = line[:keycol]+line[keycol+1:]
            data.setColNames(header)
        else:
            includend, no_nan = True, True
            for i in floating:
                e = line[i]
                try:
                    e = float(e)
                except:
                    no_nan = False
                    if nan != 'DROP':
                        e = nan
                    else:
                        e = None
                line[i] = e
            dat = []
            if len(includelist) > 0:
                for i in includelist:
                    dat.append(line[i])
            else:
                dat = line[:keycol]+line[keycol+1:]
            if evenflatter:
                dat = dat[0]
            key = line[keycol]
            dat = data.matrixRow(key, dat, data)
            if includend:
                if no_nan:
                    data.add(dat)
                else:
                    if add_nan:
                        data.add(nan)
        return data, firstline, includend
    def dict_of_dict_init(data):
        if data == None:
            data = {'DeFaUlT': []}
        return data
    def dict_of_dict_parse(data, line, lineNum, firstline, includend, test_only=False, uniquekeys=True, keycol=0, includelist=[], evenflatter=False, floating=[], nan='DROP', add_nan=True, header=None, hasheader=True):
        if test_only:
            return locals()
        if hasheader and firstline:
            firstline = False
            if header == None:
                header = []
                if len(includelist) > 0:
                    for i in includelist:
                        header.append(line[i])
                else:
                    header = line[:keycol]+line[keycol+1:]
            data['DeFaUlT'] = header
        else:
            includend, no_nan = True, True
            for i in floating:
                e = line[i]
                try:
                    e = float(e)
                except:
                    no_nan = False
                    if nan != 'DROP':
                        e = nan
                    else:
                        e = None
                line[i] = e
            dat = []
            if len(includelist) > 0:
                for i in includelist:
                    dat.append(line[i])
            else:
                dat = line[:keycol]+line[keycol+1:]
            if evenflatter:
                dat = dat[0]
            dat = dict(zip(data['DeFaUlT'], dat))
            key = line[keycol]
            if includend:
                if no_nan:
                    if uniquekeys:
                        data[key] = dat
                    else:
                        if key not in data:
                            data[key] = []
                        data[key].append(dat)
                else:
                    if add_nan:
                        if uniquekeys:
                            data[key] = dat
                        else:
                            if key not in data:
                                data[key] = []
                            data[key].append(dat)
        return data, firstline, includend
    def dict_of_list_init(data):
        if data == None:
            data = {}
        return data
    def dict_of_list_parse(data, line, lineNum, firstline, includend, test_only=False, uniquekeys=True, keycol=0, includelist=[], evenflatter=False, floating=[], nan='DROP', add_nan=True):
        if test_only:
            return locals()
        includend, no_nan = True, True
        for i in floating:
            e = line[i]
            try:
                e = float(e)
            except:
                no_nan = False
                if nan != 'DROP':
                    e = nan
                else:
                    e = None
            line[i] = e
        dat = []
        if len(includelist) > 0:
            for i in includelist:
                dat.append(line[i])
        else:
            dat = line[:keycol]+line[keycol+1:]
        if evenflatter:
            dat = dat[0]
        key = line[keycol]
        if includend:
            if no_nan:
                if uniquekeys:
                    data[key] = dat
                else:
                    if key not in data:
                        data[key] = []
                    data[key].append(dat)
            else:
                if add_nan:
                    if uniquekeys:
                        data[key] = dat
                    else:
                        if key not in data:
                            data[key] = []
                        data[key].append(dat)
        return data, firstline, includend
    def list_of_list_init(data):
        if data == None:
            data = []
        return data
    def list_of_list_parse(data, line, lineNum, firstline, includend, evenflatter=evenflatter, test_only=False):
        if test_only:
            return locals()
        if evenflatter:
            data.append(line[0])
        else:
            data.append(line)
        return data, firstline, includend
    method_dict = {'dict_of_obj': [dict_of_obj_init, dict_of_obj_parse],
    'dict_of_dict': [dict_of_dict_init, dict_of_dict_parse],
    'dict_of_list': [dict_of_list_init, dict_of_list_parse],
    'list_of_list': [list_of_list_init, list_of_list_parse]
    }
    init_in_dict, parse_in_dict = True, True
    if tabformatter != None:
        init_in_dict = False
        datinit = tabformatter
    if lineformatter != None:
        parse_in_dict = False
        datparse = lineformatter
    if tabformat in method_dict:
        if init_in_dict:
            datinit = method_dict[tabformat][0]
        if parse_in_dict:
            datparse = method_dict[tabformat][1]
    else:
        if init_in_dict:
            datinit = list_of_list_init
        if parse_in_dict:
            datparse = list_of_list_parse
    data = datinit(data)
    passargs = argsetter(datparse(None, None, None, None, None, test_only=True), locals())
    with open(fn) as f:
        firstline, includend = True, False
        lineNum = 0
        for line in f:
            line = line.split(linebr)[0]
            line = line.split('\r')[0]
            line = line.replace('"', '')
            line = line.split(sep)
            data, firstline, includend = datparse(data, line, lineNum, firstline, includend, **passargs) 
            lineNum += 1
    if isinstance(data, dict):
        if 'DeFaUlT' in data:
            del data['DeFaUlT']
    return data

class namedMatrix:
    def __init__(self, unique=True, aggr={}):
        self.data = aggr
        self.order = list(aggr.keys())
        self.colnames = {}
        self.N = len(self.colnames)
        self.unique = unique
        return
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self.order)):
            name = self.order[i]
            yield i, name, self.data[name]
    
    def rows(self):
        return self.data.keys()
    
    def setColNames(self, names):
        for i in range(0, len(names)):
            self.colnames[names[i]] = i
        self.N = len(self.colnames)
        return
    
    def add(self, row):
        key = row.key
        self.data[key] = row
        if key not in self.order:
            self.order.append(key)
        return

    class matrixRow:
        def __init__(self, key, data, parent):
            N = parent.N
            data = data[:N]
            if len(data) < N:
                for i in range(len(data, N)):
                    data.append('')
            if parent.unique:
                self.data = tuple(data)
            else:
                if key not in parent.data:
                    self.data = []
                    for i in range(0, N):
                        self.data.append([data[i]])
                else:
                    self.data = parent.data[key].data
                    for i in range(0, N):
                        self.data[i].append(data[i])
            self.key = key
            self.keys = parent.colnames
            return
        
        def __getitem__(self, key):
            e = self.keys[key]
            return self.data[e]

class objectContainer:
    def __init__(self):
        self.data = {}
        self.order = []
        return
    
    def __str__(self):
        return str(len(self.order)) + 'items'
    
    def __len__(self):
        return len(self.order)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        else:
            if isinstance(key, int):
                return self.data[self.order[key]]
            else:
                return self.data[key.label]

    def __iter__(self):
        for i in range(len(self.order)):
            name = self.order[i]
            yield i, name, self.data[name]
    
    def __setitem__(self, key, item):
        if key not in self.data:
            self.order.append(key)
            self.data[key] = item
        return

    def add(self, sample):
        print 'This is a prototype class!!! The "add" function has to be miplemented for individual use cases!'
        return